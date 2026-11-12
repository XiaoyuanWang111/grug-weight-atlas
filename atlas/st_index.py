"""Shared helpers: tensor index of a safetensors checkpoint on Hugging Face, throttled range reads, bfloat16
conversion, and the loader of an atlas run.

index(repo) returns {tensor name: {"file", "dtype", "shape", "start", "end"}} with absolute byte offsets in the file,
plus "__repo__": {"sha", "files"}, and caches it under <repository root>/index/<repo, "/" written as "__">.json.
read_elements(repo, entry, offset, count) fetches a run of elements of one tensor; to_float32 decodes them;
read_bf16_bits returns the raw bit patterns of a BF16 tensor. bf16_to_float32 and float32_to_bf16 convert between
float32 values and bfloat16 bit patterns (uint16); the encoder rounds to nearest, ties to even. load_atlas reads
the meta.json and units.jsonl of an atlas run.

Network: index() makes one request to the Hugging Face model API (file list and commit) and two range requests per
safetensors file (header length, header); range_read() makes one range request to
https://huggingface.co/<repo>/resolve/<revision>/<file>. Requests start at most once every 0.2 s across threads, and
an HTTP 429 answer is retried after the wait that Hugging Face announces.
"""

import json
import re
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
UA = {"User-Agent": "status-check"}
DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
    "F64": np.float64,
    "BOOL": np.bool_,
}
ITEM = {name: np.dtype(numpy_type).itemsize for name, numpy_type in DTYPES.items()}  # bytes per stored element

MIN_INTERVAL = 0.2  # seconds between request starts, across threads: at most 1500 requests per 5 minutes
_gate = threading.Lock()
_next_start = [0.0]


def _wait_turn():
    with _gate:
        now = time.time()
        start = max(now, _next_start[0])
        _next_start[0] = start + MIN_INTERVAL
    if start > now:
        time.sleep(start - now)


def _get(url, headers=None, timeout=180, tries=10):
    last = None
    for attempt in range(tries):
        _wait_turn()
        try:
            request = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last = error
            if error.code == 429:
                # Hugging Face answers 429 with the seconds left in its 5-minute window, in Retry-After or RateLimit
                left = error.headers.get("Retry-After") or ""
                match = re.search(r"t=(\d+)", error.headers.get("RateLimit") or "")
                wait = float(left) if left.isdigit() else float(match.group(1)) if match else 60.0 * (attempt + 1)
                time.sleep(min(wait + 5, 330))
            else:
                time.sleep(2 + 3 * attempt)
        except Exception as error:
            last = error
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{url}: {last}")


def range_read(repo, fname, start, end, revision="main"):
    """Bytes [start, end) of one file of a model repo."""
    return _get(f"https://huggingface.co/{repo}/resolve/{revision}/{fname}", {"Range": f"bytes={start}-{end - 1}"})


def index(repo, revision="main"):
    cache = INDEX_DIR / (repo.replace("/", "__") + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    info = json.loads(_get(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))
    out = {"__repo__": {"sha": info.get("sha"), "files": {}}}
    for sib in sorted(info["siblings"], key=lambda s: s["rfilename"]):
        fname = sib["rfilename"]
        if not fname.endswith(".safetensors"):
            continue
        n = struct.unpack("<Q", range_read(repo, fname, 0, 8, revision))[0]
        header = json.loads(range_read(repo, fname, 8, 8 + n, revision))
        out["__repo__"]["files"][fname] = sib.get("size")
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            out[name] = {
                "file": fname,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "start": 8 + n + a,
                "end": 8 + n + b,
            }
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def read_elements(repo, entry, offset, count, revision="main"):
    """Raw bytes of elements [offset, offset + count) of the tensor that an index entry describes."""
    item = ITEM[entry["dtype"]]
    start = entry["start"] + offset * item
    return range_read(repo, entry["file"], start, start + count * item, revision)


def read_bf16_bits(repo, name, start=0, count=None):
    """bfloat16 bit patterns, as int32, of `count` elements of a BF16 tensor from element `start`; count None reads
    as many elements as the tensor holds (the whole tensor when start is 0)."""
    e = index(repo)[name]
    assert e["dtype"] == "BF16", (repo, e["dtype"])
    n = int(np.prod(e["shape"])) if count is None else count
    return np.frombuffer(read_elements(repo, e, start, n), dtype=np.uint16).astype(np.int32)


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns (any integer array holding values 0..65535) to float32: the upper 16 bits."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to bfloat16 (nearest, ties to even), returned as uint16 bit patterns."""
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def to_float32(raw: bytes, dtype: str) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=DTYPES[dtype])
    if dtype == "BF16":
        return bf16_to_float32(arr)
    return arr.astype(np.float32)


def read(repo, entry, revision="main"):
    """One whole tensor as float32."""
    return to_float32(range_read(repo, entry["file"], entry["start"], entry["end"], revision), entry["dtype"])


def load_atlas(root):
    """meta.json and the rows of units.jsonl of an atlas run directory."""
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    lines = (root / "units.jsonl").read_text(encoding="utf-8").split("\n")
    return meta, [json.loads(line) for line in lines if line.strip()]
