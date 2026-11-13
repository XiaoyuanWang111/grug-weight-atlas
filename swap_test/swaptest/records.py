"""Crash-tolerant files: append-only JSONL with one flushed line per record, and atomic JSON snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a JSONL file; a final line without its newline (a torn write) is ignored."""
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    return [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def drop_torn_tail(path: Path) -> None:
    """Cut a final line that lacks its newline, so the next append starts on a fresh line."""
    if not path.exists():
        return
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep != len(data):
        with path.open("r+b") as f:
            f.truncate(keep)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record and force it to disk before returning."""
    append_jsonl_many(path, [record])


def append_jsonl_many(path: Path, records: list[dict[str, Any]]) -> None:
    """Append records with one write and force them to disk before returning."""
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        f.flush()
        os.fsync(f.fileno())


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace `path` with the JSON of `obj` in one rename, so readers never see a partial file."""
    _write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    _write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
