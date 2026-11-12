"""Build the tensor patch that turns the public half-strength merge of marin#9984 into a cell of its 2 x 2 test.

#9984 merged B + a (E - B) + b (E2 - B) with B = Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92,
E = CruxEval-Expert-E-Step272 and E2 = MMLUPro-Expert-E2-Step254. a = b = 0.5 is public (Merge-TA0.5); a = b = 1.0
was withheld. One cell of the 2 x 2 holds the attention GatedNorm projections and the attention gate (78 tensors) at
a = b = 1.0 and every other tensor at 0.5: it is Merge-TA0.5 with these 78 tensors replaced. The 26 router weights,
the other 2-D tensors of the plain-Adam optimizer group, are written to the same file as a second group. The script

  1. reads the 104 tensors of B, E, E2, Merge-TA0.5 and Merge-TIES-d0.5 at the commits the index records,
  2. recomputes the half-strength merge with the arithmetic of marin.merging.arithmetic.merge_tensor
     (float32, updates scaled then summed, base added last, cast to bfloat16) and counts stored values that differ
     from the public Merge-TA0.5,
  3. recomputes TIES (density 0.5, a = b = 1) the same way and counts stored values that differ from the public
     Merge-TIES-d0.5 (ties among equal magnitudes at the top-k cut make an exact match unlikely),
  4. writes the full-strength values of the 104 tensors to <out dir>/gates_full_strength.safetensors with
     manifest.json (source revisions, sha256, the two tensor groups, displacement per family).

Usage: python atlas/gate_patch.py [<out dir, default patch_2x2 in the repository root>]
       (the printed table is patch_2x2/build_log.txt)

Network: the tensor index of the five repos (st_index.index, cached), then one range request per tensor and repo:
520 requests, 35,409,920 bfloat16 values per repo, about 354 MB in all.
"""

import hashlib
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import st_index

PREFIX = "open-athena/Snowball-67B-A2B-"
REPOS = {
    "B": PREFIX + "10T-Mixed-RLVR-Sync-Step92",
    "E": PREFIX + "CruxEval-Expert-E-Step272",
    "E2": PREFIX + "MMLUPro-Expert-E2-Step254",
    "TA": PREFIX + "Merge-TA0.5",
    "TIES": PREFIX + "Merge-TIES-d0.5",
}
LAYERS = 26
GATE_FAMILIES = ("attn_gated_norm.down_proj.weight", "attn_gated_norm.up_proj.weight", "self_attn.attn_gate.weight")
# The router weight is the other 2-D tensor family of the plain-Adam group; it goes into the same file so that
# the whole group can be swapped, and is listed apart in the manifest.
FAMILIES = (*GATE_FAMILIES, "mlp.router.weight")
NAMES = [f"model.layers.{layer}.{family}" for layer in range(LAYERS) for family in FAMILIES]


def task_arithmetic(base, donors, coefficients):
    """bfloat16 bits of base + sum_i c_i (donor_i - base), computed in float32 like merge_tensor."""
    result = np.zeros_like(base)
    for donor, c in zip(donors, coefficients, strict=True):
        result = result + (donor - base) * np.float32(c)
    return st_index.float32_to_bf16(result * np.float32(1.0) + base)


def ties(base, donors, coefficients, density):
    """bfloat16 bits of TIES: keep the top `density` share of each update by magnitude, scale it by its coefficient,
    elect the sign of the sum, and average the agreeing updates by their coefficient mass."""
    updates = []
    for donor, c in zip(donors, coefficients, strict=True):
        update = donor - base
        count = max(1, int(np.ceil(update.size * density)))
        cut = np.partition(np.abs(update), update.size - count)[update.size - count]
        update = np.where(np.abs(update) >= cut, update, np.float32(0)) * np.float32(c)
        updates.append(update.astype(np.float32))
    total = np.zeros_like(base)
    for update in updates:
        total = total + update
    elected = np.sign(total)
    result = np.zeros_like(base)
    mass = np.zeros_like(base)
    for update, c in zip(updates, coefficients, strict=True):
        agree = (np.sign(update) == elected) & (update != 0)
        result = result + update * agree
        mass = mass + agree.astype(np.float32) * np.float32(c)
    result = result / np.where(mass == 0, np.float32(1), mass)
    return st_index.float32_to_bf16(result + base)


def write_bf16_safetensors(path, tensors, shapes, metadata):
    """Write uint16 bfloat16 bit arrays in name order: 8-byte little-endian header length, JSON header padded with
    spaces to a multiple of 8 bytes, then the data."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        data = tensors[name].tobytes()
        header[name] = {"dtype": "BF16", "shape": shapes[name], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    header["__metadata__"] = metadata
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with Path(path).open("wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for data in blobs:
            f.write(data)


def shown(path):
    """A path as printed: relative to the repository root when it lies inside it."""
    try:
        return Path(path).resolve().relative_to(st_index.ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else st_index.ROOT / "patch_2x2"
    index = {key: st_index.index(repo) for key, repo in REPOS.items()}
    sha = {key: index[key]["__repo__"]["sha"] for key in REPOS}
    for key in REPOS:
        missing = [n for n in NAMES if n not in index[key]]
        if missing:
            raise SystemExit(f"{key}: {len(missing)} tensors missing, first {missing[0]}")

    def fetch(job):
        key, name = job
        entry = index[key][name]
        if entry["dtype"] != "BF16":
            raise SystemExit(f"{key} {name}: stored dtype {entry['dtype']}, expected BF16")
        raw = st_index.range_read(REPOS[key], entry["file"], entry["start"], entry["end"], sha[key])
        return key, name, np.frombuffer(raw, dtype=np.uint16).copy()

    jobs = [(key, name) for name in NAMES for key in REPOS]
    stored = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done, (key, name, bits) in enumerate(pool.map(fetch, jobs), 1):
            stored[(key, name)] = bits
            if done % 60 == 0:
                print(f"read {done}/{len(jobs)}", flush=True)

    stats = {
        f: {
            "n": 0,
            "ta_diff": 0,
            "ties_diff": 0,
            "b2": 0.0,
            "d2": {k: 0.0 for k in ("E", "E2", "TA", "TIES", "FULL")},
            "worst": (0.0, None),
        }
        for f in FAMILIES
    }
    patch = {}
    for name in NAMES:
        family = next(f for f in FAMILIES if name.endswith(f))
        layer = int(name.split(".")[2])
        base, e, e2 = (st_index.bf16_to_float32(stored[(k, name)]) for k in ("B", "E", "E2"))
        half = task_arithmetic(base, [e, e2], [0.5, 0.5])
        ties_bits = ties(base, [e, e2], [1.0, 1.0], 0.5)
        full = task_arithmetic(base, [e, e2], [1.0, 1.0])
        patch[name] = full
        s = stats[family]
        s["n"] += base.size
        s["ta_diff"] += int((half != stored[("TA", name)]).sum())
        s["ties_diff"] += int((ties_bits != stored[("TIES", name)]).sum())
        b2 = float((base.astype(np.float64) ** 2).sum())
        s["b2"] += b2
        for key, values in (
            ("E", e),
            ("E2", e2),
            ("TA", st_index.bf16_to_float32(stored[("TA", name)])),
            ("TIES", st_index.bf16_to_float32(stored[("TIES", name)])),
            ("FULL", st_index.bf16_to_float32(full)),
        ):
            d2 = float(((values.astype(np.float64) - base) ** 2).sum())
            s["d2"][key] += d2
            if key == "FULL":
                rel = 100 * (d2 / b2) ** 0.5
                if rel > s["worst"][0]:
                    s["worst"] = (rel, layer)

    print(
        f"\n{'family':<36}{'scalars':>11}{'TA0.5 differ':>14}{'TIES differ':>13} |{'E':>7}{'E2':>7}{'TA0.5':>7}"
        f"{'TIES':>7}{'full':>7}{'worst layer':>14}"
    )
    for family, s in stats.items():
        rel = {k: 100 * (v / s["b2"]) ** 0.5 for k, v in s["d2"].items()}
        print(
            f"{family:<36}{s['n']:>11,}{s['ta_diff']:>14,}{s['ties_diff']:>13,} |{rel['E']:7.2f}{rel['E2']:7.2f}"
            f"{rel['TA']:7.2f}{rel['TIES']:7.2f}{rel['FULL']:7.2f}{s['worst'][0]:9.2f} L{s['worst'][1]}"
        )
    print("displacement columns: |x - B| / |B| in percent; 'full' is the stored (rounded) full-strength value")

    out_dir.mkdir(exist_ok=True)
    path = out_dir / "gates_full_strength.safetensors"
    metadata = {"format": "pt", "description": f"B + 1.0 (E - B) + 1.0 (E2 - B) for {len(patch)} tensors"}
    write_bf16_safetensors(path, patch, {name: index["B"][name]["shape"] for name in patch}, metadata)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "file": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "tensors": len(patch),
        "groups": {
            "attention_gates": [n for n in sorted(patch) if n.endswith(GATE_FAMILIES)],
            "router_weights": [n for n in sorted(patch) if n.endswith("mlp.router.weight")],
        },
        "formula": "bfloat16(float32(B) + 1.0 * (float32(E) - float32(B)) + 1.0 * (float32(E2) - float32(B)))",
        "sources": {key: {"repo": REPOS[key], "revision": sha[key]} for key in ("B", "E", "E2")},
        "apply_to": {"repo": REPOS["TA"], "revision": sha["TA"]},
        "check": {
            f: {
                "scalars": s["n"],
                "recomputed_TA0.5_values_that_differ_from_public": s["ta_diff"],
                "recomputed_TIES_values_that_differ_from_public": s["ties_diff"],
                "displacement_percent": {k: round(100 * (v / s["b2"]) ** 0.5, 4) for k, v in s["d2"].items()},
            }
            for f, s in stats.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {shown(path)} ({path.stat().st_size:,} bytes, sha256 {digest[:16]}...) and manifest.json")


if __name__ == "__main__":
    main()
