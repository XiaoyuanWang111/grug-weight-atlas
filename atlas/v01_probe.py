"""Are the values that Campaign 2's dare-ties-02 wrote still present in a later checkpoint?

For the first two support rows of the routed-expert gate_proj in layers 0, 12 and 25 it reads Base (Step12), the
NUPA donor (Step92), the merge (Campaign2-DARE-TIES-02) and a later checkpoint L. Positions of a row are split by
what the merge did there:
  N  merge = bit-exact Base + 5 x (Step92 - Base) and differs from Base   (the NUPA-delta positions)
  U  merge = Base                                                         (untouched positions of the same row)
For each row it prints: the share of N in the row; on N, the share of L's stored values equal to the merge's, the
displacement |L - merge| relative to |merge - Base|, and the regression coefficient of (L - Base) on (merge - Base)
(1 = the written delta is fully there); on U, the share of L's values equal to Base and |L - Base| / |Base| in percent
(how far later training moved untouched weights).

The recipe is campaign2-recipes/dare-ties-02.json of the Hugging Face dataset
open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09; the script uses its "support" and its density.

Usage: python atlas/v01_probe.py <recipe json> <later repo> [<later repo> ...]

Network: the tensor index of every repo (st_index.index, cached), then per later checkpoint one range request per row
and repo at the commit the index records: 6 rows x 4 repos, 6,553,600 bytes per bfloat16 row and 13,107,200 bytes
per float32 row.
"""

import json
import sys
from pathlib import Path

import numpy as np

import st_index

BASE = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
NUPA = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"
MERGE = "open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10"


def row(repo, tensor, r):
    """Expert row r of one routed-expert tensor as float64, and the stored dtype."""
    idx = st_index.index(repo)
    entry = idx[tensor]
    if entry["dtype"] not in ("BF16", "F32"):
        raise KeyError(entry["dtype"])
    per_row = int(np.prod(entry["shape"][1:]))
    raw = st_index.read_elements(repo, entry, r * per_row, per_row, idx["__repo__"]["sha"])
    return st_index.to_float32(raw, entry["dtype"]).astype(np.float64), entry["dtype"]


def to_bf16_value(x):
    """x rounded to bfloat16 (nearest, ties to even), as float64."""
    return st_index.bf16_to_float32(st_index.float32_to_bf16(x)).astype(np.float64)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    later_repos = sys.argv[2:]
    support = {entry["name"]: entry["rows"] for entry in recipe["support"]}
    k_dare = 1 / recipe["parameters"]["density"]
    for later in later_repos:
        print(f"\nlater checkpoint {later} (revision {st_index.index(later)['__repo__']['sha']})")
        print(
            f"{'tensor':<44}{'row':>4}{'dtype':>6} | N share of row {'':<2}{'L = merge on N':>15}{'|L-M|/|M-B| N':>15}"
            f"{'coef on N':>10} |{'L = Base on U':>14}{'|L-B|/|B| U %':>14}"
        )
        for layer in (0, 12, 25):
            tensor = f"model.layers.{layer}.mlp.experts.gate_proj.weight"
            for r in support[tensor][:2]:
                (b, _), (n, _), (m, _) = row(BASE, tensor, r), row(NUPA, tensor, r), row(MERGE, tensor, r)
                lat, dtype = row(later, tensor, r)
                on_n = (m != b) & (to_bf16_value(b + k_dare * (n - b)) == m)
                on_u = m == b
                dm, dl = (m - b)[on_n], (lat - b)[on_n]
                coef = float((dm * dl).sum() / (dm * dm).sum())
                resid = float(np.sqrt(((lat - m)[on_n] ** 2).sum() / (dm * dm).sum()))
                moved_u = 100 * float(np.sqrt(((lat - b)[on_u] ** 2).sum() / (b[on_u] ** 2).sum()))
                print(
                    f"{tensor.replace('model.', ''):<44}{r:>4}{dtype:>6} | {100 * on_n.mean():>13.2f}%  "
                    f"{100 * (lat[on_n] == m[on_n]).mean():>14.2f}%{resid:>15.4f}{coef:>10.4f} |"
                    f"{100 * (lat[on_u] == b[on_u]).mean():>13.2f}%{moved_u:>14.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
