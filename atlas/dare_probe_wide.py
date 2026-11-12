"""What does Campaign 2's DARE-TIES merge dare-ties-02 store at the scalars it changed?

For each of the three routed-expert projections in layers 0, 12 and 25 it takes the first two rows (experts) of the
recipe's support and the first row outside it, reads that expert row from Base (Step12), the two donors
(Step92 = NUPA, Step20 = BFCL) and the merge, and splits the stored values that differ from Base into: bit-exact
Base + k x (NUPA - Base), Base + k x (BFCL - Base), Base + k x the mean of both deltas, with k = 1 / density (5),
then other multiples (k / 2, 1, 0.5, 2 k) of one delta, and unexplained. It also prints each row's displacement
|x - Base| / |Base| in percent for both donors and the merge.

The recipe is campaign2-recipes/dare-ties-02.json of the Hugging Face dataset
open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09; the script uses its "support" (rows per tensor)
and "parameters" (density, coefficients).

Usage: python atlas/dare_probe_wide.py <recipe json> [<output json, default results/dare_probe/dare_probe_wide.json>]

Network: the tensor index of the four repos (st_index.index, cached), then one range request per expert row and
repo, at the commit the index records: 27 rows x 4 repos, 6,553,600 bytes each.
"""

import json
import sys
from pathlib import Path

import numpy as np

import st_index

BASE = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
NUPA = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"
BFCL = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04"
MERGE = "open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10"


def row(repo, tensor, r):
    """bfloat16 bits of expert row r of one routed-expert tensor."""
    idx = st_index.index(repo)
    entry = idx[tensor]
    if entry["dtype"] != "BF16":
        raise SystemExit(f"{repo} {tensor}: {entry['dtype']}")
    per_row = int(np.prod(entry["shape"][1:]))
    raw = st_index.read_elements(repo, entry, r * per_row, per_row, idx["__repo__"]["sha"])
    return np.frombuffer(raw, dtype=np.uint16)


def f64(u):
    return st_index.bf16_to_float32(u).astype(np.float64)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else st_index.ROOT / "results" / "dare_probe" / "dare_probe_wide.json"
    )
    support = {entry["name"]: entry["rows"] for entry in recipe["support"]}
    density = recipe["parameters"]["density"]
    k_dare = 1 / density

    results = []
    print(
        f"merge {MERGE}\nrecipe: density {density}, coefficients {recipe['parameters']['coefficients']}, "
        f"support rows in {len(support)} tensors, {sum(len(v) for v in support.values())} rows in all"
    )
    print(
        f"{'tensor':<46}{'row':>4}{'in support':>11}{'changed %':>10}{'5x NUPA %':>10}{'5x BFCL %':>10}"
        f"{'5x mean %':>10}{'other k %':>10}{'unexpl.':>8} |{'NUPA':>7}{'BFCL':>7}{'merge':>7}"
    )
    for layer in (0, 12, 25):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensor = f"model.layers.{layer}.mlp.experts.{proj}.weight"
            rows_in = support.get(tensor, [])
            outside = next(i for i in range(256) if i not in rows_in)
            for r, inside in [*((x, True) for x in rows_in[:2]), (outside, False)]:
                b, n, f, m = (row(repo, tensor, r) for repo in (BASE, NUPA, BFCL, MERGE))
                changed = m != b
                fb, fn, ff, fm = f64(b), f64(n), f64(f), f64(m)

                def rel(x, fb=fb):
                    return 100 * float(np.sqrt(((x - fb) ** 2).sum() / (fb**2).sum()))

                counts = {"nupa5": 0, "bfcl5": 0, "mean5": 0, "other": 0}
                left = changed.copy()
                for key, predicted in (
                    ("nupa5", fb + k_dare * (fn - fb)),
                    ("bfcl5", fb + k_dare * (ff - fb)),
                    ("mean5", fb + k_dare * ((fn - fb) + (ff - fb)) / 2),
                ):
                    hit = left & (st_index.float32_to_bf16(predicted.astype(np.float32)) == m)
                    counts[key] = int(hit.sum())
                    left &= ~hit
                for donor in (fn, ff):
                    for k in (k_dare / 2, 1.0, 0.5, 2 * k_dare):
                        hit = left & (st_index.float32_to_bf16((fb + k * (donor - fb)).astype(np.float32)) == m)
                        counts["other"] += int(hit.sum())
                        left &= ~hit
                total = int(changed.sum())

                def pct(v, total=total):
                    return 100 * v / total if total else 0.0

                results.append(
                    {
                        "tensor": tensor,
                        "row": r,
                        "in_support": inside,
                        "scalars": int(b.size),
                        "changed": total,
                        **counts,
                        "unexplained": int(left.sum()),
                        "donor_differs_nupa": int((n != b).sum()),
                        "donor_differs_bfcl": int((f != b).sum()),
                        "displacement_percent": {"nupa": rel(fn), "bfcl": rel(ff), "merge": rel(fm)},
                    }
                )
                print(
                    f"{tensor.replace('model.', ''):<46}{r:>4}{'yes' if inside else 'no':>11}"
                    f"{100 * total / b.size:>10.2f}{pct(counts['nupa5']):>10.2f}{pct(counts['bfcl5']):>10.2f}"
                    f"{pct(counts['mean5']):>10.2f}{pct(counts['other']):>10.2f}{int(left.sum()):>8} |"
                    f"{rel(fn):>7.2f}{rel(ff):>7.2f}{rel(fm):>7.2f}",
                    flush=True,
                )
    inside = [x for x in results if x["in_support"]]
    tot = sum(x["changed"] for x in inside)
    print(
        f"\nsupport rows read: {len(inside)}; changed values {tot:,} of {sum(x['scalars'] for x in inside):,} "
        f"({100 * tot / sum(x['scalars'] for x in inside):.2f}%)"
    )
    for key, label in (
        ("nupa5", "Base + 5 x (NUPA - Base)"),
        ("bfcl5", "Base + 5 x (BFCL - Base)"),
        ("mean5", "Base + 5 x mean of both"),
        ("other", "other multiples of one delta"),
        ("unexplained", "unexplained"),
    ):
        v = sum(x[key] for x in inside)
        print(f"  {label:<30}{v:>12,} {100 * v / tot:6.2f}%")
    print(
        "rows outside the support with any changed value:",
        sum(1 for x in results if not x["in_support"] and x["changed"]),
    )
    out_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
