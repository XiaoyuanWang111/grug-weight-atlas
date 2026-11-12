"""Per-unit relative displacement of the sampled routed experts in an atlas run, for one child.

Prints, per layer and projection, |child - parent| / |parent| in percent for each sampled expert (with the parent's
norm of that unit), then the pooled value (root of the summed squares) and the plain mean over units.

Usage: python atlas/expert_units.py <atlas dir> <substring of the child repo>

Network: none; it reads <atlas dir>/meta.json and units.jsonl.
"""

import collections
import sys

import st_index


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    meta, rows = st_index.load_atlas(sys.argv[1])
    child = next(i for i, c in enumerate(meta["children"]) if sys.argv[2] in c)
    table = collections.defaultdict(list)
    d2 = p2 = 0.0
    values = []
    for r in rows:
        if "mlp.experts" not in r["component"]:
            continue
        rel = 100 * r["children"][child]["delta_norm"] / r["parent_norm"]
        expert = r["unit"].rsplit("[", 1)[1].rstrip("]")
        table[(r["layer"], r["component"].split(".")[-2])].append((int(expert), rel, r["parent_norm"]))
        d2 += r["children"][child]["delta_norm"] ** 2
        p2 += r["parent_norm"] ** 2
        values.append(rel)
    print(f"child {meta['children'][child]} against {meta['parent']}")
    for (layer, proj), items in sorted(table.items()):
        items.sort()
        print(f"L{layer:<3}{proj:<10}" + " ".join(f"[{e}] {rel:5.2f} (|p| {norm:5.1f})" for e, rel, norm in items))
    print(
        f"pooled {100 * (d2 / p2) ** 0.5:.3f}% | mean over {len(values)} units {sum(values) / len(values):.3f}% | "
        f"min {min(values):.2f}% max {max(values):.2f}%"
    )


if __name__ == "__main__":
    main()
