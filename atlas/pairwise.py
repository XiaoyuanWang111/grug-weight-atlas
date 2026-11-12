"""Pairwise distances between all checkpoints of an atlas run (the parent and every child), by component group.

For each group it prints two matrices over [parent, children...]:
  1. relative distance |a - b| / |parent| in percent, from the Gram matrix of the children's deltas:
     |a - b|^2 = <da, da> + <db, db> - 2 <da, db>, with da = a - parent;
  2. share of stored scalars that differ between a and b, in percent (parent row: "changed"; child pairs: from the
     count of equal stored values, which atlas runs without that count leave at 100.00).
Routed experts are a sample; they are weighted so that the group counts at its true size in the "all" matrices.

Usage: python atlas/pairwise.py <atlas dir> [<expert weight, default 32>]

Network: none; it reads <atlas dir>/meta.json and units.jsonl.
"""

import re
import sys

import numpy as np

import st_index

GROUPS = [
    "attention gate + GatedNorm",
    "router",
    "attention q k v o",
    "shared expert",
    "routed experts",
    "other norms",
    "embedding + head (last rows)",
]


def short(repo):
    name = re.sub(r"^.*?/(Snowball|Grug)-67B-A2B-", "", repo)
    name = re.sub(r"-2026\.\d\d\.\d\d.*$", "", name)
    return name.replace("Antidoom-RLVR1-", "").replace("10T-Mixed-", "").replace("Merge-", "M-")[:16]


def group(component):
    if "attn_gated_norm" in component or "attn_gate" in component:
        return "attention gate + GatedNorm"
    if "mlp.router" in component:
        return "router"
    if "self_attn" in component:
        return "attention q k v o"
    if "shared_expert" in component:
        return "shared expert"
    if "mlp.experts" in component:
        return "routed experts"
    if "embed_tokens" in component or "lm_head" in component:
        return "embedding + head (last rows)"
    return "other norms"


def accumulate(rows, n, expert_weight):
    """Per group: weighted size, parent energy, Gram matrix, changed counts and equal-value counts of n children."""
    acc = {
        g: {"size": 0.0, "parent": 0.0, "gram": np.zeros((n, n)), "changed": np.zeros(n), "equal": np.zeros((n, n))}
        for g in [*GROUPS, "all"]
    }
    for r in rows:
        w = expert_weight if "mlp.experts" in r["component"] else 1.0
        for g in (group(r["component"]), "all"):
            a = acc[g]
            a["size"] += w * r["size"]
            a["parent"] += w * r["parent_norm"] ** 2
            a["gram"] += w * np.array(r["gram"])
            a["changed"] += w * np.array([c["changed"] * r["size"] for c in r["children"]])
            for pair in r["pairs"]:
                if len(pair) >= 5:
                    a["equal"][pair[0], pair[1]] += w * pair[4]
                    a["equal"][pair[1], pair[0]] += w * pair[4]
    return acc


def matrices(a, n):
    """(relative distance, stored scalars that differ), both in percent, over [parent, n children]."""
    dist = np.zeros((n + 1, n + 1))
    diff = np.zeros((n + 1, n + 1))
    for i in range(n):
        dist[0, i + 1] = dist[i + 1, 0] = 100 * np.sqrt(a["gram"][i, i] / a["parent"])
        diff[0, i + 1] = diff[i + 1, 0] = 100 * a["changed"][i] / a["size"]
        for j in range(n):
            if i != j:
                d2 = a["gram"][i, i] + a["gram"][j, j] - 2 * a["gram"][i, j]
                dist[i + 1, j + 1] = 100 * np.sqrt(max(d2, 0.0) / a["parent"])
                diff[i + 1, j + 1] = 100 * (1 - a["equal"][i, j] / a["size"])
    return dist, diff


def show(title, m, fmt, names):
    print(f"  {title}")
    print("  " + " " * 18 + "".join(f"{n[:11]:>12}" for n in names))
    for i, name in enumerate(names):
        print("  " + f"{name[:17]:<18}" + "".join(f"{fmt.format(m[i, j]):>12}" for j in range(len(names))))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    meta, rows = st_index.load_atlas(sys.argv[1])
    expert_weight = float(sys.argv[2]) if len(sys.argv) > 2 else 32.0
    names = [short(meta["parent"]) + "*", *(short(c) for c in meta["children"])]
    n = len(meta["children"])
    acc = accumulate(rows, n, expert_weight)
    layers = sorted({r["layer"] for r in rows if r["layer"] is not None})
    print(f"units {len(rows)} | layers {layers} | parent (marked *) {meta['parent']}")
    for g in [*GROUPS, "all"]:
        a = acc[g]
        if a["size"] == 0:
            continue
        dist, diff = matrices(a, n)
        print(f"\n== {g} ({a['size'] / 1e6:,.1f} M scalars)")
        show("relative distance, percent of the parent's norm", dist, "{:.4f}", names)
        show("stored scalars that differ, percent", diff, "{:.2f}", names)


if __name__ == "__main__":
    main()
