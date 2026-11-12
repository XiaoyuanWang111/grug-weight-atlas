"""Per-layer displacement of the attention GatedNorm and attention gate for the experts and merges of marin#9984.

Rows: layers. Columns, for each of four families (attention GatedNorm up and down projection, attention gate, and
the MLP GatedNorm up projection for comparison): E, E2, TA0.5 (measured), TIES (measured), TA1.0 (computed from the
two expert deltas) and the cosine between the two expert deltas. Values are |x - B| / |B| in percent.

Usage: python atlas/gate_layers.py <atlas dir>   (the atlas run of ta_full_strength.py)

Network: none; it reads <atlas dir>/meta.json and units.jsonl.
"""

import re
import sys

import numpy as np

import st_index

FAMILIES = [
    "model.attn_gated_norm.up_proj.weight",
    "model.attn_gated_norm.down_proj.weight",
    "model.self_attn.attn_gate.weight",
    "model.mlp_gated_norm.up_proj.weight",
]


def rel(value, parent_energy):
    return 100 * np.sqrt(value / parent_energy)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    meta, rows = st_index.load_atlas(sys.argv[1])
    names = [re.sub(r"^.*?67B-A2B-", "", c) for c in meta["children"]]
    e, e2, ties, ta = (
        next(i for i, n in enumerate(names) if s in n)
        for s in ("CruxEval-Expert-E", "MMLUPro-Expert-E2", "Merge-TIES", "Merge-TA0.5")
    )
    table = {}
    for r in rows:
        if r["component"] in FAMILIES:
            g, p = np.array(r["gram"]), r["parent_norm"] ** 2
            table[(r["component"], r["layer"])] = (
                rel(g[e, e], p),
                rel(g[e2, e2], p),
                rel(g[ta, ta], p),
                rel(g[ties, ties], p),
                rel(g[e, e] + g[e2, e2] + 2 * g[e, e2], p),
                g[e, e2] / np.sqrt(g[e, e] * g[e2, e2]) if g[e, e] and g[e2, e2] else float("nan"),
            )
    layers = sorted({layer for _, layer in table})
    for family in FAMILIES:
        print(f"\n{family.replace('model.', '')}")
        print("layer       E      E2   TA0.5    TIES  TA1.0*  cos(E,E2)")
        for layer in layers:
            v = table[(family, layer)]
            print(f"{layer:5d} " + " ".join(f"{x:7.3f}" for x in v[:5]) + f" {v[5]:9.3f}")
    print("\n* computed: |dE + dE2| / |B|")


if __name__ == "__main__":
    main()
