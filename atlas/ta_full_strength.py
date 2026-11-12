"""Where would the unpublished full-strength merge of marin#9984 sit, tensor family by tensor family?

#9984 built B + a (E - B) + b (E2 - B) from the student B = Step92 and the experts E = CruxEval-Expert-E-Step272
and E2 = MMLUPro-Expert-E2-Step254. a = b = 0.5 (Merge-TA0.5) and TIES (density 0.5, a = b = 1; Merge-TIES-d0.5)
were published; a = b = 1 was withheld. Its displacement from B before rounding follows from the two expert deltas,
|dE + dE2|^2 = |dE|^2 + |dE2|^2 + 2 <dE, dE2>, and an atlas run with B as parent holds those inner products for
every unit it read.

For every tensor family it prints the relative displacement |x - B| / |B| in percent for the two experts, the two
published merges (measured), the full-strength merge (computed from the experts, before rounding) and two trained
arms (MOPD-Step100, MixedRS-SFT-Step727); then the cosine between the expert deltas, the ratio of the full-strength
displacement to the larger expert displacement, and the layer where the full-strength displacement is largest.

Usage: python atlas/ta_full_strength.py <atlas dir>   (an atlas run with parent Step92 and all six children above)

Network: none; it reads <atlas dir>/meta.json and units.jsonl.
"""

import collections
import re
import sys

import numpy as np

import st_index


def rel(gram, parent, i):
    return 100 * np.sqrt(gram[i, i] / parent)


def full(gram, parent, e, e2):
    return 100 * np.sqrt((gram[e, e] + gram[e2, e2] + 2 * gram[e, e2]) / parent)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    meta, rows = st_index.load_atlas(sys.argv[1])
    names = [re.sub(r"^.*?67B-A2B-", "", c) for c in meta["children"]]

    def child(sub):
        return next(i for i, n in enumerate(names) if sub in n)

    e, e2, ties, ta, mopd, rs = (
        child("CruxEval-Expert-E"),
        child("MMLUPro-Expert-E2"),
        child("Merge-TIES"),
        child("Merge-TA0.5"),
        child("MOPD"),
        child("MixedRS-SFT"),
    )
    acc = collections.defaultdict(lambda: {"size": 0, "parent": 0.0, "gram": 0.0, "layers": {}})
    for r in rows:
        key = r["component"].replace("model.", "")
        a = acc[key]
        gram = np.array(r["gram"])
        a["size"] += r["size"]
        a["parent"] += r["parent_norm"] ** 2
        a["gram"] = a["gram"] + gram
        if r["layer"] is not None:
            cell = a["layers"].setdefault(r["layer"], {"parent": 0.0, "gram": 0.0})
            cell["parent"] += r["parent_norm"] ** 2
            cell["gram"] = cell["gram"] + gram

    print(
        f"{'tensor family':<38}{'scalars':>10}{'B RMS':>9} |{'E':>8}{'E2':>8}{'TA0.5':>8}{'TIES':>8}{'TA1.0*':>8} |"
        f"{'MOPD':>8}{'RS-SFT':>8} |{'cos E,E2':>9}{'TA1.0/max':>10}{'worst layer':>12}"
    )
    order = sorted(acc.items(), key=lambda kv: -full(kv[1]["gram"], kv[1]["parent"], e, e2))
    for key, a in order:
        g, p = a["gram"], a["parent"]
        if g[e, e] == 0 and g[e2, e2] == 0:
            print(f"{key:<38}{a['size']:>10,}{np.sqrt(p / a['size']):9.4f} | unchanged in both experts")
            continue
        cosine = g[e, e2] / np.sqrt(g[e, e] * g[e2, e2]) if g[e, e] and g[e2, e2] else float("nan")
        worst = (
            max((full(c["gram"], c["parent"], e, e2), layer) for layer, c in a["layers"].items())
            if a["layers"]
            else (float("nan"), None)
        )
        print(
            f"{key:<38}{a['size']:>10,}{np.sqrt(p / a['size']):9.4f} |{rel(g, p, e):8.3f}{rel(g, p, e2):8.3f}"
            f"{rel(g, p, ta):8.3f}{rel(g, p, ties):8.3f}{full(g, p, e, e2):8.3f} |{rel(g, p, mopd):8.3f}"
            f"{rel(g, p, rs):8.3f} |{cosine:9.3f}{full(g, p, e, e2) / max(rel(g, p, e), rel(g, p, e2)):10.2f}"
            f"{worst[0]:8.3f} L{worst[1]}"
        )
    print("* computed from the two expert deltas; relative displacement in percent of the student's norm in that family")
    print("routed experts: 8 of 256 experts per layer")


if __name__ == "__main__":
    main()
