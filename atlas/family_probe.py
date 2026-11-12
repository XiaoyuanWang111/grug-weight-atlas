"""Relative displacement of small control tensors between a parent and child checkpoints, by tensor family.

Reads, for every layer, the GatedNorm projections (attention and MLP), the attention gate and the router weight,
plus the GatedNorms of the embedding and of the final norm, and for three layers (first, middle and last: 0, 13 and
25 in a 26-layer model) the reference matrices q_proj, o_proj, shared_expert.up_proj and routed expert 0 of
gate_proj. Everything is read by HTTP range; nothing is stored.

Per child and family it prints: scalars read, RMS of the parent, |child - parent| / |parent| in percent, the share
of stored values that differ, and the largest relative displacement of one layer.

Usage: python atlas/family_probe.py <parent repo> <child repo> [<child repo> ...]

Network: the tensor index of every repo (st_index.index, cached), then one range request per unit and repo; for a
Grug 67B checkpoint 122,572,800 bfloat16 values (about 245 MB) per repo.
"""

import collections
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import st_index

ALL_LAYERS = re.compile(r"(gated_norm\.(down|up)_proj\.weight|self_attn\.attn_gate\.weight|mlp\.router\.weight)$")
SOME_LAYERS = re.compile(
    r"(self_attn\.(q|o)_proj\.weight|shared_expert\.up_proj\.weight|mlp\.experts\.gate_proj\.weight)$"
)


def family(name):
    return re.sub(r"layers\.\d+\.", "", name).replace("model.", "")


def plan_units(idx, parent, names):
    layers = sorted({int(m.group(1)) for k in names if (m := re.search(r"layers\.(\d+)\.", k))})
    picked = {layers[0], layers[len(layers) // 2], layers[-1]}
    units = []
    for name in names:
        layer = re.search(r"layers\.(\d+)\.", name)
        layer = int(layer.group(1)) if layer else None
        shape = idx[parent][name]["shape"]
        if ALL_LAYERS.search(name):
            units.append((name, layer, 0, int(np.prod(shape))))
        elif SOME_LAYERS.search(name) and layer in picked:
            if "mlp.experts." in name:
                units.append((name, layer, 0, int(np.prod(shape[1:]))))  # routed expert 0
            else:
                units.append((name, layer, 0, int(np.prod(shape))))
    return layers, units


def fetch(idx, parent, repo, name, offset, count):
    e = idx[repo].get(name)
    if e is None or e["shape"] != idx[parent][name]["shape"]:
        return None
    raw = st_index.read_elements(repo, e, offset, count)
    return st_index.to_float32(raw, e["dtype"]).astype(np.float64)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parent, children = sys.argv[1], sys.argv[2:]
    repos = [parent, *children]
    idx = {repo: st_index.index(repo) for repo in repos}
    names = sorted(k for k in idx[parent] if k != "__repo__")
    layers, units = plan_units(idx, parent, names)

    acc = {
        c: collections.defaultdict(lambda: {"n": 0, "parent": 0.0, "delta": 0.0, "changed": 0, "worst": (0.0, None)})
        for c in children
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, layer, offset, count in units:
            arrays = list(pool.map(lambda repo, n=name, o=offset, c=count: fetch(idx, parent, repo, n, o, c), repos))
            parent_values = arrays[0]
            p2 = float(parent_values @ parent_values)
            for child, arr in zip(children, arrays[1:], strict=True):
                if arr is None:
                    continue
                d = arr - parent_values
                a = acc[child][family(name)]
                d2 = float(d @ d)
                a["n"] += parent_values.size
                a["parent"] += p2
                a["delta"] += d2
                a["changed"] += int((d != 0).sum())
                rel = 100 * np.sqrt(d2 / p2) if p2 else 0.0
                if rel > a["worst"][0]:
                    a["worst"] = (rel, layer)

    print(f"parent {parent} | layers {len(layers)} | dtypes {sorted({idx[parent][n]['dtype'] for n in names})}")
    for child in children:
        print(f"\nchild {child} | dtypes {sorted({idx[child][n]['dtype'] for n in names if n in idx[child]})}")
        print(
            f"  {'family':<36}{'scalars':>12}{'parent RMS':>12}{'displacement %':>16}{'changed %':>11}"
            f"{'worst layer %':>15}"
        )
        for fam, a in sorted(acc[child].items(), key=lambda kv: -kv[1]["delta"] / max(kv[1]["parent"], 1e-300)):
            worst = f"{a['worst'][0]:.3f} L{a['worst'][1]}" if a["worst"][1] is not None else "-"
            print(
                f"  {fam:<36}{a['n']:>12,}{np.sqrt(a['parent'] / a['n']):12.5f}"
                f"{100 * np.sqrt(a['delta'] / a['parent']):16.4f}{100 * a['changed'] / a['n']:11.2f}{worst:>15}"
            )


if __name__ == "__main__":
    main()
