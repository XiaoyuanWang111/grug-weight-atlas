"""Weight-delta atlas: children of one parent checkpoint, tensor by tensor, read by HTTP range (nothing stored).

A unit is a whole tensor outside the routed experts, a block of 4096 embedding or lm_head rows, or one routed
expert's slice of an expert tensor, for --experts-per-layer evenly spaced experts (0, 32, ..., 224 for 8 of 256).
Units are ordered layer by layer in the order given by --layers, so a partial run already covers whole layers. Per
unit one JSON line goes to <out>/units.jsonl:
  unit, tensor, component, layer, size, dtypes, parent_norm,
  children: [{delta_norm, changed, max_abs, sigma1 (top singular value of the delta, 2-D units),
              steps: [n1, n2, n3_4, n5_8, n_more] (BF16 only: how many stored values moved by that many steps of
              the BF16 grid; a sign change counts as "more")}],
  gram: inner products between the children's deltas (float64 accumulation),
  pairs: [i, j, both changed, opposite signs, equal stored values].
<out>/meta.json records the parent, the children, the sampling and the commit of every repo; <out>/log.txt the
progress. Units already in units.jsonl are skipped, so the script resumes; a unit that fails to download is retried
after a pause.

Usage:
  python atlas/atlas2.py --out DIR --parent REPO --children REPO [REPO ...]
                         [--experts-per-layer 8] [--layers all | 0,12,25] [--embed-blocks all | last | none]
  --embed-blocks reads every 4096-row block of embed_tokens and lm_head, only the last block, or none of them.

Network: the tensor index of every repo (st_index.index, cached), then for each unit one range request per
checkpoint for exactly the unit's bytes. The log prints the volume to read before it starts.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import st_index

EMBED_ROWS = 4096
ORDER = {"router": 0, "self_attn": 1, "shared_expert": 2}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--children", nargs="+", required=True)
    parser.add_argument("--experts-per-layer", type=int, default=8)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--embed-blocks", default="all", choices=["all", "last", "none"])
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def component(name):
    layer = re.search(r"layers\.(\d+)\.", name)
    return re.sub(r"layers\.\d+\.", "", name), (int(layer.group(1)) if layer else None)


def plan_units(shapes, names, layers, experts_per_layer, embed_blocks):
    """unit = (unit id, tensor name, element offset within the tensor, element count, 2-D shape or None)"""
    units = []
    for layer in layers:
        here = [k for k in names if component(k)[1] == layer]
        plain = [k for k in here if "mlp.experts." not in k]
        plain.sort(key=lambda k: (next((v for key, v in ORDER.items() if key in k), 3), k))
        for name in plain:
            shape = shapes[name]
            units.append((name, name, 0, int(np.prod(shape)), shape if len(shape) == 2 else None))
        for name in [k for k in here if "mlp.experts." in k]:
            n_exp, a, b = shapes[name]
            for e_id in range(0, n_exp, n_exp // experts_per_layer):
                units.append((f"{name}[{e_id}]", name, e_id * a * b, a * b, [a, b]))
    for name in [k for k in names if component(k)[1] is None]:
        shape = shapes[name]
        if component(name)[0] in ("model.embed_tokens.weight", "lm_head.weight"):
            if embed_blocks == "none":
                continue
            rows, cols = shape
            starts = list(range(0, rows, EMBED_ROWS))
            for r0 in starts if embed_blocks == "all" else starts[-1:]:
                r1 = min(rows, r0 + EMBED_ROWS)
                units.append((f"{name}[{r0}:{r1}]", name, r0 * cols, (r1 - r0) * cols, None))
        else:
            units.append((name, name, 0, int(np.prod(shape)), shape if len(shape) == 2 else None))
    return units


def fetch(idx, repo, name, offset, count):
    e = idx[repo][name]
    raw = st_index.read_elements(repo, e, offset, count)
    if len(raw) != count * st_index.ITEM[e["dtype"]]:
        raise RuntimeError(f"short read {repo} {name}: {len(raw)} of {count * st_index.ITEM[e['dtype']]}")
    return st_index.to_float32(raw, e["dtype"])


def dot64(a, b):
    return float(np.einsum("i,i->", a, b, dtype=np.float64))


def top_singular(mat, iters=12):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(mat.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    sigma = 0.0
    for _ in range(iters):
        u = mat @ v
        sigma = float(np.linalg.norm(u))
        if sigma == 0.0:
            return 0.0
        v = mat.T @ (u / sigma)
        nv = float(np.linalg.norm(v))
        if nv == 0.0:
            return 0.0
        v /= nv
    return sigma


def grid_steps(parent, child):
    """Histogram of how many BF16 grid steps each changed value moved: [1, 2, 3-4, 5-8, more or sign change]."""
    p = (parent.view(np.uint32) >> 16).astype(np.int32)
    c = (child.view(np.uint32) >> 16).astype(np.int32)
    same_sign = (p >> 15) == (c >> 15)
    steps = np.abs((p & 0x7FFF) - (c & 0x7FFF))
    steps[~same_sign] = 1 << 20
    steps = steps[steps > 0]
    edges = [1, 2, 3, 5, 9]
    return [int(((steps >= lo) & (steps < hi)).sum()) for lo, hi in zip(edges, [*edges[1:], 1 << 30], strict=True)]


def measure(idx, repos, workers, unit, name, offset, length, shape2d):
    def get(repo):
        return fetch(idx, repo, name, offset, length)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = list(pool.map(get, repos))
    parent = arrays[0]
    comp, layer = component(name)
    dtypes = [idx[r][name]["dtype"] for r in repos]
    row = {
        "unit": unit,
        "tensor": name,
        "component": comp,
        "layer": layer,
        "size": int(parent.size),
        "dtypes": dtypes,
        "parent_norm": float(np.sqrt(dot64(parent, parent))),
        "children": [],
        "gram": [],
        "pairs": [],
    }
    steps = [
        grid_steps(parent, child) if dtypes[0] == "BF16" and dtypes[i + 1] == "BF16" else None
        for i, child in enumerate(arrays[1:])
    ]
    deltas = arrays[1:]
    for d in deltas:
        d -= parent  # float32; inner products below accumulate in float64
    del arrays
    n = len(deltas)
    gram = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            gram[i][j] = gram[j][i] = dot64(deltas[i], deltas[j])
    nonzero = [d != 0 for d in deltas]
    for i, d in enumerate(deltas):
        entry = {
            "delta_norm": float(np.sqrt(gram[i][i])),
            "changed": float(nonzero[i].mean()),
            "max_abs": float(np.abs(d).max()),
        }
        if shape2d is not None and entry["delta_norm"] > 0:
            entry["sigma1"] = top_singular(d.reshape(shape2d))
        if steps[i] is not None:
            entry["steps"] = steps[i]
        row["children"].append(entry)
    row["gram"] = gram
    negative = [d < 0 for d in deltas]
    for i in range(n):
        for j in range(i + 1, n):
            both = nonzero[i] & nonzero[j]
            row["pairs"].append(
                [
                    i,
                    j,
                    int(both.sum()),
                    int((both & (negative[i] != negative[j])).sum()),
                    int((deltas[i] == deltas[j]).sum()),
                ]
            )
    return row


def main():
    args = parse_args()
    out_dir = Path(args.out)
    parent_repo, children = args.parent, args.children
    repos = [parent_repo, *children]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "log.txt").open("a", encoding="utf-8") as log:

        def say(*parts):
            line = time.strftime("%H:%M:%S ") + " ".join(str(p) for p in parts)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        idx = {}
        for repo in repos:
            idx[repo] = st_index.index(repo)
            say("indexed", repo, len(idx[repo]) - 1, "tensors, sha", idx[repo]["__repo__"]["sha"])
        names = sorted(k for k in idx[parent_repo] if k != "__repo__")
        for repo in children:
            other = {k for k in idx[repo] if k != "__repo__"}
            if other != set(names):
                say(
                    "TENSOR NAMES DIFFER in",
                    repo,
                    "only in parent:",
                    sorted(set(names) - other)[:6],
                    "only in child:",
                    sorted(other - set(names))[:6],
                )
                sys.exit(2)
            bad = [k for k in names if idx[repo][k]["shape"] != idx[parent_repo][k]["shape"]]
            if bad:
                say(
                    "SHAPES DIFFER in", repo, [(k, idx[parent_repo][k]["shape"], idx[repo][k]["shape"]) for k in bad[:6]]
                )
                sys.exit(2)
            recast = sorted(
                {
                    (component(k)[0], idx[parent_repo][k]["dtype"], idx[repo][k]["dtype"])
                    for k in names
                    if idx[repo][k]["dtype"] != idx[parent_repo][k]["dtype"]
                }
            )
            if recast:
                say("stored dtype differs from the parent's in", repo, recast)

        all_layers = sorted({component(k)[1] for k in names if component(k)[1] is not None})
        layers = all_layers if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        meta = {
            "parent": parent_repo,
            "children": children,
            "experts_per_layer": args.experts_per_layer,
            "layers": layers,
            "embed_blocks": args.embed_blocks,
            "sha": {r: idx[r]["__repo__"]["sha"] for r in repos},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

        shapes = {k: idx[parent_repo][k]["shape"] for k in names}
        units = plan_units(shapes, names, layers, args.experts_per_layer, args.embed_blocks)
        path = out_dir / "units.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    done.add(json.loads(line)["unit"])
        todo = [u for u in units if u[0] not in done]

        def unit_bytes(name, length):
            return length * sum(st_index.ITEM[idx[r][name]["dtype"]] for r in repos)

        total_bytes = sum(unit_bytes(u[1], u[3]) for u in todo)
        say(
            f"units {len(units)}, done {len(done)}, to do {len(todo)}, {total_bytes / 1e9:.1f} GB to read over "
            f"{len(repos)} checkpoints"
        )

        with path.open("a", encoding="utf-8") as out:
            t0, read_bytes = time.time(), 0
            for n_unit, (unit, name, offset, length, shape2d) in enumerate(todo):
                for attempt in range(8):
                    try:
                        row = measure(idx, repos, args.workers, unit, name, offset, length, shape2d)
                        break
                    except Exception as error:
                        say(f"unit {unit} failed (attempt {attempt + 1}): {str(error)[:200]}")
                        time.sleep(30 + 30 * attempt)
                else:
                    say("giving up on", unit)
                    sys.exit(3)
                out.write(json.dumps(row) + "\n")
                out.flush()
                read_bytes += unit_bytes(name, length)
                if n_unit % 20 == 0 or n_unit == len(todo) - 1:
                    dt = time.time() - t0
                    say(
                        f"{n_unit + 1}/{len(todo)} {unit} | {read_bytes / 1e9:.2f} GB in {dt / 60:.1f} min "
                        f"({read_bytes / 2**20 / max(dt, 1):.1f} MiB/s), "
                        f"eta {(total_bytes - read_bytes) / max(read_bytes, 1) * dt / 60:.0f} min"
                    )
        say("finished")


if __name__ == "__main__":
    main()
