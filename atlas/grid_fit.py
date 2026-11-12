"""Test one model of how a stored BF16 child differs from its stored BF16 parent, on one tensor.

Model: child = round_to_BF16(parent + update), with the update of every scalar drawn from N(0, sigma^2), the same
sigma for small and large weights. A stored value then changes when |update| exceeds half a grid step, so in the
binade [2^e, 2^(e+1)), where the step is s = 2^(e-7), the share of changed values is 2 * (1 - Phi(s / (2 sigma))).

For each child the script fits sigma by maximum likelihood over binades, prints observed and predicted change
rates per binade, and then, under the fitted model, how much of the update the stored difference keeps:
  erased    share of scalars whose update is rounded away (stored value unchanged),
  kept      regression coefficient of the stored difference on the update, E[u q] / E[u^2],
  cosine    E[u q] / sqrt(E[u^2] E[q^2]),
where u is the update and q the stored difference. It also prints what the alternative "both checkpoints are
rounded copies of float32 masters" would predict for the change rate, E[min(1, |u| / s)].

Usage: python atlas/grid_fit.py [--slice START:COUNT] <tensor name> <parent repo> <child repo> [<child repo> ...]

Network: as grid_probe.py: one range request per repo for the tensor (or the slice); 13.1 MB per repo for
model.layers.12.self_attn.q_proj.weight.
"""

import math
import re
import sys

import numpy as np

import st_index


def phi_tail(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def nll(log_sigma, k, n_in, steps):
    """Negative binomial log-likelihood of k changed values out of n_in per binade at sigma = exp(log_sigma)."""
    sigma = math.exp(log_sigma)
    prob = np.clip(np.array([2 * phi_tail(s / (2 * sigma)) for s in steps]), 1e-12, 1 - 1e-12)
    return -float((k * np.log(prob) + (n_in - k) * np.log(1 - prob)).sum())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    start, count = 0, None
    if argv[0] == "--slice":
        start, count = (int(x) for x in argv[1].split(":"))
        argv = argv[2:]
    name, parent, children = argv[0], argv[1], argv[2:]

    p = st_index.read_bf16_bits(parent, name, start, count)
    exponent = ((p & 0x7FFF) >> 7) - 127
    binades = [e for e in range(-20, 3) if (exponent == e).sum() >= 2000]
    n_in = np.array([(exponent == e).sum() for e in binades], dtype=np.float64)
    steps = np.array([2.0 ** (e - 7) for e in binades])
    rng = np.random.default_rng(0)
    z = rng.standard_normal(400_000)
    print(f"tensor {name} | {p.size:,} values | binades used {binades[0]}..{binades[-1]}")
    for repo in children:
        c = st_index.read_bf16_bits(repo, name, start, count)
        changed = c != p
        k = np.array([(changed & (exponent == e)).sum() for e in binades], dtype=np.float64)
        grid = np.linspace(math.log(1e-7), math.log(1e-3), 2000)
        best = grid[int(np.argmin([nll(g, k, n_in, steps) for g in grid]))]
        fine = np.linspace(best - 0.01, best + 0.01, 400)
        sigma = math.exp(fine[int(np.argmin([nll(g, k, n_in, steps) for g in fine]))])
        pred = np.array([2 * phi_tail(s / (2 * sigma)) for s in steps])
        master = np.array([np.minimum(1.0, np.abs(z) * sigma / s).mean() for s in steps])
        obs = k / n_in
        u = z * sigma
        e_uq = e_qq = 0.0
        for s, n in zip(steps, n_in, strict=True):
            q = s * np.round(u / s)
            e_uq += n * float((u * q).mean())
            e_qq += n * float((q * q).mean())
        e_uu = n_in.sum() * sigma**2
        print(
            f"\n{re.sub(r'^.*?67B-A2B-', '', repo)}: fitted sigma {sigma:.3e} | changed {100 * changed.mean():.2f}% "
            f"observed, {100 * (pred * n_in).sum() / n_in.sum():.2f}% predicted | largest gap "
            f"{100 * np.abs(obs - pred).max():.2f} points"
        )
        print("  binade    step      values %   observed %  predicted %   float32-master model %")
        for e, s, n, o, m, a in zip(binades, steps, n_in, obs, pred, master, strict=True):
            print(f"  2^{e:<4d} {s:9.2e} {100 * n / p.size:9.2f} {100 * o:11.3f} {100 * m:11.3f} {100 * a:18.3f}")
        print(
            f"  under the model: erased {100 * (1 - (pred * n_in).sum() / n_in.sum()):.1f}% of scalars | "
            f"kept {e_uq / e_uu:.3f} | cosine {e_uq / math.sqrt(e_uu * e_qq):.3f} | "
            f"stored difference energy / update energy {e_qq / e_uu:.2f}"
        )


if __name__ == "__main__":
    main()
