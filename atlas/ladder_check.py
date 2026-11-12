"""Does the one-parameter rounding model reproduce the BF16 erasure numbers of marin#10700?

#10700 merged Snowball v0.1 (BF16) toward v0.1.1 (float32) as v0.1 + alpha * delta and stored each point in BF16.
Reported there: 65,695,399,644 of 67,078,882,816 entries have a nonzero donor delta; at alpha = 1 BF16 erases
69.6% of them; at alpha = 0.5 it changes 16.15% of all entries, erases 83.51% of the real deltas, and the
relative L2 error against the ideal float32 update is 84.46%.

Model: every scalar's donor delta is N(0, sigma^2) regardless of the weight's size, and the stored value is
round_to_BF16(anchor + alpha * delta). The share of weights per binade is taken from one 2560 x 2560 matrix of a
Snowball checkpoint (all weight matrices of this model have the same RMS, 9.75e-3). sigma is fitted to the single
number "69.6% erased at alpha = 1"; everything at other alphas is then a prediction.

Usage: python atlas/ladder_check.py

Network: none.
"""

import math

import numpy as np

# share of values per binade 2^e .. 2^(e+1), e = -18 .. -5, from model.layers.12.self_attn.q_proj.weight of Step92
SHARE = np.array([0.04, 0.07, 0.14, 0.28, 0.56, 1.12, 2.25, 4.47, 8.76, 16.38, 26.65, 28.56, 10.30, 0.38])
SHARE = SHARE / SHARE.sum()
STEP = np.array([2.0 ** (e - 7) for e in range(-18, -4)])
NONZERO = 65_695_399_644 / 67_078_882_816


def stats(sigma, z):
    """changed share, kept coefficient, cosine, relative L2 error of the stored difference against the update."""
    u = z * sigma
    changed = uq = qq = err = 0.0
    for share, s in zip(SHARE, STEP, strict=True):
        q = s * np.round(u / s)
        changed += share * float((q != 0).mean())
        uq += share * float((u * q).mean())
        qq += share * float((q * q).mean())
        err += share * float(((q - u) ** 2).mean())
    uu = sigma**2
    return changed, uq / uu, uq / math.sqrt(uu * qq), math.sqrt(err / uu)


def main():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(2_000_000)
    target = 1 - 0.696  # share of real deltas that survive at alpha = 1
    lo, hi = 1e-7, 1e-3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if stats(mid, z)[0] < target:
            lo = mid
        else:
            hi = mid
    sigma1 = math.sqrt(lo * hi)
    print(f"sigma fitted to '69.6% erased at alpha = 1': {sigma1:.3e}")
    print(
        "alpha  changed % of real deltas  erased %  changed % of all entries  kept   effective alpha  cosine  "
        "relative L2 error %"
    )
    for alpha in (0.25, 0.5, 0.75, 0.875, 0.9375, 1.0):
        changed, kept, cosine, err = stats(alpha * sigma1, z)
        print(
            f"{alpha:<6} {100 * changed:21.2f} {100 * (1 - changed):9.2f} {100 * changed * NONZERO:25.2f} {kept:6.3f} "
            f"{alpha * kept:16.3f} {cosine:7.3f} {100 * err:20.2f}"
        )
    print("reported in #10700 at alpha = 0.5: changed 16.15% of all entries, erased 83.51%, relative L2 error 84.46%")


if __name__ == "__main__":
    main()
