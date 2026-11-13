"""Exact one-sided Fisher test, Wilson and Newcombe intervals, and the pre-registered decision rule.

Decision rule for an arm X against the reference arm A, on R = share of replies that reach the cap without a stop
token: "effect" if R_X - R_A >= 0.25 and the one-sided Fisher exact p (X has more capped replies) is below 0.01;
"no effect" if the upper end of the Newcombe 95% interval (hybrid score, method 10) for R_X - R_A is below 0.15;
otherwise "inconclusive".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import NormalDist

Z95 = NormalDist().inv_cdf(0.975)
EFFECT_MIN_DIFF = 0.25
EFFECT_MAX_P = 0.01
NO_EFFECT_MAX_UPPER = 0.15


def fisher_greater(x1: int, n1: int, x2: int, n2: int) -> float:
    """P(X >= x1) where X, the successes in group 1, is hypergeometric given both margins (exact)."""
    total = x1 + x2
    lo, hi = max(x1, total - n2), min(n1, total)
    tail = sum(math.comb(n1, k) * math.comb(n2, total - k) for k in range(lo, hi + 1))
    return float(Fraction(tail, math.comb(n1 + n2, total)))


def wilson(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for x successes of n, within [0, 1].

    The ends are exact at the boundaries: the lower end is 0 when x = 0 and the upper end is 1 when x = n (the
    closed form gives these values only up to rounding).
    """
    if not 0 <= x <= n or n <= 0:
        raise ValueError(f"need 0 <= x <= n and n > 0, got x={x}, n={n}")
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lower = 0.0 if x == 0 else max(0.0, centre - half)
    upper = 1.0 if x == n else min(1.0, centre + half)
    return lower, upper


def newcombe(x1: int, n1: int, x2: int, n2: int, z: float = Z95) -> tuple[float, float]:
    """Interval for p1 - p2 from the two Wilson intervals (Newcombe 1998, method 10)."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


@dataclass(frozen=True)
class Comparison:
    capped: int
    n: int
    ref_capped: int
    ref_n: int
    diff: float
    p_value: float
    lower: float
    upper: float
    outcome: str


def compare(capped: int, n: int, ref_capped: int, ref_n: int) -> Comparison:
    diff = capped / n - ref_capped / ref_n
    p = fisher_greater(capped, n, ref_capped, ref_n)
    lower, upper = newcombe(capped, n, ref_capped, ref_n)
    if diff >= EFFECT_MIN_DIFF and p < EFFECT_MAX_P:
        outcome = "effect"
    elif upper < NO_EFFECT_MAX_UPPER:
        outcome = "no effect"
    else:
        outcome = "inconclusive"
    return Comparison(capped, n, ref_capped, ref_n, diff, p, lower, upper, outcome)
