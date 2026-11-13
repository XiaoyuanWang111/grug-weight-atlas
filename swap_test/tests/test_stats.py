"""Exact Fisher test against hand-computed tables, Newcombe's published example, and the decision rule."""

import math

import pytest

from swaptest.stats import compare, fisher_greater, newcombe, wilson


def test_fisher_against_hand_computed_tables():
    # Tea tasting: 3 of 4 in group 1 vs 1 of 4; P(X >= 3) = (C(4,3) C(4,1) + C(4,4) C(4,0)) / C(8,4) = 17/70.
    assert fisher_greater(3, 4, 1, 4) == pytest.approx(17 / 70, abs=1e-15)
    assert fisher_greater(1, 1, 0, 1) == pytest.approx(0.5, abs=1e-15)
    # 12 capped replies, all in group 1 of 48 vs 0 of 48: prod_{i<12} (48 - i) / (96 - i).
    assert fisher_greater(12, 48, 0, 48) == pytest.approx(math.prod((48 - i) / (96 - i) for i in range(12)), rel=1e-12)
    assert fisher_greater(0, 48, 0, 48) == 1.0
    # 2 vs 2 of 48: P(X >= 2) = 1 - P(X <= 1) with C(96,4) = 3,321,960 tables.
    p_le_1 = (math.comb(48, 0) * math.comb(48, 4) + math.comb(48, 1) * math.comb(48, 3)) / math.comb(96, 4)
    assert fisher_greater(2, 48, 2, 48) == pytest.approx(1 - p_le_1, abs=1e-15)


def test_fisher_agrees_with_scipy_when_installed():
    stats = pytest.importorskip("scipy.stats")
    for x1, x2 in ((30, 2), (8, 1), (5, 5), (0, 3), (48, 47)):
        expected = stats.fisher_exact([[x1, 48 - x1], [x2, 48 - x2]], alternative="greater").pvalue
        assert fisher_greater(x1, 48, x2, 48) == pytest.approx(expected, rel=1e-9)


def test_newcombe_matches_the_published_example():
    # Newcombe (1998), example 56/70 - 48/80: method 10 gives 0.0524 to 0.3339.
    lo, hi = newcombe(56, 70, 48, 80)
    assert lo == pytest.approx(0.0524, abs=5e-5) and hi == pytest.approx(0.3339, abs=5e-5)
    assert wilson(0, 48)[0] == 0.0 and wilson(48, 48)[1] == 1.0


def test_decision_rule_outcomes():
    effect = compare(30, 48, 1, 48)
    assert effect.outcome == "effect" and effect.diff >= 0.25 and effect.p_value < 0.01
    none = compare(1, 48, 1, 48)
    assert none.outcome == "no effect" and none.upper < 0.15
    unclear = compare(8, 48, 1, 48)
    assert unclear.outcome == "inconclusive"
    # a large difference that is not significant at 0.01 is inconclusive, not an effect
    small_n = compare(3, 6, 0, 6)
    assert small_n.diff >= 0.25 and small_n.p_value >= 0.01 and small_n.outcome == "inconclusive"
