"""Statistics of the assistant-prefix probe (`probes/assistant_prefix/regrade.py`) on made-up rewards."""

import numpy as np
import pytest

import regrade

LABELS = regrade.RENDERER_LABELS


def bootstrap_draws(count: int) -> np.ndarray:
    return np.random.default_rng(7).integers(0, count, size=(4000, count))


def code_fence_like_rewards() -> tuple[np.ndarray, np.ndarray]:
    """200 rows with full credit in 13 rows of `old` and 33 of `new`, 3 of them shared: 30 against 10 discordant."""
    old, new = np.zeros(200), np.zeros(200)
    old[:13] = 1.0
    new[10:43] = 1.0
    return new, old


def test_mcnemar_counts_discordant_rows_and_gives_the_exact_p():
    new = np.array([1.0] * 8 + [0.0] * 2)
    old = np.array([0.0] * 6 + [1.0] * 4)

    b, c, p = regrade.mcnemar(new, old)

    assert (b, c) == (6, 2)
    assert p == pytest.approx(2 * (1 + 8 + 28) / 2**8)


def test_mcnemar_without_discordant_rows_gives_p_one():
    same = np.array([1.0, 0.0, 1.0])

    assert regrade.mcnemar(same, same) == (0, 0, 1.0)


@pytest.mark.parametrize(
    "estimate,low,high,p,expected",
    [
        (0.75, 0.66, 0.83, 1e-9, LABELS[0]),
        (-0.18, -0.31, -0.06, 0.0096, LABELS[1]),
        (0.02, -0.03, 0.07, 0.5, "no effect"),
        (0.095, 0.035, 0.155, 0.004, "inconclusive"),
        (0.12, 0.04, 0.20, 0.2, "inconclusive"),
        (0.12, -0.01, 0.25, 0.01, "inconclusive"),
    ],
    ids=["gain", "loss", "inside-the-margin", "below-ten-points", "p-too-large", "interval-spans-zero"],
)
def test_decide_applies_the_pre_registered_rule(estimate, low, high, p, expected):
    assert regrade.decide(estimate, low, high, p, LABELS) == expected


def test_contrast_reports_the_paired_difference_and_its_interval():
    new, old = code_fence_like_rewards()

    result = regrade.contrast(new, old, bootstrap_draws(200), LABELS)

    assert result["estimate"] == pytest.approx(0.10)
    assert (result["b"], result["c"]) == (30, 10)
    assert 0 < result["interval"][0] < 0.10 < result["interval"][1]
    assert result["label"] == LABELS[0]


def test_robust_contrast_keeps_the_label_without_unresolved_replies():
    new, old = code_fence_like_rewards()
    resolved = np.zeros(200, dtype=bool)

    result = regrade.robust_contrast(new, old, resolved, resolved, bootstrap_draws(200), LABELS)

    assert result["decision"] == LABELS[0]


def test_robust_contrast_is_inconclusive_when_an_unresolved_reply_can_change_the_label():
    new, old = code_fence_like_rewards()
    resolved = np.zeros(200, dtype=bool)
    old_unresolved = resolved.copy()
    old_unresolved[199] = True  # counted as full credit for `old` in the low extreme: 19 of 200 = 9.5 points

    result = regrade.robust_contrast(new, old, resolved, old_unresolved, bootstrap_draws(200), LABELS)

    assert result["label"] == LABELS[0]
    assert result["decision"] == "inconclusive (capped replies)"
