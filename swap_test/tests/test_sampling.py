"""Shared-noise Gumbel-max sampler: common random numbers across arms, seeds, greedy mode, distribution."""

import math

import torch

from swaptest.sampling import Sampler, gumbel_noise


def test_same_logits_give_the_same_token_in_every_row():
    logits = torch.randn(1, 50).repeat(4, 1)
    sampler = Sampler(seed=9984, temperature=1.0)
    for position in range(20):
        tokens = sampler(logits, [3, 3, 3, 3], [position] * 4)
        assert len(set(tokens.tolist())) == 1


def test_noise_ignores_row_order_and_batch_composition():
    logits = torch.randn(3, 50)
    sampler = Sampler(seed=1, temperature=1.0)
    together = sampler(logits, [10, 11, 12], [5, 6, 7]).tolist()
    alone = [int(sampler(logits[i : i + 1], [10 + i], [5 + i])) for i in range(3)]
    reversed_rows = sampler(logits.flip(0), [12, 11, 10], [7, 6, 5]).tolist()[::-1]
    assert together == alone == reversed_rows


def test_different_seeds_prompts_and_positions_give_different_noise():
    a = gumbel_noise(1, 0, 0, 1000)
    assert not torch.equal(a, gumbel_noise(2, 0, 0, 1000))
    assert not torch.equal(a, gumbel_noise(1, 1, 0, 1000))
    assert not torch.equal(a, gumbel_noise(1, 0, 1, 1000))
    assert torch.equal(a, gumbel_noise(1, 0, 0, 1000))
    flat = torch.zeros(1, 64)
    s1 = [int(Sampler(1, 1.0)(flat, [0], [t])) for t in range(100)]
    s2 = [int(Sampler(2, 1.0)(flat, [0], [t])) for t in range(100)]
    assert sum(x != y for x, y in zip(s1, s2, strict=True)) > 80


def test_greedy_is_argmax():
    logits = torch.randn(5, 30)
    assert torch.equal(Sampler(0, 0.0)(logits, [0] * 5, [0] * 5), logits.argmax(-1))


def test_samples_follow_the_softmax():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    sampler = Sampler(seed=5, temperature=1.0)
    n = 6000
    counts = torch.bincount(torch.cat([sampler(logits, [0], [t]) for t in range(n)]), minlength=4).double()
    expected = torch.softmax(logits[0].double(), -1) * n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 16.3  # 99.9% quantile of chi-square with 3 degrees of freedom


def test_temperature_scales_the_logits():
    logits = torch.tensor([[1.0, 0.0]])
    hot = sum(int(Sampler(3, 4.0)(logits, [0], [t])) for t in range(3000)) / 3000
    cold = sum(int(Sampler(3, 0.25)(logits, [0], [t])) for t in range(3000)) / 3000
    assert abs(hot - 1 / (1 + math.exp(0.25))) < 0.04
    assert cold < 0.05
