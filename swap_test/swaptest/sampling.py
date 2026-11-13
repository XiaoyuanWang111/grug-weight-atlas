"""Token sampling with common random numbers across arms.

Temperature sampling uses the Gumbel-max trick: the token is argmax(logits / T + g), where g is standard Gumbel
noise over the vocabulary. The noise depends only on (seed, prompt id, position in the reply), never on the arm,
the batch or the row order, so two arms that produce the same logits at the same position pick the same token,
and a row regenerated after a restart sees the same noise. Temperature 0 means greedy decoding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch


def noise_seed(seed: int, prompt_id: int, position: int) -> int:
    """63-bit generator seed derived from (seed, prompt id, position) by BLAKE2b."""
    digest = hashlib.blake2b(f"{seed}:{prompt_id}:{position}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def gumbel_noise(seed: int, prompt_id: int, position: int, vocab: int) -> torch.Tensor:
    """Standard Gumbel noise of shape [vocab] in float64."""
    generator = torch.Generator().manual_seed(noise_seed(seed, prompt_id, position))
    uniform = torch.rand(vocab, generator=generator, dtype=torch.float64)
    return -torch.log(-torch.log(uniform))  # uniform == 0 gives -inf, which is never selected


class Sampler:
    """Picks one token per row from logits [B, V]."""

    def __init__(self, seed: int, temperature: float) -> None:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.seed = seed
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor, prompt_ids: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        if self.temperature == 0:
            return logits.float().argmax(-1)
        vocab = logits.shape[-1]
        noise = torch.stack([gumbel_noise(self.seed, p, t, vocab) for p, t in zip(prompt_ids, positions, strict=True)])
        return (logits.double() / self.temperature + noise).argmax(-1)
