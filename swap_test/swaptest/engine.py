"""Batched generation: chunked prefill of left-padded prefixes, then lockstep decoding that drops finished rows.

A wave is a set of rows decoded together. Each row's prefix (prompt plus any tokens generated before a restart)
is right-aligned in a shared slot range; rows may belong to different arms. After every step, rows that sampled a
stop token or reached the cap leave the batch and the cache.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

import torch

from swaptest.kvcache import KVCache
from swaptest.model import GrugModel
from swaptest.patch import arm_index
from swaptest.sampling import Sampler


@dataclass
class Row:
    """One reply in progress: an (arm, prompt) pair and the tokens generated so far."""

    arm: str
    prompt_id: int
    prompt: list[int]
    generated: list[int] = field(default_factory=list)
    elapsed: float = 0.0  # seconds spent on this row by earlier processes

    @property
    def key(self) -> tuple[str, int]:
        return (self.arm, self.prompt_id)


@dataclass(frozen=True)
class Finished:
    row: Row
    stop_reason: int | str  # the stop token id, or "cap"
    wall_s: float


def left_pad(seqs: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align token lists in one [B, L] tensor (padding id 0); return it and each row's padding."""
    length = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), length, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, length - len(s) :] = torch.tensor(s, dtype=torch.long)
    return ids, torch.tensor([length - len(s) for s in seqs], dtype=torch.long)


def prefill(
    model: GrugModel,
    cache: KVCache,
    ids: torch.Tensor,
    pad: torch.Tensor,
    arm: torch.Tensor,
    chunk: int,
    row_block: int,
    on_chunk: Callable[[slice, int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Run `ids` [B, L] into the cache in blocks of rows and chunks of slots; return the last hidden state [B, D].

    `on_chunk(rows, start, hidden)` receives every chunk's hidden states, for callers that need all positions.
    """
    last = []
    for r0 in range(0, ids.shape[0], row_block):
        rows = slice(r0, min(ids.shape[0], r0 + row_block))
        for c0 in range(0, ids.shape[1], chunk):
            hidden = model.forward(ids[rows, c0 : c0 + chunk], cache, c0, pad[rows], arm[rows], rows=rows)
            if on_chunk is not None:
                on_chunk(rows, c0, hidden)
        last.append(hidden[:, -1])
    return torch.cat(last)


def generate(
    model: GrugModel,
    rows: list[Row],
    sampler: Sampler,
    cap: int,
    stop_ids: Collection[int],
    on_finish: Callable[[Finished], None],
    on_step: Callable[[int, list[Row], float], None] | None = None,
    chunk: int = 256,
    row_block: int = 16,
) -> None:
    """Decode `rows` together until each samples a stop token or holds `cap` generated tokens.

    `on_finish` is called once per row as it finishes; `on_step(step, active_rows, seconds_since_start)` after
    every decode step, once finished rows have been removed.
    """
    for row in rows:
        if len(row.generated) >= cap or (row.generated and row.generated[-1] in stop_ids):
            raise ValueError(f"row {row.key} is already finished")
    start_time = time.monotonic()
    ids, pad = left_pad([row.prompt + row.generated for row in rows])
    arm = torch.tensor([arm_index(row.arm) for row in rows], dtype=torch.long)
    remaining = max(cap - len(row.generated) for row in rows)
    cache = model.new_cache(len(rows), ids.shape[1] + remaining - 1)
    logits = model.logits(prefill(model, cache, ids, pad, arm, chunk, row_block))
    slot, step, active = ids.shape[1], 0, list(rows)
    while True:
        tokens = sampler(logits, [r.prompt_id for r in active], [len(r.generated) for r in active])
        finished = []
        for i, (row, token) in enumerate(zip(active, tokens.tolist(), strict=True)):
            row.generated.append(token)
            if token in stop_ids or len(row.generated) >= cap:
                reason = token if token in stop_ids else "cap"
                on_finish(Finished(row, reason, row.elapsed + time.monotonic() - start_time))
                finished.append(i)
        step += 1
        if finished:
            order = cache.drop(finished)
            active = [active[j] for j in order]
            tokens, pad, arm = tokens[order], pad[order], arm[order]
        if on_step is not None:
            on_step(step, active, time.monotonic() - start_time)
        if not active:
            return
        logits = model.logits(model.forward(tokens[:, None], cache, slot, pad, arm)[:, -1])
        slot += 1
