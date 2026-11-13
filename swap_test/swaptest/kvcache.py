"""Preallocated key/value cache for a batch of rows that advance in lockstep, one slot per token.

All rows share slot numbers: row r holds its prompt right-aligned in slots `pad[r]` .. `L - 1` of a batch whose
longest prefix has L tokens, and every decode step writes one slot for every row. Long (full-attention) layers
keep every slot; short (sliding-window) layers keep a ring of the last `sliding_window` slots, so a slot's
entry in a ring of `size` entries is `slot % size`. Finished rows are removed by moving the last rows into their
places, which copies only the moved rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from swaptest.config import ModelConfig


def entry_slots(size: int, end: int) -> torch.Tensor:
    """Slot held by each used entry of a ring of `size` entries once slots 0 .. end - 1 have been written."""
    j = torch.arange(min(end, size))
    return (end - 1) - torch.remainder(end - 1 - j, size)


class KVCache:
    """Per-layer key and value buffers of shape [rows, kv_heads, entries, head_dim]."""

    def __init__(self, cfg: ModelConfig, rows: int, max_len: int, dtype: torch.dtype) -> None:
        self.max_len = max_len
        self.rows = rows
        self.sizes = [max_len if cfg.is_long(i) else min(cfg.sliding_window, max_len) for i in range(cfg.num_layers)]
        self.k = [torch.zeros(rows, cfg.num_kv_heads, s, cfg.head_dim, dtype=dtype) for s in self.sizes]
        self.v = [torch.zeros(rows, cfg.num_kv_heads, s, cfg.head_dim, dtype=dtype) for s in self.sizes]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def write(self, layer: int, rows: slice, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k, v of shape [rows, T, kv_heads, head_dim] for slots start .. start + T - 1."""
        size, count = self.sizes[layer], k.shape[1]
        if start + count > self.max_len:
            raise ValueError(f"slot {start + count - 1} is beyond the cache length {self.max_len}")
        kt, vt = k.transpose(1, 2), v.transpose(1, 2)
        if count > size:  # only the last `size` slots survive in a ring
            kt, vt = kt[:, :, -size:], vt[:, :, -size:]
            start, count = start + count - size, size
        first = start % size
        head = min(count, size - first)
        for buffer, new in ((self.k[layer], kt), (self.v[layer], vt)):
            buffer[rows, :, first : first + head] = new[:, :, :head]
            if head < count:
                buffer[rows, :, : count - head] = new[:, :, head:]

    def entries(self, layer: int, rows: slice, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Views of the used entries once slots 0 .. end - 1 have been written (see `entry_slots`)."""
        used = min(end, self.sizes[layer])
        return self.k[layer][rows, :, :used], self.v[layer][rows, :, :used]

    def drop(self, finished: Sequence[int]) -> list[int]:
        """Remove rows; return, for each remaining row in its new position, its old index."""
        gone = set(finished)
        keep = self.rows - len(gone)
        holes = [i for i in range(keep) if i in gone]
        movers = [i for i in range(keep, self.rows) if i not in gone]
        order = list(range(keep))
        for hole, mover in zip(holes, movers, strict=True):
            order[hole] = mover
        if holes:
            dst, src = torch.tensor(holes), torch.tensor(movers)
            for buffer in self.k + self.v:
                buffer[dst] = buffer[src]
        self.rows = keep
        return order
