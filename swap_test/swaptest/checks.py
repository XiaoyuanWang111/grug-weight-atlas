"""Numerical self-checks and the speed model shared by the small-model validation and the 67B checks."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from typing import Any

import torch

from swaptest.chat import ChatTokenizer
from swaptest.engine import left_pad, prefill
from swaptest.model import GrugModel
from swaptest.patch import arm_index
from swaptest.sampling import Sampler


def full_logits(model: GrugModel, ids: torch.Tensor, arm: torch.Tensor | None = None) -> torch.Tensor:
    """Logits [B, T, V] (float32) of one forward over all of `ids` [B, T]."""
    b, t = ids.shape
    cache = model.new_cache(b, t)
    return model.logits(model.forward(ids, cache, 0, torch.zeros(b, dtype=torch.long), arm)).float()


def cached_logits(
    model: GrugModel, ids: torch.Tensor, prefill_len: int, chunk: int, arm: torch.Tensor | None = None
) -> torch.Tensor:
    """Logits [B, T, V] from a chunked prefill of the first `prefill_len` tokens, then one token per step."""
    b, t = ids.shape
    cache, pad = model.new_cache(b, t), torch.zeros(b, dtype=torch.long)
    parts = [
        model.logits(model.forward(ids[:, c0 : min(c0 + chunk, prefill_len)], cache, c0, pad, arm)).float()
        for c0 in range(0, prefill_len, chunk)
    ]
    parts += [model.logits(model.forward(ids[:, s : s + 1], cache, s, pad, arm)).float() for s in range(prefill_len, t)]
    return torch.cat(parts, dim=1)


def compare_logits(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Max abs difference and top-1 agreement over all positions of two logit tensors [..., V]."""
    nan = bool(torch.isnan(a).any() or torch.isnan(b).any())
    return {
        "positions": a[..., 0].numel(),
        "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        "max_abs_diff": float((a - b).abs().nan_to_num(nan=float("inf")).max()),
        "nan": nan,
    }


def mixed_vs_alone(model: GrugModel, prompt: list[int], steps: int, arms: str) -> dict[str, Any]:
    """Arm A decoded alone against arm A inside a batch of `arms`, on the same tokens.

    Arm A alone decodes greedily for `steps` tokens; the batch is then fed the same tokens in every row.
    Returns the comparison for arm A, and for each other arm its largest logit difference from arm A.
    """

    def decode(rows: str, forced: list[int] | None) -> tuple[torch.Tensor, list[int]]:
        ids = torch.tensor(prompt).repeat(len(rows), 1)
        arm = torch.tensor([arm_index(a) for a in rows])
        pad = torch.zeros(len(rows), dtype=torch.long)
        cache = model.new_cache(len(rows), len(prompt) + steps)
        logits = [model.logits(prefill(model, cache, ids, pad, arm, 256, len(rows))).float()]
        tokens = []
        for s in range(steps):
            token = forced[s] if forced is not None else int(logits[-1][0].argmax())
            tokens.append(token)
            step_ids = torch.full((len(rows), 1), token)
            logits.append(model.logits(model.forward(step_ids, cache, len(prompt) + s, pad, arm)[:, -1]).float())
        return torch.stack(logits, dim=1), tokens

    alone, tokens = decode("A", None)
    mixed, _ = decode(arms, tokens)
    a = arms.index("A")
    result = compare_logits(alone[0], mixed[a])
    result["other_arms_max_abs_diff_from_A"] = {
        arm: float((mixed[i] - mixed[a]).abs().max()) for i, arm in enumerate(arms) if arm != "A"
    }
    return result


def long_text_ids(tokenizer: ChatTokenizer, prompts: list[dict[str, Any]], length: int) -> list[int]:
    """`length` tokens of real text: a start token, then GSM8K questions and solutions."""
    text = "\n\n".join(f"Question: {p['question']}\nAnswer: {p['answer']}" for p in prompts)
    ids = [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]
    if len(ids) < length:
        raise ValueError(f"the prompts give only {len(ids)} tokens of text, need {length}")
    return ids[:length]


def decode_rows(prompts: Sequence[list[int]], arms: str, batch: int) -> list[tuple[list[int], str]]:
    """`batch` (prompt, arm) rows laid out as the main run lays out a wave: prompt by prompt, the arms of one
    prompt next to each other; prompts are reused from the first when the batch needs more."""
    return [(prompts[i // len(arms) % len(prompts)], arms[i % len(arms)]) for i in range(batch)]


def time_decode(
    model: GrugModel,
    prompts: Sequence[list[int]],
    arms: str,
    batch: int,
    steps: int,
    chunk: int = 256,
    row_block: int = 16,
    sampler: Sampler | None = None,
) -> dict[str, float]:
    """Prefill `batch` rows (see `decode_rows`) and time `steps` decode steps.

    With `sampler`, every step samples as the main run does (its noise generation is part of the step time);
    otherwise the steps are greedy.
    """
    rows = decode_rows(prompts, arms, batch)
    ids, pad = left_pad([p for p, _ in rows])
    arm = torch.tensor([arm_index(a) for _, a in rows])
    cache = model.new_cache(batch, ids.shape[1] + steps)
    t0 = time.perf_counter()
    logits = model.logits(prefill(model, cache, ids, pad, arm, chunk, row_block))
    prefill_s = time.perf_counter() - t0
    prompt_ids = list(range(batch))
    t0 = time.perf_counter()
    for s in range(steps):
        if sampler is None:
            tokens = logits.float().argmax(-1, keepdim=True)
        else:
            tokens = sampler(logits, prompt_ids, [s] * batch)[:, None]
        logits = model.logits(model.forward(tokens, cache, ids.shape[1] + s, pad, arm)[:, -1])
    step_s = (time.perf_counter() - t0) / steps
    real_tokens = int((ids.shape[1] - pad).sum())
    return {
        "batch": batch,
        "prefill_s": prefill_s,
        "prefill_tokens": real_tokens,
        "step_s": step_s,
        "tokens_per_s": batch / step_s,
    }


class SpeedModel:
    """Decode-step time t(b) = U(b) * layers * (expert_bytes / BW + c) + dense_bytes / BW + kv_bytes / BW.

    U(b) = E * (1 - (1 - K / E) ** b) is the expected number of distinct experts per layer for b tokens under
    uniform routing; BW (bytes/s) and c (seconds per expert call) are fitted to measured step times.
    """

    def __init__(self, model: GrugModel) -> None:
        cfg, w = model.cfg, model.w
        expert_names = {
            f"model.layers.{i}.mlp.experts.{n}_proj.weight"
            for i in range(cfg.num_layers)
            for n in ("gate", "up", "down")
        }
        self.cfg = cfg
        self.expert_bytes = sum(w[n].numel() * w[n].element_size() for n in expert_names) / (
            cfg.num_layers * cfg.num_experts
        )
        self.dense_bytes = sum(
            t.numel() * t.element_size()
            for n, t in w.items()
            if n not in expert_names and n != "model.embed_tokens.weight"
        )
        elem = w["model.embed_tokens.weight"].element_size()
        self.kv_bytes_per_slot = 2 * cfg.num_kv_heads * cfg.head_dim * elem
        self.bandwidth = math.nan
        self.call_s = math.nan

    def distinct_experts(self, b: int) -> float:
        e, k = self.cfg.num_experts, self.cfg.top_k
        return e * (1 - (1 - k / e) ** b)

    def kv_bytes(self, context: int) -> float:
        windows = [
            context if self.cfg.is_long(i) else min(context, self.cfg.sliding_window)
            for i in range(self.cfg.num_layers)
        ]
        return sum(windows) * self.kv_bytes_per_slot

    def fit(self, measurements: Sequence[tuple[int, float]]) -> None:
        """Least squares for (1 / BW, c) from (batch, seconds per step) pairs at short context."""
        rows = []
        for b, seconds in measurements:
            calls = self.distinct_experts(b) * self.cfg.num_layers
            rows.append((calls * self.expert_bytes + self.dense_bytes, calls, seconds))
        saa = sum(a * a for a, _, _ in rows)
        sac = sum(a * c for a, c, _ in rows)
        scc = sum(c * c for _, c, _ in rows)
        say = sum(a * y for a, _, y in rows)
        scy = sum(c * y for _, c, y in rows)
        det = saa * scc - sac * sac
        inv_bw, self.call_s = (say * scc - scy * sac) / det, (saa * scy - sac * say) / det
        self.bandwidth = 1.0 / inv_bw if inv_bw > 0 else math.inf

    def step_seconds(self, contexts: Sequence[int]) -> float:
        b = len(contexts)
        calls = self.distinct_experts(b) * self.cfg.num_layers
        traffic = calls * self.expert_bytes + self.dense_bytes + sum(self.kv_bytes(c) for c in contexts)
        return traffic / self.bandwidth + calls * self.call_s

    def project_hours(
        self,
        runaway: Sequence[float],
        per_arm: int,
        cap: int,
        prompt_len: int,
        prefill_tokens_per_s: float,
        batch_size: int | None = None,
        seed: int = 0,
    ) -> float:
        """Simulated hours of the main run: each arm's replies run to the cap with that arm's probability in
        `runaway`, otherwise have a log-normal length (median 600 tokens, sigma 0.7).

        Rows are ordered as the run orders them (prompt by prompt, arms adjacent) and decoded in waves of
        `batch_size` rows (default: all rows in one wave); the waves run one after the other.
        """
        rng = random.Random(seed)
        lengths = [
            cap if rng.random() < p else min(cap, int(math.exp(rng.gauss(math.log(600), 0.7))))
            for _ in range(per_arm)
            for p in runaway
        ]
        size = batch_size or len(lengths)
        seconds, stride = len(lengths) * prompt_len / prefill_tokens_per_s, 16
        for w0 in range(0, len(lengths), size):
            wave = lengths[w0 : w0 + size]
            for t in range(1, cap + 1, stride):
                active = sum(x >= t for x in wave)
                if not active:
                    break
                seconds += stride * self.step_seconds([prompt_len + t] * active)
        return seconds / 3600
