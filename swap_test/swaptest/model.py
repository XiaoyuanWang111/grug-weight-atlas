"""CPU forward of the Grug / Snowball MoE with a preallocated KV cache and per-row weight overrides.

The arithmetic follows MarinSkyRL's eager reference `skyrl_train/models/grug_moe.py` (marin 80004cc06):
GrugMoeRMSNorm (l.382), the no-weight q/k RMS norm (l.395), GrugMoeGatedNorm (l.402), GrugMoeRouter.forward
(l.542: float32 router, top-(K+1) on biased logits, sigmoid combine weights renormalised to 2.5),
GrugMoeExperts.forward_eager (l.454), _apply_half_rope (l.680), GrugMoeAttention.forward (l.716: XSA and the
`2 * sigmoid` per-head gate) and GrugMoeDecoderLayer.forward (l.850). Float32 is used exactly where the reference
uses it. The attention core is `scaled_dot_product_attention` (float32 accumulation for bfloat16 inputs), where
the reference materialises float32 scores; the reference casts the softmax weights to the value dtype, so both
produce outputs in the activation dtype.

Rows of one batch may belong to different arms. An arm replaces a few small tensors (attention gated norms,
attention gates, router weights); every other weight, in particular all routed experts, is shared, so arms that
differ only in those tensors cost one pass over the experts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from swaptest.config import ModelConfig
from swaptest.kvcache import KVCache, entry_slots

QK_NORM_EPS = 1e-6
XSA_EPS = 1e-6
ROUTING_RENORM_SUM = 2.5
ROUTER_RENORM_EPS = 1e-9
ATTN_GATE_SCALE = 2.0

Variants = list[tuple[tuple[int, ...], torch.Tensor]]


@dataclass(frozen=True)
class Overrides:
    """For each overridden tensor name, the arms that replace it and the value they use."""

    variants: dict[str, Variants]

    @classmethod
    def from_arm_tables(cls, tables: Mapping[int, Mapping[str, torch.Tensor]]) -> Overrides:
        """Group arms that use the same tensor object, so a value shared by two arms is applied once."""
        variants: dict[str, Variants] = {}
        for arm, table in sorted(tables.items()):
            for name, tensor in table.items():
                entries = variants.setdefault(name, [])
                for i, (arms, value) in enumerate(entries):
                    if value is tensor:
                        entries[i] = ((*arms, arm), value)
                        break
                else:
                    entries.append(((arm,), tensor))
        return cls(variants)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    f = x.float()
    y = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def half_rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding on the first half of each head (NeoX halves); x [B, T, H, D], pos [B, T]."""
    rot = x.shape[-1] // 2
    half = rot // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    angles = pos.float().unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
    a, b = x[..., :rot].float().chunk(2, dim=-1)
    rotated = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)
    return torch.cat((rotated, x[..., rot:]), dim=-1)


def attention_mask(
    key_slots: torch.Tensor, start: int, count: int, pad: torch.Tensor, window: int | None
) -> torch.Tensor:
    """[B, T, S] mask: causal, inside the window for short layers, no left padding.

    A query may always see its own slot, so left-padding queries attend to themselves and stay finite.
    """
    q = torch.arange(start, start + count)[None, :, None]
    k = key_slots[None, None, :]
    allowed = (k <= q) & (k >= pad[:, None, None])
    if window is not None:
        allowed &= k >= q - (window - 1)
    return allowed | (k == q)


def attend(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Grouped-query attention. q [B, T, H, D]; keys, values [B, KVH, S, D]; mask [B, T, S] -> [B, T, H, D].

    Query head h uses key/value head h // (H / KVH), as `repeat_interleave` does in the reference.
    """
    b, t, h, d = q.shape
    kvh, s = keys.shape[1], keys.shape[2]
    rep = h // kvh
    qg = q.view(b, t, kvh, rep, d).permute(0, 2, 1, 3, 4).reshape(b, kvh, t * rep, d)
    m = mask[:, None, :, None, :].expand(b, 1, t, rep, s).reshape(b, 1, t * rep, s)
    out = F.scaled_dot_product_attention(qg, keys, values, attn_mask=m, scale=1.0 / math.sqrt(d))
    return out.view(b, kvh, t, rep, d).permute(0, 2, 1, 3, 4).reshape(b, t, h, d)


def _apply(x: torch.Tensor, base: torch.Tensor, variants: Variants, groups: dict[tuple[int, ...], torch.Tensor]):
    """x [N, in] @ base.T, with the rows of each overriding arm group recomputed with its own weight."""
    out = F.linear(x, base)
    for arms, weight in variants:
        rows = groups[arms]
        if rows.numel():
            out[rows] = F.linear(x[rows], weight)
    return out


class GrugModel:
    """Forward pass over a weight dict with the checkpoint's tensor names."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], overrides: Overrides | None = None):
        self.cfg = cfg
        self.w = weights
        self.dtype = weights["model.embed_tokens.weight"].dtype
        self.variants = overrides.variants if overrides is not None else {}
        for name, entries in self.variants.items():
            for _, value in entries:
                if value.shape != weights[name].shape or value.dtype != weights[name].dtype:
                    raise ValueError(f"override {name} does not match the checkpoint tensor's shape or dtype")
        self.arm_sets = sorted({arms for entries in self.variants.values() for arms, _ in entries})
        routers = [f"model.layers.{i}.mlp.router.weight" for i in range(cfg.num_layers)]
        self.router_w = {name: weights[name].float() for name in routers}
        self.router_variants = {
            name: [(arms, value.float()) for arms, value in self.variants[name]]
            for name in routers
            if name in self.variants
        }

    def new_cache(self, rows: int, max_len: int) -> KVCache:
        return KVCache(self.cfg, rows, max_len, self.dtype)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.w["lm_head.weight"])

    def forward(
        self,
        ids: torch.Tensor,
        cache: KVCache,
        start: int,
        pad: torch.Tensor,
        arm: torch.Tensor | None = None,
        rows: slice | None = None,
        trace: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Final hidden states [B, T, D] for tokens `ids` [B, T] at slots start .. start + T - 1.

        `rows` selects the cache rows (default: the first B); `pad` [B] is each row's left padding in slots;
        `arm` [B] is each row's arm id (default: no overrides). Keys and values are appended to the cache. With
        `trace`, the hidden state after the embedding, after every layer and after the final norm is appended.
        """
        b, t = ids.shape
        rows = slice(0, b) if rows is None else rows
        if rows.stop - rows.start != b or rows.stop > cache.rows:
            raise ValueError("rows must select exactly the batch's rows of the cache")
        pos = (torch.arange(start, start + t)[None, :] - pad[:, None]).clamp_min(0)
        groups = self._groups(arm, b, t)
        masks: dict[tuple[int, bool, int | None], torch.Tensor] = {}
        h = self.w["model.embed_tokens.weight"][ids]
        h = self._gated_norm(
            rms_norm(h, self.w["model.embed_norm.weight"], self.cfg.rms_norm_eps), "model.embed_gated_norm", groups
        )
        if trace is not None:
            trace.append(h)
        for layer in range(self.cfg.num_layers):
            p = f"model.layers.{layer}"
            a_in = self._gated_norm(
                rms_norm(h, self.w[f"{p}.input_layernorm.weight"], self.cfg.rms_norm_eps),
                f"{p}.attn_gated_norm",
                groups,
            )
            h = h + self._attention(a_in, layer, cache, rows, start, pos, pad, masks, groups)
            m_in = self._gated_norm(
                rms_norm(h, self.w[f"{p}.post_attention_layernorm.weight"], self.cfg.rms_norm_eps),
                f"{p}.mlp_gated_norm",
                groups,
            )
            flat = m_in.reshape(b * t, -1)
            h = h + self._moe(flat, layer, groups).view_as(h) + self._shared_expert(m_in, p)
            if trace is not None:
                trace.append(h)
        h = self._gated_norm(
            rms_norm(h, self.w["model.norm.weight"], self.cfg.rms_norm_eps), "model.final_gated_norm", groups
        )
        if trace is not None:
            trace.append(h)
        return h

    def _groups(self, arm: torch.Tensor | None, b: int, t: int) -> dict[tuple[int, ...], torch.Tensor]:
        """Token-row indices (rows flattened as [B * T]) of each group of arms that shares an override."""
        if not self.arm_sets:
            return {}
        if arm is None:
            arm = torch.zeros(b, dtype=torch.long)
        token_arm = arm.repeat_interleave(t)
        return {arms: torch.isin(token_arm, torch.tensor(arms)).nonzero().squeeze(1) for arms in self.arm_sets}

    def _linear(self, x: torch.Tensor, name: str, groups: dict[tuple[int, ...], torch.Tensor]) -> torch.Tensor:
        return _apply(x, self.w[name], self.variants.get(name, []), groups)

    def _gated_norm(self, x: torch.Tensor, prefix: str, groups: dict[tuple[int, ...], torch.Tensor]):
        rows = x.reshape(-1, x.shape[-1])
        hidden = F.silu(self._linear(rows, f"{prefix}.down_proj.weight", groups))
        gate = torch.sigmoid(self._linear(hidden, f"{prefix}.up_proj.weight", groups))
        return (rows * gate.to(rows.dtype)).view_as(x)

    def _shared_expert(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        w = self.w
        up = F.silu(F.linear(x, w[f"{prefix}.shared_expert.gate_proj.weight"])) * F.linear(
            x, w[f"{prefix}.shared_expert.up_proj.weight"]
        )
        return F.linear(up, w[f"{prefix}.shared_expert.down_proj.weight"])

    def _attention(self, x, layer, cache, rows, start, pos, pad, masks, groups) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        b, t, _ = x.shape
        p = f"model.layers.{layer}.self_attn"
        q = F.linear(x, w[f"{p}.q_proj.weight"]).view(b, t, cfg.num_heads, cfg.head_dim)
        k = F.linear(x, w[f"{p}.k_proj.weight"]).view(b, t, cfg.num_kv_heads, cfg.head_dim)
        v = F.linear(x, w[f"{p}.v_proj.weight"]).view(b, t, cfg.num_kv_heads, cfg.head_dim)
        q, k = rms_norm(q, None, QK_NORM_EPS), rms_norm(k, None, QK_NORM_EPS)
        long = cfg.is_long(layer)
        if not long:
            q, k = half_rope(q, pos, cfg.rope_theta), half_rope(k, pos, cfg.rope_theta)
        q = q * (cfg.qk_mult * (cfg.qk_mult_long_scale if long else 1.0))

        size = cache.sizes[layer]
        # Writing first is safe unless a multi-token chunk wraps the ring over entries its own queries still need.
        direct = t == 1 or start + t <= size
        if direct:
            cache.write(layer, rows, start, k, v)
            keys, values = cache.entries(layer, rows, start + t)
            key_slots = entry_slots(size, start + t)
        else:
            old_k, old_v = cache.entries(layer, rows, start)
            keys = torch.cat((old_k, k.transpose(1, 2)), dim=2)
            values = torch.cat((old_v, v.transpose(1, 2)), dim=2)
            key_slots = torch.cat((entry_slots(size, start), torch.arange(start, start + t)))
            cache.write(layer, rows, start, k, v)
        window = None if long else cfg.sliding_window
        key = (size, direct, window)
        if key not in masks:
            masks[key] = attention_mask(key_slots, start, t, pad, window)
        out = attend(q, keys, values, masks[key])

        v_self = v.repeat_interleave(cfg.kv_group, dim=2)  # value at the query's own position (XSA)
        dot = (out * v_self).sum(-1, keepdim=True)
        out = out - (dot / (v_self.square().sum(-1, keepdim=True) + XSA_EPS)) * v_self
        gate = ATTN_GATE_SCALE * torch.sigmoid(self._linear(x.reshape(b * t, -1), f"{p}.attn_gate.weight", groups))
        out = out * gate.view(b, t, cfg.num_heads, 1).to(out.dtype)
        return F.linear(out.reshape(b, t, -1), w[f"{p}.o_proj.weight"])

    def _moe(self, x: torch.Tensor, layer: int, groups: dict[tuple[int, ...], torch.Tensor]) -> torch.Tensor:
        """Dropless routed experts for token rows x [N, D]."""
        cfg, w = self.cfg, self.w
        p = f"model.layers.{layer}.mlp"
        name = f"{p}.router.weight"
        logits = _apply(x.float(), self.router_w[name], self.router_variants.get(name, []), groups)
        biased = logits + w[f"{p}.router.bias"].float()
        selected = torch.sort(biased, dim=-1, descending=True, stable=True).indices[:, : cfg.top_k]
        combine = torch.sigmoid(torch.gather(logits, -1, selected))
        combine = combine * (ROUTING_RENORM_SUM / (combine.sum(-1, keepdim=True) + ROUTER_RENORM_EPS))
        combine = combine.to(x.dtype)

        flat = selected.reshape(-1)
        order = torch.argsort(flat, stable=True)  # assignments grouped by expert, in expert order
        token = order // cfg.top_k
        counts = torch.bincount(flat, minlength=cfg.num_experts).tolist()
        xs = x.index_select(0, token)
        ys = torch.empty_like(xs)
        gate_w, up_w, down_w = (w[f"{p}.experts.{n}_proj.weight"] for n in ("gate", "up", "down"))
        first = 0
        for expert, count in enumerate(counts):
            if count:
                xe = xs[first : first + count]
                ys[first : first + count] = F.linear(
                    F.silu(F.linear(xe, gate_w[expert])) * F.linear(xe, up_w[expert]), down_w[expert]
                )
                first += count
        weighted = (ys * combine.reshape(-1)[order].unsqueeze(-1)).float()
        # index_add_ accumulates in index order, i.e. per token in increasing expert order, as the reference does.
        out = torch.zeros(x.shape, dtype=torch.float32).index_add_(0, token, weighted)
        return out.to(x.dtype)
