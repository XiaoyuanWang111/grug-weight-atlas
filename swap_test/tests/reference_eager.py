"""Full-sequence eager forward transcribed from MarinSkyRL's `skyrl_train/models/grug_moe.py` (marin 80004cc06).

Kept deliberately literal for tests: key/value heads expanded with `repeat_interleave`, float32 scores with a
-1e9 fill, softmax weights cast to the value dtype, and a loop over every expert index as in `forward_eager`.
No cache, no padding, no overrides.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from swaptest.config import ModelConfig


def _rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    f = x.float()
    y = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
    return (y * w.float() if w is not None else y).to(x.dtype)


def _gated(x: torch.Tensor, w: dict, p: str) -> torch.Tensor:
    gate = torch.sigmoid(F.linear(F.silu(F.linear(x, w[f"{p}.down_proj.weight"])), w[f"{p}.up_proj.weight"]))
    return x * gate.to(x.dtype)


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    rot = x.shape[-1] // 2
    half = rot // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    ang = pos.float().unsqueeze(-1) * inv
    cos, sin = ang.cos().unsqueeze(2), ang.sin().unsqueeze(2)
    a, b = x[..., :rot].float().chunk(2, dim=-1)
    return torch.cat((torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype), x[..., rot:]), -1)


def reference_logits(cfg: ModelConfig, w: dict[str, torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
    b, t = ids.shape
    pos = torch.arange(t).unsqueeze(0).expand(b, t)
    h = _gated(
        _rms(w["model.embed_tokens.weight"][ids], w["model.embed_norm.weight"], cfg.rms_norm_eps),
        w,
        "model.embed_gated_norm",
    )
    rep = cfg.num_heads // cfg.num_kv_heads
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        x = _gated(_rms(h, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps), w, f"{p}.attn_gated_norm")
        q = F.linear(x, w[f"{p}.self_attn.q_proj.weight"]).view(b, t, cfg.num_heads, cfg.head_dim)
        k = F.linear(x, w[f"{p}.self_attn.k_proj.weight"]).view(b, t, cfg.num_kv_heads, cfg.head_dim)
        v = F.linear(x, w[f"{p}.self_attn.v_proj.weight"]).view(b, t, cfg.num_kv_heads, cfg.head_dim)
        q, k = _rms(q, None, 1e-6), _rms(k, None, 1e-6)
        long = cfg.is_long(i)
        if not long:
            q, k = _rope(q, pos, cfg.rope_theta), _rope(k, pos, cfg.rope_theta)
        q = q * (cfg.qk_mult * (cfg.qk_mult_long_scale if long else 1.0))
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float() / math.sqrt(cfg.head_dim), k.float())
        qp, kp = torch.arange(t).view(t, 1), torch.arange(t).view(1, t)
        allowed = kp <= qp
        if not long:
            allowed = allowed & (kp >= qp - (cfg.sliding_window - 1))
        scores = torch.where(allowed, scores, torch.tensor(-1e9))
        out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, -1).to(v.dtype), v)
        dot = (out * v).sum(-1, keepdim=True)
        out = out - (dot / (v.square().sum(-1, keepdim=True) + 1e-6)) * v
        gate = 2.0 * torch.sigmoid(F.linear(x, w[f"{p}.self_attn.attn_gate.weight"])).unsqueeze(-1)
        h = h + F.linear((out * gate.to(out.dtype)).reshape(b, t, -1), w[f"{p}.self_attn.o_proj.weight"])

        m = _gated(_rms(h, w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps), w, f"{p}.mlp_gated_norm")
        flat = m.reshape(-1, m.shape[-1])
        logits = F.linear(flat.float(), w[f"{p}.mlp.router.weight"].float())
        biased = logits + w[f"{p}.mlp.router.bias"].float()
        sel = torch.sort(biased, dim=-1, descending=True, stable=True).indices[:, : cfg.top_k]
        cw = torch.sigmoid(torch.gather(logits, -1, sel))
        cw = (cw * (2.5 / (cw.sum(-1, keepdim=True) + 1e-9))).to(flat.dtype)
        routed = torch.zeros_like(flat, dtype=torch.float32)
        for e in range(cfg.num_experts):
            tok, slot = torch.where(sel == e)
            if tok.numel() == 0:
                continue
            xe = flat.index_select(0, tok)
            ye = F.linear(
                F.silu(F.linear(xe, w[f"{p}.mlp.experts.gate_proj.weight"][e]))
                * F.linear(xe, w[f"{p}.mlp.experts.up_proj.weight"][e]),
                w[f"{p}.mlp.experts.down_proj.weight"][e],
            )
            routed.index_add_(0, tok, (ye * cw[tok, slot].unsqueeze(-1)).float())
        shared = F.linear(
            F.silu(F.linear(m, w[f"{p}.shared_expert.gate_proj.weight"]))
            * F.linear(m, w[f"{p}.shared_expert.up_proj.weight"]),
            w[f"{p}.shared_expert.down_proj.weight"],
        )
        h = h + routed.to(flat.dtype).view_as(h) + shared
    h = _gated(_rms(h, w["model.norm.weight"], cfg.rms_norm_eps), w, "model.final_gated_norm")
    return F.linear(h, w["lm_head.weight"]).float()
