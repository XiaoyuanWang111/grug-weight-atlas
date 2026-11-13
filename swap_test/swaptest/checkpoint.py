"""Load a Grug checkpoint from a directory of safetensors shards without changing any stored dtype.

The public 67B exports store every tensor in bfloat16 except the 26 router biases (`*.mlp.router.bias`), which
are float32; the small Snowflake exports store the router biases in bfloat16 as well. The model computes the
router in float32 either way, so tensors are kept exactly as stored.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

from swaptest.config import GATED_NORM_RANK, ModelConfig

ROUTER_BIAS_SUFFIX = ".mlp.router.bias"


def shard_files(directory: Path) -> list[Path]:
    """Shards named by `model.safetensors.index.json`, or every `*.safetensors` file when there is no index."""
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return sorted({directory / name for name in weight_map.values()})
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {directory}")
    return files


def load_weights(directory: Path, copy: bool = True) -> dict[str, torch.Tensor]:
    """Read every shard into memory; tensor names must be unique and match the index when there is one.

    `safetensors.torch.load_file` returns tensors backed by a private memory map of the shard: their pages are
    read lazily on first use and, being clean file pages, the kernel may drop them again under memory pressure.
    With `copy` (the default) each shard's tensors are copied into ordinary process memory and the map is
    released before the next shard is opened, so the whole model is resident after loading, later decode steps
    never wait for the disk, and at most one shard is held twice at any moment.
    """
    weights: dict[str, torch.Tensor] = {}
    for path in shard_files(directory):
        for name, tensor in load_file(str(path)).items():
            if name in weights:
                raise ValueError(f"tensor {name} appears in more than one shard")
            weights[name] = tensor.clone() if copy else tensor
    index = directory / "model.safetensors.index.json"
    if index.exists():
        listed = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
        if listed != set(weights):
            raise ValueError(f"index and shards differ: {sorted(listed ^ set(weights))[:10]}")
    return weights


def expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Name and shape of every tensor of a schema-1 checkpoint."""
    d, v, r = cfg.hidden_size, cfg.vocab_size, GATED_NORM_RANK
    q, kv = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
    e, i, s = cfg.num_experts, cfg.moe_intermediate_size, cfg.shared_intermediate_size
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (v, d),
        "lm_head.weight": (v, d),
        "model.embed_norm.weight": (d,),
        "model.norm.weight": (d,),
    }
    gated = ("model.embed_gated_norm", "model.final_gated_norm")
    for layer in range(cfg.num_layers):
        p = f"model.layers.{layer}"
        gated += (f"{p}.attn_gated_norm", f"{p}.mlp_gated_norm")
        shapes |= {
            f"{p}.input_layernorm.weight": (d,),
            f"{p}.post_attention_layernorm.weight": (d,),
            f"{p}.self_attn.q_proj.weight": (q, d),
            f"{p}.self_attn.k_proj.weight": (kv, d),
            f"{p}.self_attn.v_proj.weight": (kv, d),
            f"{p}.self_attn.o_proj.weight": (d, q),
            f"{p}.self_attn.attn_gate.weight": (cfg.num_heads, d),
            f"{p}.mlp.router.weight": (e, d),
            f"{p}{ROUTER_BIAS_SUFFIX}": (e,),
            f"{p}.mlp.experts.gate_proj.weight": (e, i, d),
            f"{p}.mlp.experts.up_proj.weight": (e, i, d),
            f"{p}.mlp.experts.down_proj.weight": (e, d, i),
            f"{p}.shared_expert.gate_proj.weight": (s, d),
            f"{p}.shared_expert.up_proj.weight": (s, d),
            f"{p}.shared_expert.down_proj.weight": (d, s),
        }
    for prefix in gated:
        shapes[f"{prefix}.down_proj.weight"] = (r, d)
        shapes[f"{prefix}.up_proj.weight"] = (d, r)
    return shapes


def check_weights(weights: dict[str, torch.Tensor], cfg: ModelConfig) -> torch.dtype:
    """Check names, shapes and dtypes; return the compute dtype (that of the embedding).

    Every tensor must have the compute dtype except router biases, which may also be float32.
    """
    expected = expected_shapes(cfg)
    missing, unexpected = sorted(set(expected) - set(weights)), sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise ValueError(f"checkpoint and model keys differ: missing {missing[:10]}, unexpected {unexpected[:10]}")
    wrong = [(n, tuple(weights[n].shape), s) for n, s in expected.items() if tuple(weights[n].shape) != s]
    if wrong:
        raise ValueError(f"shape mismatch (name, found, expected): {wrong[:10]}")
    dtype = weights["model.embed_tokens.weight"].dtype
    for name, tensor in weights.items():
        allowed = {dtype, torch.float32} if name.endswith(ROUTER_BIAS_SUFFIX) else {dtype}
        if tensor.dtype not in allowed:
            raise ValueError(f"{name} has dtype {tensor.dtype}, expected one of {sorted(map(str, allowed))}")
    return dtype


def dtype_counts(weights: dict[str, torch.Tensor]) -> dict[str, int]:
    """Number of stored values per dtype, for the load report."""
    counts: Counter[str] = Counter()
    for tensor in weights.values():
        counts[str(tensor.dtype).removeprefix("torch.")] += tensor.numel()
    return dict(counts)


def cast_weights(weights: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Copy with every tensor in `dtype`, except router biases, which become float32 (their compute dtype)."""
    return {
        name: tensor.to(torch.float32 if name.endswith(ROUTER_BIAS_SUFFIX) else dtype)
        for name, tensor in weights.items()
    }
