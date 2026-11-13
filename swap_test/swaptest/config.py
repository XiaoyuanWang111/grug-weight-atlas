"""Model configuration of a Grug MoE checkpoint, read from its Hugging Face `config.json`.

Only the June "Snowball" recipe with stacked experts (artifact schema 1) is supported, which is what the eager
PyTorch reference in MarinSkyRL (`skyrl_train/models/grug_moe.py`) runs. Field aliases are resolved the way that
reference resolves them, and configurations that select other architectures are rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GLOBAL_EVERY = 4
GATED_NORM_RANK = 128
SUPPORTED_SCHEMA = 1

# Fields whose non-default values select architectures this port does not implement.
_REQUIRED_DEFAULTS: dict[str, Any] = {
    "latent_dim": None,
    "local_kv_heads": None,
    "global_kv_heads": None,
    "log_scaling_n_floor": None,
    "log_scaling_alpha": None,
    "log_scaling_power": None,
    "num_shared_experts": 1,
    "global_every": GLOBAL_EVERY,
    "rope_fused": False,
    "sconv": False,
    "tie_word_embeddings": False,
    "disable_pko": True,
    "disable_long_rope": True,
    "grugmoe_attention_mode": "production",
}


@dataclass(frozen=True)
class ModelConfig:
    """Shapes and constants of one Grug MoE model."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    shared_intermediate_size: int
    sliding_window: int
    qk_mult: float
    qk_mult_long_scale: float
    rope_theta: float
    rms_norm_eps: float

    def is_long(self, layer: int) -> bool:
        """Full causal attention without RoPE: every fourth layer and the last one."""
        return (layer + 1) % GLOBAL_EVERY == 0 or layer == self.num_layers - 1

    @property
    def long_layers(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_long(i)]

    @property
    def kv_group(self) -> int:
        """Query heads per key/value head."""
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        if raw.get("model_type") != "grug_moe":
            raise ValueError(f"model_type must be grug_moe, got {raw.get('model_type')!r}")
        architectures = raw.get("architectures")
        if architectures not in (None, ["GrugMoeForCausalLM"]):
            raise ValueError(f"unsupported architectures {architectures!r}")
        schema = int(raw.get("grugmoe_artifact_schema_version", SUPPORTED_SCHEMA))
        if schema != SUPPORTED_SCHEMA:
            raise ValueError(f"only artifact schema {SUPPORTED_SCHEMA} (stacked experts) is supported, got {schema}")
        for key, default in _REQUIRED_DEFAULTS.items():
            if raw.get(key, default) != default:
                raise ValueError(f"unsupported configuration {key}={raw[key]!r} (this port requires {default!r})")
        if raw.get("use_cache"):
            raise ValueError("config requests use_cache, which the reference model rejects")
        if "qk_mult" not in raw:
            raise ValueError("config has no qk_mult")
        cfg = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=_alias(raw, "hidden_size", "hidden_dim"),
            num_layers=_alias(raw, "num_hidden_layers", "num_layers"),
            num_heads=_alias(raw, "num_attention_heads", "num_heads"),
            num_kv_heads=_alias(raw, "num_key_value_heads", "num_kv_heads"),
            head_dim=int(raw.get("head_dim", 128)),
            num_experts=_alias(raw, "num_experts", "num_local_experts"),
            top_k=_alias(raw, "num_experts_per_tok", "num_experts_per_token"),
            moe_intermediate_size=_alias(raw, "moe_intermediate_size", "intermediate_size", "intermediate_dim"),
            shared_intermediate_size=_alias(raw, "shared_expert_intermediate_size", "shared_expert_intermediate_dim"),
            sliding_window=int(raw.get("sliding_window", 2048)),
            qk_mult=float(raw["qk_mult"]),
            qk_mult_long_scale=float(raw.get("qk_mult_long_scale", 1.0)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            rms_norm_eps=float(_alias(raw, "rms_norm_eps", "layer_norm_eps", default=1e-5)),
        )
        if cfg.num_heads % cfg.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if cfg.head_dim % 4:
            raise ValueError("head_dim must be divisible by four for half-RoPE")
        if cfg.top_k >= cfg.num_experts:
            raise ValueError("top-(K+1) routing needs num_experts_per_tok < num_experts")
        return cfg

    @classmethod
    def from_checkpoint(cls, directory: Path) -> ModelConfig:
        return cls.from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8")))


def _alias(raw: dict[str, Any], *names: str, default: float | None = None) -> Any:
    """Value of the first present alias; all present aliases must agree."""
    values = [raw[name] for name in names if raw.get(name) is not None]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"config aliases {names} disagree: {values}")
    if values:
        return values[0]
    if default is None:
        raise KeyError(f"config has none of {names}")
    return default


def stop_token_ids(directory: Path) -> list[int]:
    """End-of-turn token ids from `generation_config.json` (falling back to `config.json`)."""
    for name in ("generation_config.json", "config.json"):
        path = directory / name
        if path.exists():
            eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
            if eos is not None:
                return sorted({int(eos)} if isinstance(eos, int) else {int(e) for e in eos})
    raise ValueError(f"no eos_token_id in {directory}")
