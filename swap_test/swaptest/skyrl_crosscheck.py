"""Logits of MarinSkyRL's eager `GrugMoeForCausalLM`, for comparison with this port.

`skyrl_src` is a directory holding `skyrl_train/models/grug_moe.py`, its helper modules (`ep_gradient.py`,
`grug_query_bias.py`, `layers/moe_routing.py`, `router_instrumentation.py`) and `skyrl_train/utils/
flash_attention.py`, copied from Marin at the pinned commit without the package `__init__` files, so they import
as namespace packages and the full MarinSkyRL package is not needed. It requires `transformers` 5 and `loguru`.
The model is built on the meta device and run with the checkpoint tensors through `functional_call`, as Marin's
`experiments/weight_merging/calibration_model.py` does, with the eager attention backend.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch.func import functional_call


def skyrl_logits(
    skyrl_src: Path, config: dict[str, Any], weights: dict[str, torch.Tensor], ids: torch.Tensor
) -> torch.Tensor:
    """Float32 logits [B, T, V] of the reference model for `ids` [B, T] (no padding)."""
    if str(skyrl_src) not in sys.path:
        sys.path.insert(0, str(skyrl_src))
    grug = importlib.import_module("skyrl_train.models.grug_moe")
    model_config = grug.GrugMoeConfig(**config)
    model_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = grug.GrugMoeForCausalLM(model_config).eval().requires_grad_(False)
    expected = set(model.state_dict())
    if expected != set(weights):
        raise ValueError(f"reference model and checkpoint keys differ: {sorted(expected ^ set(weights))[:10]}")
    out = functional_call(model, weights, args=(), kwargs={"input_ids": ids})
    return out.logits.float()
