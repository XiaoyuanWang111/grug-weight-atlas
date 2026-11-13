"""Override tensors of the swap test, read from the full-strength patch file.

The patch (`gates_full_strength.safetensors`) holds two groups of tensors computed at full merge strength:
`attention_gates` (per layer: `attn_gated_norm.down_proj`, `attn_gated_norm.up_proj` and `self_attn.attn_gate`,
78 tensors for 26 layers) and `router_weights` (per layer: `mlp.router.weight`, 26 tensors). The groups are
selected by name rules; on the published patch the rules reproduce the groups of its manifest exactly.

Arms of phase 1: A uses no override, B the attention gates, C the router weights, D both groups. Phase 2 adds arms
P, N and G over a merge built in memory (`swaptest.merge`); their overrides come from other checkpoints, not from
the patch. Every arm letter has a stable integer id, its position in ARM_ORDER.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

GROUP_PATTERNS: dict[str, re.Pattern[str]] = {
    "attention_gates": re.compile(
        r"model\.layers\.(\d+)\.(?:attn_gated_norm\.(?:down|up)_proj|self_attn\.attn_gate)\.weight"
    ),
    "router_weights": re.compile(r"model\.layers\.(\d+)\.mlp\.router\.weight"),
}
TENSORS_PER_LAYER = {"attention_gates": 3, "router_weights": 1}
ARM_GROUPS: dict[str, tuple[str, ...]] = {
    "A": (),
    "B": ("attention_gates",),
    "C": ("router_weights",),
    "D": ("attention_gates", "router_weights"),
}
PHASE1_ARMS = "ABCD"
ARM_ORDER = PHASE1_ARMS + "PNG"


def arm_index(arm: str) -> int:
    """Stable integer id of an arm letter."""
    if len(arm) != 1 or arm not in ARM_ORDER:
        raise ValueError(f"unknown arm {arm!r}; arms are {ARM_ORDER}")
    return ARM_ORDER.index(arm)


def group_names(names: Iterable[str], group: str, num_layers: int) -> list[str]:
    """Names of one group, checked to hold exactly the expected tensors of every layer."""
    pattern = GROUP_PATTERNS[group]
    selected = sorted(name for name in names if pattern.fullmatch(name))
    per_layer = Counter(int(pattern.fullmatch(name).group(1)) for name in selected)
    expected = {layer: TENSORS_PER_LAYER[group] for layer in range(num_layers)}
    if dict(per_layer) != expected:
        raise ValueError(
            f"group {group}: expected {TENSORS_PER_LAYER[group]} tensors in each of {num_layers} layers "
            f"({TENSORS_PER_LAYER[group] * num_layers} in all), found {len(selected)}"
        )
    return selected


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """The patch manifest (the published one ships with this package)."""
    if path is None:
        return json.loads(resources.files("swaptest").joinpath("data/patch_manifest.json").read_text("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest_groups(manifest: dict[str, Any], names: Iterable[str], num_layers: int) -> None:
    """The name rules must select exactly the manifest's groups."""
    names = list(names)
    for group in GROUP_PATTERNS:
        if sorted(manifest["groups"][group]) != group_names(names, group, num_layers):
            raise ValueError(f"name rules and manifest disagree on group {group}")


@dataclass(frozen=True)
class Patch:
    """Override tensors by group."""

    groups: dict[str, dict[str, torch.Tensor]]

    def arm_tables(self, arms: str) -> dict[int, dict[str, torch.Tensor]]:
        """For each arm id, the tensors that replace the base checkpoint's."""
        return {
            arm_index(arm): {name: t for group in ARM_GROUPS[arm] for name, t in self.groups[group].items()}
            for arm in arms
        }


def load_patch(
    path: Path, base: dict[str, torch.Tensor], num_layers: int, manifest: dict[str, Any] | None = None
) -> Patch:
    """Read the patch and check it against the base weights.

    Every patch tensor must replace a base tensor of the same shape and dtype and must differ from it (a patch
    equal to the base would make the arms identical). With a manifest, the groups must match it.
    """
    tensors = load_file(str(path))
    groups = {g: {name: tensors[name] for name in group_names(tensors, g, num_layers)} for g in GROUP_PATTERNS}
    leftover = set(tensors) - {name for group in groups.values() for name in group}
    if leftover:
        raise ValueError(f"patch holds tensors outside the two groups: {sorted(leftover)[:10]}")
    if manifest is not None:
        check_manifest_groups(manifest, tensors, num_layers)
    identical = []
    for name, tensor in tensors.items():
        if name not in base:
            raise ValueError(f"patch tensor {name} is not in the checkpoint")
        if tensor.shape != base[name].shape or tensor.dtype != base[name].dtype:
            raise ValueError(
                f"patch tensor {name} is {tuple(tensor.shape)} {tensor.dtype}, "
                f"checkpoint has {tuple(base[name].shape)} {base[name].dtype}"
            )
        if torch.equal(tensor, base[name]):
            identical.append(name)
    if identical:
        raise ValueError(f"{len(identical)} patch tensors equal the checkpoint's: {identical[:5]}")
    return Patch(groups)
