"""Load a model with the overrides of a set of arms, with a short load report.

Phase 1 loads a checkpoint from disk and takes the overrides from the patch file. Phase 2 builds a merge in memory
from a spec file and reads each arm's overrides from the checkpoint the spec names for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from swaptest.checkpoint import check_weights, dtype_counts, load_weights
from swaptest.config import ModelConfig, stop_token_ids
from swaptest.merge import Checkpoint, MergeSpec, build_merge
from swaptest.model import GrugModel, Overrides
from swaptest.patch import ARM_GROUPS, arm_index, check_manifest_groups, group_names, load_manifest, load_patch
from swaptest.records import sha256_file


@dataclass
class LoadedModel:
    cfg: ModelConfig
    weights: dict[str, torch.Tensor]
    model: GrugModel
    stop_ids: list[int]
    patch_sha256: str | None
    spec_sha256: str | None = None


def memory_gib() -> dict[str, float] | None:
    """Resident set and peak resident set of this process, and the machine's total and available memory, in GiB
    (Linux only; None elsewhere)."""
    try:
        status = Path("/proc/self/status").read_text()
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return None

    def gib(text: str, key: str) -> float:
        line = next(line for line in text.splitlines() if line.startswith(key + ":"))
        return int(line.split()[1]) / 2**20  # the kernel reports kB

    return {
        "rss": gib(status, "VmRSS"),
        "peak_rss": gib(status, "VmHWM"),
        "total": gib(meminfo, "MemTotal"),
        "available": gib(meminfo, "MemAvailable"),
    }


def memory_report() -> str:
    """Resident set of this process and memory available on the machine (Linux only)."""
    m = memory_gib()
    if m is None:
        return "memory report: not available on this platform"
    return (
        f"memory: process holds {m['rss']:.1f} GiB (peak {m['peak_rss']:.1f} GiB); machine has "
        f"{m['available']:.1f} of {m['total']:.1f} GiB available"
    )


def weight_gib(weights: dict[str, torch.Tensor]) -> float:
    return sum(t.numel() * t.element_size() for t in weights.values()) / 2**30


def load_model(checkpoint: Path, patch: Path | None, arms: str, log: Callable[[str], None]) -> LoadedModel:
    """Load weights as stored, check them, and build the overrides that `arms` need from the patch.

    When the patch is the published file (its SHA-256 equals the packaged manifest's), its groups must also
    equal the manifest's groups.
    """
    unknown = sorted(set(arms) - set(ARM_GROUPS))
    if unknown:
        raise ValueError(f"arms {unknown} are not arms of a checkpoint with a patch; they need a merge spec")
    t0 = time.monotonic()
    cfg = ModelConfig.from_checkpoint(checkpoint)
    weights = load_weights(checkpoint)
    dtype = check_weights(weights, cfg)
    log(
        f"loaded {len(weights)} tensors ({weight_gib(weights):.2f} GiB) in {time.monotonic() - t0:.0f} s; "
        f"compute dtype {dtype}; values per dtype {dtype_counts(weights)}"
    )
    overrides, patch_sha256 = None, None
    if any(ARM_GROUPS[arm] for arm in arms):
        if patch is None:
            raise ValueError(f"arms {arms} need a patch file")
        patch_sha256 = sha256_file(patch)
        loaded = load_patch(patch, weights, cfg.num_layers)
        manifest = load_manifest()
        if patch_sha256 == manifest["sha256"]:
            check_manifest_groups(manifest, [n for g in loaded.groups.values() for n in g], cfg.num_layers)
            log("patch is the published file; its groups equal the manifest's")
        else:
            log(f"patch SHA-256 {patch_sha256} is not the published file's; manifest check skipped")
        sizes = ", ".join(f"{group} {len(names)}" for group, names in loaded.groups.items())
        log(f"patch: {sizes}; every tensor matches the checkpoint's shape and dtype and differs from it")
        overrides = Overrides.from_arm_tables(loaded.arm_tables(arms))
    log(memory_report())
    return LoadedModel(cfg, weights, GrugModel(cfg, weights, overrides), stop_token_ids(checkpoint), patch_sha256)


def arm_overrides(
    spec: MergeSpec, weights: dict[str, torch.Tensor], arms: str, num_layers: int
) -> dict[int, dict[str, torch.Tensor]]:
    """For each arm id, the tensors that replace the merged model's: the arm's group, read from its source.

    Every override must have the shape and dtype of the merged tensor and differ from it.
    """
    unknown = sorted(set(arms) - set(spec.arms))
    if unknown:
        raise ValueError(f"arms {unknown} are not defined in the merge spec (it defines {sorted(spec.arms)})")
    tables: dict[int, dict[str, torch.Tensor]] = {}
    for arm in arms:
        override = spec.arms[arm]
        table: dict[str, torch.Tensor] = {}
        if override is not None:
            source = Checkpoint(override.source)
            for name in group_names(weights, override.group, num_layers):
                tensor = source.read(name)
                if tensor.shape != weights[name].shape or tensor.dtype != weights[name].dtype:
                    raise ValueError(f"arm {arm}: {name} from {override.source} does not match the merged tensor")
                if torch.equal(tensor, weights[name]):
                    raise ValueError(f"arm {arm}: {name} from {override.source} equals the merged tensor")
                table[name] = tensor
        tables[arm_index(arm)] = table
    return tables


def load_merged_model(checkpoint: Path, spec_path: Path, arms: str, log: Callable[[str], None]) -> LoadedModel:
    """Build the merge of a spec file in memory and the overrides of `arms`; `checkpoint` supplies the configuration
    and the stop tokens (and, to callers, the tokenizer and chat template)."""
    spec = MergeSpec.load(spec_path)
    cfg = ModelConfig.from_checkpoint(checkpoint)
    t0 = time.monotonic()
    weights = build_merge(spec, log)
    dtype = check_weights(weights, cfg)
    log(
        f"built the merge in memory: {len(weights)} tensors ({weight_gib(weights):.2f} GiB) in "
        f"{time.monotonic() - t0:.0f} s; compute dtype {dtype}; values per dtype {dtype_counts(weights)}"
    )
    tables = arm_overrides(spec, weights, arms, cfg.num_layers)
    for arm in arms:
        source = spec.arms[arm]
        if source is None:
            log(f"arm {arm}: no override")
        else:
            log(f"arm {arm}: {len(tables[arm_index(arm)])} {source.group} tensors from {source.source}")
    overrides = Overrides.from_arm_tables(tables) if any(tables.values()) else None
    log(memory_report())
    model = GrugModel(cfg, weights, overrides)
    return LoadedModel(cfg, weights, model, stop_token_ids(checkpoint), None, sha256_file(spec_path))
