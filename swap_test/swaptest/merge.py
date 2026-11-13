"""Task-arithmetic merges of sharded safetensors checkpoints, computed in memory with Marin's arithmetic.

A merge is anchor + sum_i c_i (donor_i - anchor), tensor by tensor, computed as `marin.merging.arithmetic.
merge_tensor` computes method task_arithmetic (marin 80004cc06, density 1, scale 1): the anchor and each donor are
read as float32, each update (donor - anchor) is multiplied by its coefficient, the updates are added in donor
order to a tensor of zeros, the anchor is added last, and the result is cast to the anchor's stored dtype. Inputs
and results must be finite. (Marin multiplies the summed updates by the scale before adding the anchor; with
scale 1 that product is exact and is left out here.) The same operations run in the same order on slices of each
tensor along its first axis; element-wise arithmetic does not depend on the slicing, so every stored bit is the
same as for whole tensors, while only a few hundred megabytes are held besides the output.

Two options of `marin.merging.checkpoint.merge_checkpoint` are supported:
- dtype widening (`CheckpointSource.dtype_widenings`, marin #10597): tensors of a source whose names fully match
  an allow-listed pattern are stored in bfloat16 or float16 and are read as float32. The 67B experts store the
  router bias in bfloat16 while the anchor stores it in float32; without widening Marin rejects those tensors.
- per-tensor coefficients (`tensor_coefficients`): tensors whose names fully match a pattern use their own
  coefficients. With coefficients 0 the result is the anchor's value (a negative zero becomes a positive zero).

A spec file (JSON) names the anchor, the donors, the coefficients and these options, and optionally override arms:
an arm replaces the tensors of one patch group (for example the 78 attention-gate tensors) by the tensors of the
same names in another checkpoint. `build_merge` returns the merged model as a weight table and writes nothing.
`--check-against` recomputes a merge and compares it, shard by shard and bit for bit, with a merged checkpoint
on disk, reporting the first difference.

Usage:
  python -m swaptest.merge --base DIR --donor DIR [--donor DIR ...] --coefficients C1,C2,...
      [--widen PATTERN=bfloat16 ...] [--tensor-coefficients PATTERN=C1,C2,... ...]
      [--arm LETTER=none | --arm LETTER=GROUP:DIR ...] --write-spec SPEC.json
  python -m swaptest.merge --spec SPEC.json [--coefficients C1,C2,...] --check-against DIR [--json OUT.json]
      [--first-only] [--threads N]
--widen applies to every donor. Exit codes: 0 the spec was written, or the check found every stored value equal;
1 the check found a difference; 2 bad input; 3 the only differences are router biases whose reference values
equal the anchor's (the case for which the pre-registration rebuilds with router-bias coefficients 0).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from swaptest.checkpoint import ROUTER_BIAS_SUFFIX, shard_files
from swaptest.patch import ARM_ORDER, GROUP_PATTERNS
from swaptest.records import write_json_atomic

SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
WIDENABLE = {"bfloat16": torch.bfloat16, "float16": torch.float16}
INTEGER_VIEW = {2: torch.int16, 4: torch.int32}
SLICE_VALUES = 1 << 25  # values per slice: float32 work buffers of 128 MiB each
ROUTER_BIAS = re.compile(r"model\.layers\.\d+" + re.escape(ROUTER_BIAS_SUFFIX))


@dataclass(frozen=True)
class TensorInfo:
    """Where a tensor's bytes are: file, dtype, shape and the absolute offset of its first byte."""

    file: Path
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int

    @property
    def row_values(self) -> int:
        return math.prod(self.shape[1:])


def read_header(path: Path) -> dict[str, TensorInfo]:
    """Tensors of one safetensors file: an 8-byte little-endian header length, a JSON header, then the data."""
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] not in SAFETENSORS_DTYPES:
            raise ValueError(f"{path.name}: tensor {name} has dtype {entry['dtype']}, which merging does not support")
        dtype, shape = SAFETENSORS_DTYPES[entry["dtype"]], tuple(entry["shape"])
        start, end = entry["data_offsets"]
        if end - start != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"{path.name}: tensor {name} has {end - start} bytes for shape {shape} {entry['dtype']}")
        tensors[name] = TensorInfo(path, dtype, shape, 8 + length + start)
    return tensors


class Checkpoint:
    """Tensor index of a directory of safetensors shards; reads whole tensors or row ranges with plain file reads.

    Large sequential reads into preallocated tensors keep disk throughput high and leave no memory map behind.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.tensors: dict[str, TensorInfo] = {}
        for path in shard_files(directory):
            for name, info in read_header(path).items():
                if name in self.tensors:
                    raise ValueError(f"tensor {name} appears in more than one shard of {directory}")
                self.tensors[name] = info
        index = directory / "model.safetensors.index.json"
        if index.exists() and set(json.loads(index.read_text(encoding="utf-8"))["weight_map"]) != set(self.tensors):
            raise ValueError(f"index and shards of {directory} name different tensors")

    def names(self) -> list[str]:
        """Tensor names in storage order: by shard file, then by offset."""
        return sorted(self.tensors, key=lambda n: (self.tensors[n].file.name, self.tensors[n].offset))

    def read(self, name: str, start: int = 0, stop: int | None = None) -> torch.Tensor:
        """Rows start .. stop - 1 of a tensor along its first axis (the whole tensor by default)."""
        info = self.tensors[name]
        if not info.shape:
            count, offset, shape = 1, info.offset, ()
        else:
            stop = info.shape[0] if stop is None else stop
            count = (stop - start) * info.row_values
            offset = info.offset + start * info.row_values * info.dtype.itemsize
            shape = (stop - start, *info.shape[1:])
        out = torch.empty(count, dtype=info.dtype)
        if count:
            with info.file.open("rb") as f:
                f.seek(offset)
                got = f.readinto(out.view(torch.uint8).numpy())
            if got != count * info.dtype.itemsize:
                raise OSError(f"short read of {name} from {info.file}: {got} of {count * info.dtype.itemsize} bytes")
        return out.view(shape)


def merge_values(
    anchor: torch.Tensor, donors: Sequence[torch.Tensor], coefficients: Sequence[float], name: str
) -> torch.Tensor:
    """Task arithmetic on aligned tensors, exactly as `marin.merging.arithmetic.merge_tensor` (scale 1)."""
    if len(donors) != len(coefficients):
        raise ValueError("each donor needs one coefficient")
    for donor in donors:
        if donor.shape != anchor.shape or donor.dtype != anchor.dtype:
            raise ValueError(f"incompatible tensor {name}: shape or dtype differs")
    if not anchor.is_floating_point():
        raise ValueError(f"tensor {name} is not floating point")
    if not torch.isfinite(anchor).all() or any(not torch.isfinite(d).all() for d in donors):
        raise ValueError(f"nonfinite input: {name}")
    base = anchor.float()
    updates = []
    for coefficient, donor in zip(coefficients, donors, strict=True):
        update = donor.float() - base
        updates.append(update.mul_(coefficient))
    result = torch.zeros_like(base)
    for update in updates:
        result.add_(update)
    result = result.add_(base).to(anchor.dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"merge overflow: {name}")
    return result


@dataclass(frozen=True)
class Source:
    """A checkpoint directory and its widening allow-list (full-match pattern -> stored dtype name)."""

    path: Path
    widen: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmSource:
    """An override arm: the tensors of patch group `group`, read from the checkpoint at `source`."""

    group: str
    source: Path


@dataclass(frozen=True)
class MergeSpec:
    """Anchor, donors, coefficients, per-tensor coefficients (pattern -> coefficients) and override arms."""

    base: Source
    donors: tuple[Source, ...]
    coefficients: tuple[float, ...]
    tensor_coefficients: dict[str, tuple[float, ...]] = field(default_factory=dict)
    arms: dict[str, ArmSource | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.donors or len(self.coefficients) != len(self.donors):
            raise ValueError("give at least one donor and one coefficient per donor")
        for coefficients in (self.coefficients, *self.tensor_coefficients.values()):
            if len(coefficients) != len(self.donors) or any(not math.isfinite(c) or c < 0 for c in coefficients):
                raise ValueError(f"coefficients must be finite, nonnegative and one per donor: {coefficients}")
        for source in (self.base, *self.donors):
            if any(dtype not in WIDENABLE for dtype in source.widen.values()):
                raise ValueError(f"only {sorted(WIDENABLE)} tensors can be widened to float32: {source.widen}")
        for arm, override in self.arms.items():
            if arm not in ARM_ORDER or (override is not None and override.group not in GROUP_PATTERNS):
                raise ValueError(f"arm {arm!r}: arms are letters of {ARM_ORDER}, groups are {sorted(GROUP_PATTERNS)}")

    def with_coefficients(self, coefficients: Sequence[float]) -> MergeSpec:
        return MergeSpec(self.base, self.donors, tuple(coefficients), self.tensor_coefficients, self.arms)

    def to_json(self) -> dict[str, Any]:
        def source(s: Source) -> dict[str, Any]:
            return {"path": str(s.path), "widen": s.widen}

        return {
            "base": source(self.base),
            "donors": [source(d) for d in self.donors],
            "coefficients": list(self.coefficients),
            "tensor_coefficients": {p: list(c) for p, c in self.tensor_coefficients.items()},
            "arms": {a: None if o is None else {"group": o.group, "source": str(o.source)} for a, o in self.arms.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MergeSpec:
        def source(s: dict[str, Any]) -> Source:
            return Source(Path(s["path"]), dict(s.get("widen", {})))

        return cls(
            source(data["base"]),
            tuple(source(d) for d in data["donors"]),
            tuple(float(c) for c in data["coefficients"]),
            {p: tuple(float(x) for x in c) for p, c in data.get("tensor_coefficients", {}).items()},
            {
                a: None if o is None else ArmSource(o["group"], Path(o["source"]))
                for a, o in data.get("arms", {}).items()
            },
        )

    @classmethod
    def load(cls, path: Path) -> MergeSpec:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


class MergePlan:
    """The sources of a merge, checked to be aligned: the same tensor names, shapes and (after widening) dtypes."""

    def __init__(self, spec: MergeSpec) -> None:
        self.spec = spec
        self.base = Checkpoint(spec.base.path)
        self.donors = [Checkpoint(d.path) for d in spec.donors]
        names = set(self.base.tensors)
        for donor in self.donors:
            if set(donor.tensors) != names:
                diff = sorted(set(donor.tensors) ^ names)
                raise ValueError(f"tensor names of {donor.directory} and the anchor differ: {diff[:5]}")
        self.names = self.base.names()
        self.widened = [self._widened(s, c) for s, c in zip((spec.base, *spec.donors), (self.base, *self.donors))]
        for name in self.names:
            infos = [c.tensors[name] for c in (self.base, *self.donors)]
            dtypes = {torch.float32 if name in w else i.dtype for i, w in zip(infos, self.widened, strict=True)}
            if len({i.shape for i in infos}) != 1 or len(dtypes) != 1:
                raise ValueError(f"incompatible tensor {name}: shape or dtype differs between the sources")
        self.coefficients: dict[str, tuple[float, ...]] = {}
        for pattern, coefficients in spec.tensor_coefficients.items():
            matched = [n for n in self.names if re.fullmatch(pattern, n)]
            if not matched:
                raise ValueError(f"tensor-coefficient pattern {pattern!r} matches no tensor")
            for name in matched:
                if name in self.coefficients:
                    raise ValueError(f"tensor {name} matches more than one tensor-coefficient pattern")
                self.coefficients[name] = coefficients

    def _widened(self, source: Source, checkpoint: Checkpoint) -> set[str]:
        """Names this source widens; each must be stored in the declared dtype."""
        names = set()
        for pattern, dtype in source.widen.items():
            for name, info in checkpoint.tensors.items():
                if re.fullmatch(pattern, name):
                    if info.dtype != WIDENABLE[dtype]:
                        raise ValueError(f"{checkpoint.directory}: {name} is {info.dtype}, widening expects {dtype}")
                    names.add(name)
        return names

    def coefficients_for(self, name: str, coefficients: Sequence[float] | None = None) -> tuple[float, ...]:
        return self.coefficients.get(name, tuple(self.spec.coefficients if coefficients is None else coefficients))

    def slices(self, name: str) -> Iterator[tuple[int, int | None]]:
        """Row ranges along the first axis, each about SLICE_VALUES values; (0, None) for a 0-d tensor."""
        info = self.base.tensors[name]
        if not info.shape:
            yield 0, None
            return
        rows = max(1, SLICE_VALUES // max(1, info.row_values))
        for start in range(0, info.shape[0], rows):
            yield start, min(info.shape[0], start + rows)

    def _read(self, i: int, name: str, start: int, stop: int | None) -> torch.Tensor:
        checkpoint = (self.base, *self.donors)[i]
        t = checkpoint.read(name, start, stop)
        return t.to(torch.float32) if name in self.widened[i] else t

    def merged_slice(
        self, name: str, start: int, stop: int | None, coefficients: Sequence[float] | None = None
    ) -> torch.Tensor:
        anchor = self._read(0, name, start, stop)
        donors = [self._read(i + 1, name, start, stop) for i in range(len(self.donors))]
        return merge_values(anchor, donors, self.coefficients_for(name, coefficients), name)

    def merged(self, name: str, coefficients: Sequence[float] | None = None) -> torch.Tensor:
        """One merged tensor in the anchor's dtype, computed slice by slice into a fresh tensor."""
        info = self.base.tensors[name]
        out = torch.empty(info.shape, dtype=info.dtype)
        for start, stop in self.slices(name):
            values = self.merged_slice(name, start, stop, coefficients)
            if stop is None:
                out.copy_(values)
            else:
                out[start:stop] = values
        return out


def build_merge(spec: MergeSpec, log: Callable[[str], None]) -> dict[str, torch.Tensor]:
    """The merged model as a weight table (name -> tensor in the anchor's dtype); nothing is written to disk.

    Memory: the output (the size of the anchor) plus one slice's work buffers: the anchor and donor slices as
    stored, and four float32 buffers (anchor, one update per donor, sum), about 0.8 GB for two donors.
    """
    t0 = time.monotonic()
    plan = MergePlan(spec)
    weights: dict[str, torch.Tensor] = {}
    infos = [plan.base.tensors[n] for n in plan.names]
    done_bytes, total = 0, sum(i.dtype.itemsize * math.prod(i.shape) for i in infos)
    for i, name in enumerate(plan.names, 1):
        weights[name] = plan.merged(name)
        done_bytes += weights[name].numel() * weights[name].element_size()
        if i % 50 == 0 or i == len(plan.names):
            seconds = time.monotonic() - t0
            log(
                f"merged {i}/{len(plan.names)} tensors, {done_bytes / 2**30:.1f} of {total / 2**30:.1f} GiB, "
                f"{seconds:.0f} s ({done_bytes * (1 + len(plan.donors)) / max(seconds, 1e-9) / 1e9:.2f} GB/s read)"
            )
    return weights


def bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's stored bits as integers, so that comparisons tell -0.0 from 0.0 and see every NaN payload."""
    return t.contiguous().view(INTEGER_VIEW[t.element_size()])


def describe_value(t: torch.Tensor, flat: int) -> dict[str, Any]:
    """Value and stored bits (hexadecimal) of element `flat` of a flattened tensor."""
    mask = (1 << (8 * t.element_size())) - 1
    return {"value": float(t.reshape(-1)[flat]), "bits": f"{int(bits(t).reshape(-1)[flat]) & mask:#x}"}


def unravel(flat: int, shape: Sequence[int]) -> list[int]:
    index = []
    for size in reversed(shape):
        flat, i = divmod(flat, size)
        index.append(i)
    return index[::-1]


@dataclass
class TensorDifference:
    name: str
    shard: str
    values: int  # differing stored values; -1 when the dtype or shape differs
    reference_equals_base: bool


def check_against(
    spec: MergeSpec,
    reference_dir: Path,
    log: Callable[[str], None],
    first_only: bool = False,
) -> dict[str, Any]:
    """Recompute the merge of `spec` and compare it bit for bit with the checkpoint in `reference_dir`.

    Shards of the reference are visited in name order and the tensors of a shard in storage order. Returns a
    summary: whether every stored value is equal, the tensors compared, the names only one side has, every
    differing tensor (with its count of differing values and whether the reference equals the anchor there) and
    the first difference (tensor, shard, flat and multi-dimensional index, both values and their bits). With
    `first_only` the comparison stops at the first differing tensor.
    """
    t0 = time.monotonic()
    plan = MergePlan(spec)
    reference = Checkpoint(reference_dir)
    missing = sorted(set(plan.names) - set(reference.tensors))
    unexpected = sorted(set(reference.tensors) - set(plan.names))
    differing: list[TensorDifference] = []
    first: dict[str, Any] | None = None
    compared = values = 0
    by_shard: dict[str, list[str]] = {}
    for name in reference.names():
        by_shard.setdefault(reference.tensors[name].file.name, []).append(name)
    for s, (shard, names) in enumerate(sorted(by_shard.items()), 1):
        for name in names:
            if name not in plan.base.tensors:
                continue
            ref_info, base_info = reference.tensors[name], plan.base.tensors[name]
            compared += 1
            if ref_info.shape != base_info.shape or ref_info.dtype != base_info.dtype:
                differing.append(TensorDifference(name, shard, -1, False))
                if first is None:
                    first = {
                        "tensor": name,
                        "shard": shard,
                        "reason": f"reference {ref_info.dtype} {ref_info.shape}, merge {base_info.dtype} {base_info.shape}",
                    }
                continue
            count = 0
            for start, stop in plan.slices(name):
                ours, theirs = plan.merged_slice(name, start, stop), reference.read(name, start, stop)
                differ = (bits(ours) != bits(theirs)).reshape(-1)
                n = int(differ.sum())
                if n and first is None:
                    flat = int(differ.nonzero()[0])
                    whole = flat + start * ref_info.row_values
                    first = {
                        "tensor": name,
                        "shard": shard,
                        "flat_index": whole,
                        "index": unravel(whole, ref_info.shape),
                        "merge": describe_value(ours, flat),
                        "reference": describe_value(theirs, flat),
                    }
                count += n
                values += ours.numel()
            if count:
                same_as_base = all(
                    torch.equal(bits(plan.base.read(name, a, b)), bits(reference.read(name, a, b)))
                    for a, b in plan.slices(name)
                )
                differing.append(TensorDifference(name, shard, count, same_as_base))
                log(f"DIFFERENT {name} ({shard}): {count} values; reference equals the anchor: {same_as_base}")
                if first_only:
                    break
        if first_only and differing:
            break
        if s % 50 == 0:
            log(f"compared {s}/{len(by_shard)} reference shards, {compared} tensors, {time.monotonic() - t0:.0f} s")
    equal = not (missing or unexpected or differing)
    return {
        "reference": str(reference_dir),
        "spec": spec.to_json(),
        "equal": equal,
        "tensors_compared": compared,
        "values_compared": values,
        "missing_in_reference": missing,
        "only_in_reference": unexpected,
        "differing": [d.__dict__ for d in differing],
        "first_difference": first,
        "router_bias_kept_at_anchor": bool(differing)
        and not (missing or unexpected)
        and all(ROUTER_BIAS.fullmatch(d.name) and d.reference_equals_base for d in differing),
        "seconds": round(time.monotonic() - t0, 1),
    }


def parse_coefficients(text: str) -> tuple[float, ...]:
    return tuple(float(c) for c in text.split(","))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, help="spec file (instead of --base, --donor and their options)")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--donor", type=Path, action="append", default=[])
    parser.add_argument("--coefficients", type=parse_coefficients, help="one per donor, comma-separated")
    parser.add_argument("--widen", action="append", default=[], metavar="PATTERN=DTYPE")
    parser.add_argument("--tensor-coefficients", action="append", default=[], metavar="PATTERN=C1,C2")
    parser.add_argument("--arm", action="append", default=[], metavar="LETTER=GROUP:DIR")
    parser.add_argument("--write-spec", type=Path)
    parser.add_argument("--check-against", type=Path)
    parser.add_argument("--json", type=Path, help="write the check summary here")
    parser.add_argument("--first-only", action="store_true", help="stop the check at the first differing tensor")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if (args.write_spec is None) == (args.check_against is None):
        parser.error("give exactly one of --write-spec and --check-against")
    if args.spec is None and (args.base is None or not args.donor or args.coefficients is None):
        parser.error("give --spec, or --base, --donor and --coefficients")
    if args.spec is not None and (args.base or args.donor or args.widen or args.tensor_coefficients or args.arm):
        parser.error("--spec cannot be combined with --base, --donor, --widen, --tensor-coefficients or --arm")
    return args


def spec_from_args(args: argparse.Namespace) -> MergeSpec:
    if args.spec is not None:
        spec = MergeSpec.load(args.spec)
        return spec if args.coefficients is None else spec.with_coefficients(args.coefficients)
    widen = dict(item.split("=", 1) for item in args.widen)
    tensor_coefficients = {
        pattern: parse_coefficients(values) for pattern, values in (i.rsplit("=", 1) for i in args.tensor_coefficients)
    }
    arms: dict[str, ArmSource | None] = {}
    for item in args.arm:
        letter, target = item.split("=", 1)
        if target == "none":
            arms[letter] = None
        else:
            group, directory = target.split(":", 1)
            arms[letter] = ArmSource(group, Path(directory))
    return MergeSpec(
        Source(args.base), tuple(Source(d, widen) for d in args.donor), args.coefficients, tensor_coefficients, arms
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = spec_from_args(args)
        if args.write_spec is not None:
            MergePlan(spec)  # the sources exist and are aligned
            write_json_atomic(args.write_spec, spec.to_json())
            print(f"wrote {args.write_spec}")
            return 0
        if args.threads:
            torch.set_num_threads(args.threads)
        coefficients = ", ".join(f"{c:g}" for c in spec.coefficients)
        print(f"checking the merge with coefficients {coefficients} against {args.check_against}", flush=True)
        summary = check_against(spec, args.check_against, lambda m: print(m, flush=True), args.first_only)
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json is not None:
        write_json_atomic(args.json, summary)
    if summary["missing_in_reference"] or summary["only_in_reference"]:
        print(f"names only in the merge: {summary['missing_in_reference'][:5]}; only in the reference: "
              f"{summary['only_in_reference'][:5]}")
    if summary["first_difference"] is not None:
        print(f"first difference: {json.dumps(summary['first_difference'])}")
    print(
        f"{summary['tensors_compared']} tensors, {summary['values_compared']:,} values compared in "
        f"{summary['seconds']} s: {'EQUAL in every stored value' if summary['equal'] else 'DIFFERENT'}"
        f"{' (only router biases that the reference keeps at the anchor)' if summary['router_bias_kept_at_anchor'] else ''}"
    )
    if summary["equal"]:
        return 0
    return 3 if summary["router_bias_kept_at_anchor"] else 1


if __name__ == "__main__":
    sys.exit(main())
