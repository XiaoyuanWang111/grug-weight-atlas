"""task_arithmetic, ties and the safetensors writer of gate_patch.py on arrays small enough to check by hand."""

import json
import struct

import numpy as np

import gate_patch
import st_index


def f32(values):
    return np.array(values, dtype=np.float32)


def decoded(bits):
    return [float(v) for v in st_index.bf16_to_float32(bits)]


BASE = f32([1.0, -2.0, 0.5, 4.0, 2.0, -1.0])
D_E = f32([0.25, -0.5, 0.0625, 0.125, 0.75, 0.03125])
D_E2 = f32([-0.125, 0.25, 0.5, -0.0625, 0.25, 0.03125])
E = BASE + D_E
E2 = BASE + D_E2


def test_task_arithmetic_half_and_full_strength():
    # base + 0.5 dE + 0.5 dE2, and base + dE + dE2; every result is a bfloat16 value, so nothing rounds
    assert decoded(gate_patch.task_arithmetic(BASE, [E, E2], [0.5, 0.5])) == [
        1.0625,
        -2.125,
        0.78125,
        4.03125,
        2.5,
        -0.96875,
    ]
    assert decoded(gate_patch.task_arithmetic(BASE, [E, E2], [1.0, 1.0])) == [1.125, -2.25, 1.0625, 4.0625, 3.0, -0.9375]


def test_task_arithmetic_rounds_half_a_step_to_even():
    # half of a one-step update (2^-7 at 1.0) is half a bfloat16 step: the tie goes to the even value, 1.0
    base = f32([1.0, 1.0078125])
    donor = f32([1.0078125, 1.015625])
    assert decoded(gate_patch.task_arithmetic(base, [donor], [0.5])) == [1.0, 1.015625]


def test_ties_cut_sign_election_and_mean():
    # density 0.5 keeps 3 of 6 values per update: dE keeps positions 0, 1, 4 (|.| >= 0.25), dE2 keeps 1, 2, 4
    # position 0: only dE survives the cut           -> base + 0.25
    # position 1: dE -0.5 against dE2 +0.25          -> the elected sign is negative, dE2 is dropped -> base - 0.5
    # position 2: only dE2 survives                  -> base + 0.5
    # position 3: neither survives                   -> base
    # position 4: both survive with the same sign    -> base + mean(0.75, 0.25) = base + 0.5
    # position 5: neither survives (|0.03125| < cut) -> base
    got = decoded(gate_patch.ties(BASE, [E, E2], [1.0, 1.0], 0.5))
    assert got == [1.25, -2.5, 1.0, 4.0, 2.5, -1.0]


def test_ties_divides_by_coefficient_mass():
    # with coefficients [1.0, 0.5], position 2 (only dE2 survives) holds dE2 in full: 0.5 * dE2 / 0.5;
    # position 4 holds 2 + (0.75 + 0.5 * 0.25) / 1.5 = 2.5833..., which rounds down to 2 + 37 / 64 (step 1 / 64)
    got = decoded(gate_patch.ties(BASE, [E, E2], [1.0, 0.5], 0.5))
    assert got == [1.25, -2.5, 1.0, 4.0, 2.578125, -1.0]


def test_safetensors_writer_layout(tmp_path):
    tensors = {
        "model.layers.1.b.weight": st_index.float32_to_bf16(f32([1.0, -2.0, 0.5])),
        "model.layers.0.a.weight": st_index.float32_to_bf16(f32([[0.25, 3.0], [-1.0, 4.0]]).ravel()),
    }
    shapes = {"model.layers.1.b.weight": [3], "model.layers.0.a.weight": [2, 2]}
    metadata = {"format": "pt", "description": "test"}
    path = tmp_path / "patch.safetensors"
    gate_patch.write_bf16_safetensors(path, tensors, shapes, metadata)

    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert n % 8 == 0
    head = raw[8 : 8 + n]
    header = json.loads(head)
    # compact JSON, then spaces up to a multiple of 8 bytes
    assert head.rstrip(b" ") == json.dumps(header, separators=(",", ":")).encode()
    assert list(header) == ["model.layers.0.a.weight", "model.layers.1.b.weight", "__metadata__"]
    assert header["__metadata__"] == metadata
    data = raw[8 + n :]
    offset = 0
    for name in ("model.layers.0.a.weight", "model.layers.1.b.weight"):
        entry = header[name]
        assert entry["dtype"] == "BF16"
        assert entry["shape"] == shapes[name]
        start, end = entry["data_offsets"]
        assert start == offset
        assert data[start:end] == tensors[name].tobytes()
        offset = end
    assert offset == len(data)
