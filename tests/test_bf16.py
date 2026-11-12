"""bfloat16 encode and decode of st_index against a slow reference written here."""

import struct
from fractions import Fraction

import numpy as np

import st_index


def float32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_value(bits32):
    return struct.unpack("<f", struct.pack("<I", bits32))[0]


def reference_bf16(x):
    """Nearest bfloat16 to the float32 value x by exact comparison of the two candidates; a tie goes to the
    candidate whose last mantissa bit is 0. Finite inputs only."""
    bits = float32_bits(x)
    low = bits & 0xFFFF0000  # same sign, magnitude truncated to bfloat16
    high = low + 0x10000  # next bfloat16 away from zero (may step into the next binade)
    exact = Fraction(bits_value(bits))
    d_low = abs(exact - Fraction(bits_value(low)))
    d_high = abs(exact - Fraction(bits_value(high))) if (high >> 23) & 0xFF != 0xFF else None
    if d_high is None or d_low < d_high:
        chosen = low
    elif d_high < d_low:
        chosen = high
    else:
        chosen = low if (low >> 16) & 1 == 0 else high
    return chosen >> 16


def encode(values):
    return st_index.float32_to_bf16(np.array(values, dtype=np.float32))


def test_named_cases():
    cases = {
        "positive, rounds down": 1.0 + 2.0**-9,
        "positive, rounds up": 1.0 + 3 * 2.0**-9,
        "negative": -(3.0 + 2.0**-8 + 2.0**-12),
        "zero": 0.0,
        "negative zero": -0.0,
        "tie, stays on the even value": 1.0 + 2.0**-8,
        "tie, moves up to the even value": 1.0 + 3 * 2.0**-8,
        "rounds up into the next binade": 2.0 - 2.0**-9,
        "tie at the top of a binade": 2.0 - 2.0**-8,
        "negative, into the next binade": -(4.0 - 2.0**-7),
    }
    got = encode(list(cases.values()))
    for (name, x), bits in zip(cases.items(), got, strict=True):
        assert int(bits) == reference_bf16(x), name
    # spelled out, so that the reference itself is checked as well
    assert int(encode([1.0 + 2.0**-8])[0]) == 0x3F80  # 1.0
    assert int(encode([1.0 + 3 * 2.0**-8])[0]) == 0x3F82  # 1 + 2^-6
    assert int(encode([2.0 - 2.0**-9])[0]) == 0x4000  # 2.0
    assert int(encode([-0.0])[0]) == 0x8000


def test_random_values_against_reference():
    rng = np.random.default_rng(7)
    patterns = rng.integers(0, 2**32, 3000, dtype=np.uint64).astype(np.uint32)
    exponent = (patterns >> 23) & 0xFF
    patterns = patterns[(exponent > 0) & (exponent < 0xFE)]  # finite, normal, away from overflow
    values = patterns.view(np.float32)
    got = st_index.float32_to_bf16(values)
    expected = [reference_bf16(float(v)) for v in values]
    assert [int(b) for b in got] == expected


def test_decode_matches_struct():
    rng = np.random.default_rng(8)
    bits = rng.integers(0, 2**16, 2000).astype(np.uint16)
    bits = bits[((bits >> 7) & 0xFF) != 0xFF]  # leave out inf and NaN
    decoded = st_index.bf16_to_float32(bits)
    assert decoded.dtype == np.float32
    assert [float(v) for v in decoded] == [bits_value(int(b) << 16) for b in bits]


def test_decode_encode_round_trip_all_patterns():
    bits = np.arange(2**16, dtype=np.uint32).astype(np.uint16)
    assert np.array_equal(st_index.float32_to_bf16(st_index.bf16_to_float32(bits)), bits)


def test_to_float32_decodes_bf16_and_f32():
    values = np.array([1.5, -0.25, 3.0], dtype=np.float32)
    bf16_raw = st_index.float32_to_bf16(values).tobytes()
    assert np.array_equal(st_index.to_float32(bf16_raw, "BF16"), values)
    assert np.array_equal(st_index.to_float32(values.tobytes(), "F32"), values)
