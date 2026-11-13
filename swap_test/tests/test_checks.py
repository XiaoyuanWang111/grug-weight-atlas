"""Self-check helpers: mixed-versus-alone, long text, timing, the speed model fit and projection."""

import pytest
from helpers import FakeTokenizer, perturbed, random_weights, tiny_config

from swaptest.checks import SpeedModel, decode_rows, long_text_ids, mixed_vs_alone, time_decode
from swaptest.config import ModelConfig
from swaptest.model import GrugModel, Overrides
from swaptest.patch import ARM_GROUPS, GROUP_PATTERNS, arm_index
from swaptest.sampling import Sampler

CFG = ModelConfig.from_dict(tiny_config())
WEIGHTS = random_weights(CFG, seed=9)
PATCH = perturbed(WEIGHTS, CFG, scale=1.0)
TABLES = {
    arm_index(a): {n: t for n, t in PATCH.items() if any(GROUP_PATTERNS[g].fullmatch(n) for g in ARM_GROUPS[a])}
    for a in "ABCD"
}
MODEL = GrugModel(CFG, WEIGHTS, Overrides.from_arm_tables(TABLES))


def test_mixed_batch_matches_arm_a_alone():
    result = mixed_vs_alone(MODEL, [256, 1, 2, 3, 4, 5], 12, "ABCD")
    assert result["top1_agreement"] == 1.0 and result["max_abs_diff"] < 1e-5 and not result["nan"]
    assert min(result["other_arms_max_abs_diff_from_A"].values()) > 1e-3


def test_long_text_starts_with_one_start_token():
    prompts = [{"question": "What is 2 + 2?", "answer": "4\n#### 4"}] * 20
    ids = long_text_ids(FakeTokenizer(), prompts, 300)
    assert len(ids) == 300 and ids[0] == 256 and ids.count(256) == 1
    with pytest.raises(ValueError):
        long_text_ids(FakeTokenizer(), prompts[:1], 300)


def test_time_decode_reports_positive_rates():
    # Rows are laid out as in the main run: prompt by prompt, the arms of one prompt next to each other. Six rows
    # over arms ABCD are prompt 0 with A, B, C, D and prompt 1 with A, B: 4 rows of 3 tokens and 2 rows of 5.
    t = time_decode(MODEL, [[256, 1, 2], [256, 3, 4, 5, 6]], "ABCD", batch=6, steps=3, chunk=4, row_block=4)
    assert t["batch"] == 6 and t["step_s"] > 0 and t["prefill_tokens"] == 4 * 3 + 2 * 5
    sampled = time_decode(MODEL, [[256, 1, 2]], "AB", batch=2, steps=2, chunk=4, sampler=Sampler(1, 1.0))
    assert sampled["step_s"] > 0 and sampled["prefill_tokens"] == 2 * 3


def test_time_decode_rows_follow_the_run_layout():
    assert decode_rows([[1], [2], [3]], "ABCD", 6) == [
        ([1], "A"),
        ([1], "B"),
        ([1], "C"),
        ([1], "D"),
        ([2], "A"),
        ([2], "B"),
    ]
    assert [p for p, _ in decode_rows([[1], [2]], "AB", 6)] == [[1], [1], [2], [2], [1], [1]]  # prompts cycle


def test_speed_model_recovers_bandwidth_and_call_overhead():
    speed = SpeedModel(MODEL)
    true_bw, true_call = 50e9, 2e-4
    measurements = []
    for b in (1, 16, 64, 192):
        calls = speed.distinct_experts(b) * CFG.num_layers
        measurements.append((b, (calls * speed.expert_bytes + speed.dense_bytes) / true_bw + calls * true_call))
    speed.fit(measurements)
    assert speed.bandwidth == pytest.approx(true_bw, rel=1e-6) and speed.call_s == pytest.approx(true_call, rel=1e-6)
    runaway = speed.project_hours([0.03, 0.8, 0.03, 0.8], 48, 4096, 130, 1000.0)
    normal = speed.project_hours([0.03] * 4, 48, 4096, 130, 1000.0)
    assert runaway > normal > 0
    assert speed.project_hours([0.03, 0.8, 0.03, 0.8], 48, 4096, 130, 1000.0, batch_size=192) == runaway
    assert speed.project_hours([0.03, 0.8, 0.03, 0.8], 48, 4096, 130, 1000.0, batch_size=96) > runaway
    assert speed.expert_bytes == 3 * 16 * 32 * 4  # gate, up, down of one expert in float32


def test_kv_bytes_counts_windows_for_short_layers():
    speed = SpeedModel(MODEL)
    per_slot = 2 * CFG.num_kv_heads * CFG.head_dim * 4
    assert speed.kv_bytes(10) == (3 * CFG.sliding_window + 2 * 10) * per_slot
    assert speed.kv_bytes(2) == 5 * 2 * per_slot
