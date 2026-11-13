"""Analysis on synthetic replies with known answers: the three outcomes, the validity gate, interim looks."""

import json

import pytest

from swaptest import analyze as analyze_module
from swaptest.analyze import first_divergence, repetition

N = 48
STOP = 128009


def write_case(tmp_path, capped, correct_a=46, missing=()):
    """48 prompts; arm X has capped[X] replies at the cap (the first ones), the rest stop."""
    prompts = [{"idx": i, "gold": str(i), "prompt_ids": [1, 2]} for i in range(N)]
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    lines = []
    for arm, n_cap in capped.items():
        for i in range(N):
            if (arm, i) in missing:
                continue
            at_cap = i < n_cap
            ids = [7, 8, 9] * 1400 if at_cap else [arm_offset(arm) + 3, 4, 5, i, STOP]
            ok = arm != "A" or i < correct_a
            text = f"... the answer is {i if ok else i + 1}"
            lines.append(
                {
                    "arm": arm,
                    "prompt_id": i,
                    "token_ids": ids[:4096],
                    "n_new": min(len(ids), 4096),
                    "stop_reason": "cap" if at_cap else STOP,
                    "text": text + ("<|end_think|>" if i % 2 else ""),
                }
            )
    (tmp_path / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return [
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        "--replies",
        str(tmp_path / "replies.jsonl"),
        "--json",
        str(tmp_path / "summary.json"),
    ]


def arm_offset(arm):
    return 0 if arm in "AC" else 100  # B and D leave A's tokens at position 0; C never does


def summary(tmp_path):
    return json.loads((tmp_path / "summary.json").read_text())


def test_three_outcomes_and_secondary_statistics(tmp_path):
    args = write_case(tmp_path, {"A": 1, "B": 30, "C": 1, "D": 8})
    assert analyze_module.main(args) == 0
    s = summary(tmp_path)
    assert s["v1"]["passed"] and s["arms"]["A"]["rate"] == pytest.approx(1 / 48)
    assert s["arms"]["A"]["accuracy"] == pytest.approx(46 / 48)
    assert {a: c["outcome"] for a, c in s["comparisons"].items()} == {
        "B": "effect",
        "C": "no effect",
        "D": "inconclusive",
    }
    b = s["arms"]["B"]
    assert b["capped"] == 30 and b["stop_reasons"] == {"cap": 30, str(STOP): 18}
    assert b["length_quartiles"][2] == ">=4096" and s["arms"]["A"]["length_quartiles"][1] == "5"
    assert b["end_think"] == 24
    assert b["distinct4_median_capped"] == pytest.approx(3 / 1021)
    assert s["arms"]["C"]["identical_to_reference"] == 48 and s["arms"]["B"]["first_divergence_median"] == 0


def test_validity_gate_blocks_interpretation(tmp_path):
    args = write_case(tmp_path, {"A": 10, "B": 40}, correct_a=40)
    assert analyze_module.main(args) == 0
    s = summary(tmp_path)
    assert not s["v1"]["passed"]
    assert s["comparisons"]["B"]["outcome"] == "not interpreted (V1 failed)"


def test_accuracy_below_the_gate_also_fails_v1(tmp_path):
    analyze_module.main(write_case(tmp_path, {"A": 0, "B": 0}, correct_a=38))
    assert summary(tmp_path)["v1"] == {"rate": 0.0, "accuracy": pytest.approx(38 / 48), "passed": False}


def test_missing_pairs_need_interim_mode(tmp_path):
    missing = {("B", i) for i in range(40, 48)} | {("A", 47)}
    args = write_case(tmp_path, {"A": 0, "B": 0}, missing=missing)
    assert analyze_module.main(args) == 2
    assert analyze_module.main([*args, "--interim"]) == 0
    s = summary(tmp_path)
    assert s["missing"] == {"A": 1, "B": 8}
    # 0 of 40 capped in B so far; worst case 8 of 48 vs 0 of 48 is not "no effect" yet
    assert s["comparisons"]["B"]["outcome"] == "no effect" and s["comparisons"]["B"]["early_no_effect"] is False


def test_early_no_effect_call_when_it_cannot_change(tmp_path):
    args = write_case(tmp_path, {"A": 0, "B": 0}, missing={("B", 47)})
    analyze_module.main([*args, "--interim"])
    assert summary(tmp_path)["comparisons"]["B"]["early_no_effect"] is True


def test_min_stopped_gate(tmp_path):
    args = write_case(tmp_path, {"A": 3})
    assert analyze_module.main([*args, "--limit", "8", "--min-stopped", "6"]) == 3
    assert analyze_module.main([*args, "--limit", "8", "--min-stopped", "5"]) == 0


def test_extension_for_the_reference_and_one_arm(tmp_path):
    args = write_case(tmp_path, {"A": 1, "B": 30, "D": 8})
    ext_prompts = [{"idx": N + i, "gold": "0", "prompt_ids": [1, 2]} for i in range(N)]
    ext_replies = [
        {"arm": arm, "prompt_id": N + i, "token_ids": [5, STOP], "n_new": 2, "stop_reason": STOP, "text": "answer is 0"}
        for arm in "AD"
        for i in range(N)
    ]
    (tmp_path / "prompts_ext.jsonl").write_text("".join(json.dumps(p) + "\n" for p in ext_prompts))
    (tmp_path / "replies_ext.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ext_replies))
    both = [
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        str(tmp_path / "prompts_ext.jsonl"),
        "--replies",
        str(tmp_path / "replies.jsonl"),
        str(tmp_path / "replies_ext.jsonl"),
        "--json",
        args[-1],
    ]
    assert analyze_module.main(both) == 2  # arm B was not extended
    assert analyze_module.main([*both, "--arms", "AD"]) == 0
    s = summary(tmp_path)
    assert set(s["arms"]) == {"A", "D"} and s["arms"]["D"]["n"] == 96 and s["arms"]["D"]["capped"] == 8
    # inconclusive on 48 prompts (8 vs 1 capped); 8 of 96 vs 1 of 96 narrows the interval below 0.15
    assert s["comparisons"]["D"]["outcome"] == "no effect" and s["comparisons"]["D"]["newcombe95"][1] < 0.15


def test_replies_are_printed_for_reading(tmp_path, capsys):
    args = write_case(tmp_path, {"A": 1}, correct_a=1)
    assert analyze_module.main([*args, "--limit", "2", "--show-replies", "A"]) == 0
    out = capsys.readouterr().out
    assert "=== prompt 0, arm A: stop cap, 4096 tokens, answer '0', reference '0', correct" in out
    assert f"=== prompt 1, arm A: stop {STOP}, 5 tokens, answer '2', reference '1', wrong" in out


def test_teacher_forced_summary(tmp_path):
    args = write_case(tmp_path, {"A": 0, "B": 0})
    tf = [
        {
            "prompt_id": i,
            "arm": a,
            "nll_mean": 1.0 + (a == "B"),
            "stop_logprob": -0.5 - 2 * (a == "B"),
            "any_stop_logprob": -0.4,
        }
        for i in range(3)
        for a in "AB"
    ]
    (tmp_path / "tf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tf))
    analyze_module.main([*args, "--teacher-forced", str(tmp_path / "tf.jsonl")])
    t = summary(tmp_path)["teacher_forced"]
    assert t["B"]["nll_mean"] == 2.0 and t["B"]["stop_logprob_minus_reference_mean"] == pytest.approx(-2.0)


def test_repetition_and_divergence_helpers():
    distinct, ratio = repetition([1, 2, 3, 4] * 300)
    assert distinct == pytest.approx(4 / 1021) and ratio < 0.05
    assert repetition(list(range(2000)))[0] == 1.0
    assert first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([1, 2], [1, 2]) is None
