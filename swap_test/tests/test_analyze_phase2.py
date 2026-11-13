"""Phase-2 decisions on made-up replies and grades: necessity, eligibility, the paired capability comparison, V3,
the stopping criterion, the bootstrap and the quick look."""

import argparse
import json

import numpy as np
import pytest

from swaptest import analyze_phase2 as a2
from swaptest import jsonfield

STOP = 128009


def gsm8k_rows(n=48):
    return [{"idx": i, "gold": "1", "question": f"q{i}"} for i in range(n)]


def reply(arm, pid, capped, text="The answer is 1."):
    return {
        "arm": arm,
        "prompt_id": pid,
        "stop_reason": "cap" if capped else STOP,
        "n_new": 4096 if capped else 300,
        "token_ids": [pid, ord(arm), int(capped)],
        "text": text,
    }


def runaway_replies(capped_counts, n=48):
    return {(arm, i): reply(arm, i, i < count) for arm, count in capped_counts.items() for i in range(n)}


def test_runaway_necessary_gate_inconclusive_and_candidate():
    summary = a2.analyze_runaway(gsm8k_rows(), runaway_replies({"A": 1, "P": 40, "N": 1, "G": 30}), None)
    p2 = summary["phase2"]
    assert summary["v1"]["passed"] and p2["v2"]["outcome"] == "pass"
    assert summary["comparisons"]["P"]["outcome"] == "effect"
    assert p2["necessity"] == {"N": "necessary", "G": "inconclusive"}
    assert p2["eligible"] == {"P": False, "N": True, "G": False}
    assert p2["candidate"] == "N" and p2["task_arms"] == ["N"]
    decisions = a2.runaway_decisions(summary)
    assert decisions["necessity_N"] == "necessary" and decisions["task_arms"] == "N"
    assert decisions["vs_A_N"] == "no_effect"


def test_runaway_not_necessary_and_fallback_candidate():
    summary = a2.analyze_runaway(gsm8k_rows(), runaway_replies({"A": 1, "P": 47, "N": 47, "G": 1}), None)
    p2 = summary["phase2"]
    assert p2["necessity"] == {"N": "not_necessary", "G": "necessary"}
    assert p2["candidate"] == "G" and p2["task_arms"] == ["G"]


def test_runaway_without_reproduction_scores_p_too():
    summary = a2.analyze_runaway(gsm8k_rows(), runaway_replies({"A": 1, "P": 5, "N": 1, "G": 2}), None)
    p2 = summary["phase2"]
    assert p2["v2"]["outcome"] == "fail" and set(p2["necessity"].values()) == {"not_tested"}
    assert p2["task_arms"] == ["N"]  # P, 5 of 48 against 1 of 48, is inconclusive against A: not eligible
    quiet = a2.analyze_runaway(gsm8k_rows(), runaway_replies({"A": 1, "P": 1, "N": 1, "G": 2}), None)
    assert quiet["phase2"]["task_arms"] == ["N", "P"]
    partial = a2.analyze_runaway(gsm8k_rows(), runaway_replies({"A": 1, "P": 18, "N": 1, "G": 1}), None)
    assert partial["phase2"]["v2"]["outcome"] == "partial"


def test_runaway_with_failed_v1_makes_no_arm_eligible():
    rows = runaway_replies({"A": 1, "P": 40, "N": 1, "G": 1})
    for i in range(48):
        rows[("A", i)]["text"] = "no number"
    summary = a2.analyze_runaway(gsm8k_rows(), rows, None)
    assert not summary["v1"]["passed"] and summary["phase2"]["task_arms"] == []
    assert a2.runaway_decisions(summary)["task_arms"] == "none"


def test_mcnemar_exact():
    assert a2.mcnemar_p(0, 0) == 1.0
    assert a2.mcnemar_p(5, 0) == pytest.approx(2 / 32)
    assert a2.mcnemar_p(3, 3) == 1.0
    assert a2.mcnemar_p(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 2**12)


def test_bootstrap_matches_the_standard_error_and_is_seeded():
    rng = np.random.default_rng(0)
    items = [("mmlu_pro", f"g{i}", int(d)) for i, d in enumerate(rng.choice([-1, 0, 0, 1], size=400))]
    draws = a2.bootstrap_task_differences(items, 4000, seed=1)[:, a2.TASKS.index("mmlu_pro")]
    diffs = np.array([d for _, _, d in items], dtype=float)
    assert draws.std() == pytest.approx(diffs.std() / np.sqrt(len(diffs)), rel=0.08)
    again = a2.bootstrap_task_differences(items, 4000, seed=1)[:, a2.TASKS.index("mmlu_pro")]
    assert np.array_equal(draws, again)


def test_bootstrap_resamples_both_cruxeval_directions_of_a_problem_together():
    items = [(t, f"p{i}", 1 if i % 2 else -1) for i in range(50) for t in ("cruxeval_input", "cruxeval_output")]
    draws = a2.bootstrap_task_differences(items, 200, seed=2)
    i, o = a2.TASKS.index("cruxeval_input"), a2.TASKS.index("cruxeval_output")
    assert np.array_equal(draws[:, i], draws[:, o])  # identical differences, same resampled problems


def capability_setup(arm_correct, a_correct=None, capped_x=0, n=100):
    """Task prompts (n per target) and grades of arm A and arm X; *_correct(task, i) -> bool."""
    prompts, grades = {}, {}
    a_correct = a_correct or (lambda task, i: i % 10 < 8)
    for t, task in enumerate(a2.TASKS):
        for i in range(n):
            pid = 100000 * (t + 1) + i
            group = f"cruxeval:{i}" if task.startswith("cruxeval") else f"mmlu:{i}"
            prompts[pid] = {"idx": pid, "task": task, "group": group, "pool": "all"}
            for arm, correct, capped in (("A", a_correct(task, i), False), ("X", arm_correct(task, i), i < capped_x)):
                grades[(arm, pid)] = {
                    "arm": arm,
                    "prompt_id": pid,
                    "task": task,
                    "correct": bool(correct),
                    "answer": "x",
                    "capped": capped,
                    "n_new": 900,
                }
    return prompts, grades


RUNAWAY = {"phase2": {"eligible": {"X": True}}}


@pytest.mark.parametrize(
    ("arm_correct", "a_correct", "capped_x", "expected"),
    [
        (lambda t, i: True, None, 0, "better"),
        (lambda t, i: i % 10 < 8, None, 0, "not_better"),  # identical to A
        (lambda t, i: False, None, 0, "worse"),
        # +0.3 on both CRUXEval directions, -0.1 on MMLU-Pro: D = +0.10 overall, MMLU-Pro wholly below 0
        (
            lambda t, i: t.startswith("cruxeval") or i % 10 < 9,
            lambda t, i: t == "mmlu_pro" or i % 10 < 7,
            0,
            "trade_off",
        ),
        (lambda t, i: True, lambda t, i: i % 10 < 2, 0, "not_interpreted_v3"),  # A far below #9984
        (lambda t, i: True, None, 40, "not_interpreted_stopping"),  # X capped far more often than A
    ],
)
def test_capability_decisions(arm_correct, a_correct, capped_x, expected):
    prompts, grades = capability_setup(arm_correct, a_correct, capped_x)
    summary = a2.analyze_capability(prompts, grades, RUNAWAY, 0.95, 2000, 9984)
    c = summary["comparisons"]["X"]
    assert c["decision"] == expected, (c["composite"], c["targets"])
    assert set(c["weights"]) == set(a2.TASKS) and sum(c["weights"].values()) == pytest.approx(1.0)
    decisions = a2.capability_decisions(summary)
    assert decisions["decision_X"] == expected


def test_capability_composite_weights_each_domain_equally():
    prompts, grades = capability_setup(lambda t, i: t == "mmlu_pro" or i % 10 < 8)  # +0.2 on MMLU-Pro only
    c = a2.analyze_capability(prompts, grades, RUNAWAY, 0.95, 500, 1)["comparisons"]["X"]
    assert c["targets"]["mmlu_pro"]["diff"] == pytest.approx(0.2)
    assert c["composite"]["diff"] == pytest.approx(0.5 * 0.2)


def test_capability_ineligible_arm_is_not_interpreted():
    prompts, grades = capability_setup(lambda t, i: True)
    summary = a2.analyze_capability(prompts, grades, {"phase2": {"eligible": {"X": False}}}, 0.95, 500, 1)
    assert summary["comparisons"]["X"]["decision"] == "not_interpreted_stopping"


def test_v3_uses_the_subset_reference_for_a_listed_pool():
    scores = {t: {"accuracy": 0.53} for t in a2.TASKS} | {"all": {"n": 10, "capped": 0}}
    scores["cruxeval_input"]["accuracy"] = 0.7
    scores["cruxeval_output"]["accuracy"] = 0.6
    assert a2.v3_check(scores, "all")["passed"] and not a2.v3_check(scores, "listed")["passed"]


def test_grades_are_reused_for_unchanged_replies(tmp_path, monkeypatch):
    calls = []

    def fake_grade(prompt, rep, timeout):
        calls.append(rep["prompt_id"])
        return {
            "arm": rep["arm"],
            "prompt_id": rep["prompt_id"],
            "task": prompt["task"],
            "text_sha256": a2.text_sha(rep),
            "capped": False,
            "n_new": 1,
            "correct": True,
            "answer": "A",
        }

    monkeypatch.setattr(a2, "grade_reply", fake_grade)
    prompts = {7: {"task": "mmlu_pro"}, 8: {"task": "mmlu_pro"}}
    replies = {("A", 7): reply("A", 7, False), ("A", 8): reply("A", 8, False)}
    path = tmp_path / "grades.jsonl"
    a2.grade_all(prompts, replies, path, 2, 1.0)
    a2.grade_all(prompts, replies, path, 2, 1.0)
    assert sorted(calls) == [7, 8]
    replies[("A", 8)]["text"] = "changed"
    a2.grade_all(prompts, replies, path, 2, 1.0)
    assert sorted(calls) == [7, 8, 8]


def test_quicklook_limits():
    prompts = {pid: {"task": task, "gold": "A"} for pid, task in enumerate(a2.TASKS)}
    replies = {("A", pid): reply("A", pid, pid == 0) for pid in prompts}
    grades = {
        ("A", pid): {"task": p["task"], "correct": True, "capped": pid == 0, "n_new": 100, "answer": "A"}
        for pid, p in prompts.items()
    }
    limits = argparse.Namespace(max_capped=1, max_median=2000, min_correct=1)
    assert a2.analyze_quicklook(prompts, replies, grades, limits)["problems"] == []
    limits = argparse.Namespace(max_capped=0, max_median=50, min_correct=2)
    problems = a2.analyze_quicklook(prompts, replies, grades, limits)["problems"]
    assert len(problems) == 2 + len(a2.TASKS)


def test_jsonfield_prints_strings_and_json(tmp_path, capsys):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"v1": {"passed": True}, "arms": ["P", "N"], "name": "plain"}))
    assert jsonfield.main([str(path), "v1.passed"]) == 0
    assert jsonfield.main([str(path), "arms.1"]) == 0
    assert jsonfield.main([str(path), "name"]) == 0
    assert capsys.readouterr().out.split() == ["true", "N", "plain"]
    assert jsonfield.main([str(path), "v1.missing"]) == 1
