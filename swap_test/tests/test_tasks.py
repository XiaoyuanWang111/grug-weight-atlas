"""Capability tasks on made-up rows: the final-content rule, CRUXEval answer reading and grading, MMLU-Pro prompts
and extraction, and the task prompt file."""

import json

import pytest
from helpers import CRUX_ROWS, FakeTokenizer, mmlu_rows, write_task_data

from swaptest import cruxeval, mmlu_pro, taskprompts
from swaptest.answers import answer_text
from swaptest.records import read_jsonl

ROW = CRUX_ROWS[0]  # f(x) = x + 1, f(16) == 17


@pytest.mark.parametrize(
    ("text", "stopped", "expected"),
    [
        ("plain answer", True, "plain answer"),
        ("<|start_think|>draft<|end_think|>\nfinal", True, "\nfinal"),
        ("<|start_think|>draft<|end_think|>\nfinal", False, "\nfinal"),  # cut after the reasoning closed
        ("<|start_think|>a<|end_think|>b<|end_think|>c", True, "c"),  # the last end marker counts
        ("<|start_think|>only reasoning", True, "only reasoning"),  # stopped without content: the reasoning
        ("<|start_think|>only reasoning", False, ""),  # cut off inside the reasoning: no answer
        ("<|start_think|>x<|end_think|>", True, "x"),  # empty final content, stopped
        ("<|start_think|>x<|end_think|>", False, ""),
    ],
)
def test_answer_text_follows_the_final_content_rule(text, stopped, expected):
    assert answer_text(text, stopped) == expected


def test_cruxeval_prompts_fill_the_canonical_templates():
    out = cruxeval.build_prompt(ROW, "output")
    assert "[PYTHON]\ndef f(x):\n    return x + 1\nassert f(16) == ??\n[/PYTHON]\n[THOUGHT]\n" in out
    inp = cruxeval.build_prompt(ROW, "input")
    assert inp.endswith("assert f(??) == 17\n[/PYTHON]\n[THOUGHT]\n") and inp.startswith("You will be given a function")
    tricky = {"id": "sample_9", "code": "def f(x):\n    return '{output}'", "input": "1", "output": "'{output}'"}
    assert "return '{output}'" in cruxeval.build_prompt(tricky, "input")  # code is filled last, as in Marin's task


@pytest.mark.parametrize(
    ("text", "direction", "expected"),
    [
        ("[ANSWER]\nassert f(16) == 17\n[/ANSWER]", "output", "17"),
        ("[ANSWER]\nassert f(16) == 17\n[/ANSWER]", "input", "f(16)"),
        ("[ANSWER]f(16)[/ANSWER]", "input", "f(16)"),  # no ==: the whole block
        ("[ANSWER]17[/ANSWER]", "output", "17"),
        ("[ANSWER]16[/ANSWER]", "input", None),  # no call of f: rejected
        ("[ANSWER]assert f(16) == f(16)[/ANSWER]", "output", None),  # restates the call: rejected
        ("assert f(16) == 17", "output", "17"),  # no [ANSWER]: the whole text
        ("[ANSWER]assert f(1) == 2[/ANSWER] later [ANSWER]assert f(16) == 17[/ANSWER]", "output", "2"),  # first block
        ("[ANSWER]assert f(16) == 17", "output", "17"),  # unclosed block (reply cut off)
        ("[ANSWER]assert self.f(3) == 4[/ANSWER]", "input", None),  # .f( is an attribute, not a call of f
        ("[ANSWER]assert elf(3) == 4; f(2)[/ANSWER]", "input", None),  # elf( continues a name; f(2) is after ==
        ("[ANSWER]assert f(f(15)) == 17[/ANSWER]", "input", "f(f(15))"),
        ("[ANSWER]assert f('a') == 'a=='[/ANSWER]", "output", "'a=='"),  # a second == stays in the answer
        ("[ANSWER]assert f('==') == 3[/ANSWER]", "input", "f('"),  # an == inside the input breaks the call
    ],
)
def test_cruxeval_answer_reading(text, direction, expected):
    assert cruxeval.extract_answer(text, direction, "16") == expected


@pytest.mark.parametrize(
    ("row", "direction", "content", "correct"),
    [
        (CRUX_ROWS[0], "output", "[ANSWER]assert f(16) == 17[/ANSWER]", True),
        (CRUX_ROWS[0], "output", "[ANSWER]assert f(16) == 18[/ANSWER]", False),
        (CRUX_ROWS[0], "input", "[ANSWER]assert f(16) == 17[/ANSWER]", True),
        (CRUX_ROWS[0], "input", "[ANSWER]assert f(15) == 17[/ANSWER]", False),
        (CRUX_ROWS[1], "output", "[ANSWER]assert f('a') == 'a=='[/ANSWER]", True),
        (CRUX_ROWS[2], "input", "[ANSWER]assert f(len('ab'), [1, 3]) == 6[/ANSWER]", True),  # nested calls
        (CRUX_ROWS[4], "output", "[ANSWER]\nassert f(4) == [0, 1, 4, 9]\n[/ANSWER]", True),
        (CRUX_ROWS[5], "output", "[ANSWER]assert f({'b': 2, 'a': 1}) == [('a', 1), ('b', 2)][/ANSWER]", True),
        (CRUX_ROWS[0], "output", "[ANSWER]assert f(16) == 17 or __import__('os')._exit(0)[/ANSWER]", True),
        (CRUX_ROWS[0], "output", "[ANSWER]assert f(16) == __import__('os')._exit(0)[/ANSWER]", False),  # early exit
        (CRUX_ROWS[0], "output", "[ANSWER]assert f(16) == 17 +[/ANSWER]", False),  # syntax error
        (CRUX_ROWS[0], "output", "", False),  # no graded text
    ],
)
def test_cruxeval_grading_by_execution(row, direction, content, correct):
    assert cruxeval.grade(row, direction, content)["correct"] is correct


def test_cruxeval_grading_stops_a_slow_answer():
    result = cruxeval.grade(ROW, "output", "[ANSWER]assert f(16) == __import__('time').sleep(30)[/ANSWER]", timeout=2)
    assert result["correct"] is False


def test_cruxeval_limits_come_from_the_shell_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(cruxeval.os, "name", "posix")
    cmd = cruxeval.command(tmp_path / "check.py")
    assert cmd[:2] == ["/bin/sh", "-c"] and cmd[2] == 'ulimit -v 2097152 && ulimit -t 11 && exec "$0" "$@"'
    assert cmd[4:] == ["-S", "-s", str(tmp_path / "check.py")]


def test_cruxeval_rows_are_checked(tmp_path):
    path = tmp_path / "crux.jsonl"
    path.write_text(json.dumps({"id": "x1", "code": "", "input": "", "output": ""}) + "\n")
    with pytest.raises(ValueError, match="sample_N"):
        cruxeval.read_rows(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Some reasoning. The answer is (C).", "C"),
        ("the answer is c", "C"),
        ("Answer: B", "B"),
        ("first the answer is (A), later the answer is (B)", "A"),  # the first marker wins
        ("so the pick is D.", "D"),  # fallback: the last standalone capital A-J
        ("B looks right. I agree.", "I"),  # quirk of the fallback: a final pronoun I
        ("the answer is a bit unclear; it is B", "A"),  # quirk of the case-insensitive pattern
        ("<|start_think|>the answer is (A)<|end_think|>The answer is (B)", "B"),  # after the reasoning
        ("no letter here", None),
        ("answer: (e)", None),  # "Answer:" must be followed directly by the letter; the fallback needs a capital
    ],
)
def test_mmlu_pro_extraction(text, expected):
    assert mmlu_pro.extract_answer(text) == expected


def test_mmlu_pro_prompt_layout():
    support = mmlu_rows(6, True, first_id=1)
    row = mmlu_rows(3, False, first_id=70)[0]  # category law, options padded with N/A
    text = mmlu_pro.build_prompt(support, row, 2)
    assert text.startswith(
        "The following are multiple choice questions (with answers) about law. Think step by step and then finish "
        'your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n\nQuestion:\n'
    )
    law = [s for s in support if s["category"] == "law"]
    assert text.count("Answer: Let's think step by step.") == 3 and "A: Let's think step by step." not in text
    assert law[0]["question"] in text and law[1]["question"] in text and law[2]["question"] not in text
    assert "N/A" not in text and text.endswith("D. option 3 of question 70\nAnswer: Let's think step by step.")


def test_mmlu_pro_fit_keeps_the_most_examples_under_2048_tokens():
    support = mmlu_rows(10, True, first_id=1)
    for s in support:
        s["cot_content"] = "A: Let's think step by step. " + "x" * 600
    row = mmlu_rows(1, False, first_id=500)[0]
    tokenizer = FakeTokenizer()  # one token per byte
    text, shots = mmlu_pro.fit_prompt(support, row, tokenizer)
    assert len(tokenizer.encode(text, add_special_tokens=False)) < mmlu_pro.FIT_LIMIT
    more = mmlu_pro.build_prompt(support, row, shots + 1)
    assert 0 < shots < 5 and len(tokenizer.encode(more, add_special_tokens=False)) >= mmlu_pro.FIT_LIMIT
    for s in support:
        s["cot_content"] = "A: " + "x" * 3000
    assert mmlu_pro.fit_prompt(support, row, tokenizer)[1] == 0


def test_mmlu_pro_grade_uses_the_gold_letter():
    assert mmlu_pro.grade({"gold": "C"}, "The answer is (C)")["correct"]
    assert not mmlu_pro.grade({"gold": "C"}, "")["correct"]


def test_task_prompt_file(tmp_path):
    crux, test, validation = write_task_data(tmp_path / "data")
    out = tmp_path / "tasks.jsonl"
    args = ["--cruxeval", str(crux), "--mmlu-pro-test", str(test), "--mmlu-pro-validation", str(validation)]
    assert (
        taskprompts.main(
            [*args, "--checkpoint", str(tmp_path), "--out", str(out), "--count", "3"], tokenizer=FakeTokenizer()
        )
        == 0
    )
    prompts = read_jsonl(out)
    assert sorted(p["task"] for p in prompts) == ["cruxeval_input"] * 3 + ["cruxeval_output"] * 3 + ["mmlu_pro"] * 3
    assert [len(p["prompt_ids"]) for p in prompts] == sorted(len(p["prompt_ids"]) for p in prompts)
    for p in prompts:
        assert taskprompts.task_of(p["idx"]) == p["task"] and p["prompt_ids"].count(256) == 1
    crux_groups = {p["group"] for p in prompts if p["task"].startswith("cruxeval")}
    assert len(crux_groups) == 3  # both directions of a problem share its group
    again = tmp_path / "again.jsonl"
    taskprompts.main(
        [*args, "--checkpoint", str(tmp_path), "--out", str(again), "--count", "3"], tokenizer=FakeTokenizer()
    )
    assert again.read_bytes() == out.read_bytes()
    ext = tmp_path / "ext.jsonl"
    taskprompts.main(
        [*args, "--checkpoint", str(tmp_path), "--out", str(ext), "--count", "3", "--offset", "3"],
        tokenizer=FakeTokenizer(),
    )
    assert not {p["idx"] for p in read_jsonl(ext)} & {p["idx"] for p in prompts}

    ids = tmp_path / "ids.txt"
    ids.write_text("71\n73\n75\n77\n")
    listed = tmp_path / "listed.jsonl"
    taskprompts.main(
        [*args, "--checkpoint", str(tmp_path), "--out", str(listed), "--count", "2", "--mmlu-pro-ids", str(ids)],
        tokenizer=FakeTokenizer(),
    )
    mmlu = [p for p in read_jsonl(listed) if p["task"] == "mmlu_pro"]
    assert {p["source_id"] for p in mmlu} <= {71, 73, 75, 77} and {p["pool"] for p in mmlu} == {"listed"}
    assert taskprompts.main([*args, "--check-data"]) == 1  # made-up files do not have the published row counts


def test_zero_shot_mmlu_pro_prompt_file(tmp_path):
    crux, test, validation = write_task_data(tmp_path / "data")
    args = ["--cruxeval", str(crux), "--mmlu-pro-test", str(test), "--mmlu-pro-validation", str(validation)]
    args += ["--checkpoint", str(tmp_path), "--count", "3"]
    fitted, full, only = tmp_path / "fitted.jsonl", tmp_path / "full.jsonl", tmp_path / "only.jsonl"
    assert taskprompts.main([*args, "--out", str(fitted)], tokenizer=FakeTokenizer()) == 0
    assert taskprompts.main([*args, "--out", str(full), "--mmlu-pro-zero-shot"], tokenizer=FakeTokenizer()) == 0
    assert (
        taskprompts.main(
            [*args, "--out", str(only), "--mmlu-pro-zero-shot", "--tasks", "mmlu_pro"], tokenizer=FakeTokenizer()
        )
        == 0
    )

    def by_task(path, mmlu):
        return [p for p in read_jsonl(path) if (p["task"] == "mmlu_pro") == mmlu]

    assert by_task(full, False) == by_task(fitted, False)  # the CRUXEval prompts do not change
    zero = by_task(full, True)
    assert sorted(zero, key=lambda p: p["idx"]) == sorted(read_jsonl(only), key=lambda p: p["idx"])
    assert {p["source_id"] for p in zero} == {p["source_id"] for p in by_task(fitted, True)}  # the same documents
    for p in zero:
        assert p["idx"] == taskprompts.MMLU_PRO_ZERO_SHOT_BASE + p["source_id"] and p["shots"] == 0
        assert taskprompts.task_of(p["idx"]) == "mmlu_pro"
        assert p["user_text"].count("Question:\n") == 1 and p["user_text"].count("Let's think step by step.") == 1
    assert not {p["idx"] for p in zero} & {p["idx"] for p in read_jsonl(fitted)}
    # a draw past the last CRUXEval row is possible when only MMLU-Pro is written
    tail = tmp_path / "tail.jsonl"
    tail_args = [*args[:-2], "--count", "2", "--offset", str(len(CRUX_ROWS)), "--out", str(tail)]
    with pytest.raises(ValueError, match="cannot take"):
        taskprompts.main(tail_args, tokenizer=FakeTokenizer())
    assert taskprompts.main([*tail_args, "--tasks", "mmlu_pro"], tokenizer=FakeTokenizer()) == 0
    assert [p["task"] for p in read_jsonl(tail)] == ["mmlu_pro"] * 2
    with pytest.raises(SystemExit):
        taskprompts.main([*args, "--out", str(tail), "--tasks", "mmlu"], tokenizer=FakeTokenizer())


def test_sameprompts_compares_the_records_of_the_listed_tasks(tmp_path, capsys):
    from swaptest import sameprompts

    crux, test, validation = write_task_data(tmp_path / "data")
    args = ["--cruxeval", str(crux), "--mmlu-pro-test", str(test), "--mmlu-pro-validation", str(validation)]
    args += ["--checkpoint", str(tmp_path), "--count", "3"]
    fitted, zero, only = tmp_path / "fitted.jsonl", tmp_path / "zero.jsonl", tmp_path / "only.jsonl"
    taskprompts.main([*args, "--out", str(fitted)], tokenizer=FakeTokenizer())
    taskprompts.main([*args, "--out", str(zero), "--mmlu-pro-zero-shot"], tokenizer=FakeTokenizer())
    taskprompts.main(
        [*args, "--out", str(only), "--mmlu-pro-zero-shot", "--tasks", "mmlu_pro"], tokenizer=FakeTokenizer()
    )
    crux_tasks = "cruxeval_input,cruxeval_output"
    assert sameprompts.main(["--reference", str(fitted), "--file", str(zero), "--tasks", crux_tasks]) == 0
    assert "6 records equal" in capsys.readouterr().out
    assert sameprompts.main(["--reference", str(zero), "--file", str(only), "--tasks", "mmlu_pro"]) == 0
    # the two MMLU-Pro prompt forms have different ids; a file without CRUXEval prompts is not "the same"
    assert sameprompts.main(["--reference", str(fitted), "--file", str(zero), "--tasks", "mmlu_pro"]) == 1
    assert sameprompts.main(["--reference", str(fitted), "--file", str(only), "--tasks", crux_tasks]) == 1
    assert "NOT the same prompts" in capsys.readouterr().out
    changed = read_jsonl(fitted)
    changed[0]["user_text"] += " "
    edited = tmp_path / "edited.jsonl"
    edited.write_text("".join(json.dumps(p) + chr(10) for p in changed), encoding="utf-8")
    assert (
        sameprompts.main(["--reference", str(fitted), "--file", str(edited), "--tasks", ",".join(taskprompts.TASKS)])
        == 1
    )
    assert "1 differ" in capsys.readouterr().out
