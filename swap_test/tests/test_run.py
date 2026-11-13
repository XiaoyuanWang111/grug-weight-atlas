"""The run script: interrupted and restarted runs give the same replies as an uninterrupted run."""

import json

import pytest
from helpers import FakeTokenizer, write_run_inputs

from swaptest import run as run_module
from swaptest.engine import Row
from swaptest.records import read_jsonl


class Interrupted(Exception):
    pass


def start(inputs, out, *extra, hook=None):
    ckpt, patch, prompts = inputs
    args = run_module.parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--patch",
            str(patch),
            "--prompts",
            str(prompts),
            "--out",
            str(out),
            "--cap",
            "24",
            "--seed",
            "5",
            "--prefill-chunk",
            "4",
            "--prefill-rows",
            "3",
            "--log-every",
            "4",
            *extra,
        ]
    )
    run_module.run(args, FakeTokenizer(), step_hook=hook)


def interrupt_after(calls):
    count = {"n": 0}

    def hook(step):
        count["n"] += 1
        if count["n"] == calls:
            raise Interrupted

    return hook


def replies(out):
    return {
        (r["arm"], r["prompt_id"]): (r["token_ids"], r["stop_reason"], r["text"], r["n_new"])
        for r in read_jsonl(out / "replies.jsonl")
    }


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return write_run_inputs(tmp_path_factory.mktemp("inputs"))


@pytest.fixture(scope="module")
def uninterrupted(inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    start(inputs, out, "--batch-size", "12")
    return replies(out)


def test_uninterrupted_run_writes_every_pair(uninterrupted):
    assert len(uninterrupted) == 12
    reasons = {v[1] for v in uninterrupted.values()}
    assert "cap" in reasons and reasons - {"cap"}  # both stops and caps occur
    for ids, reason, _, n_new in uninterrupted.values():
        assert n_new == len(ids) and (reason == "cap" or ids[-1] == reason)


@pytest.mark.parametrize(("batch", "calls", "snapshot"), [("12", 7, "3"), ("12", 7, "50"), ("5", 30, "2")])
def test_restart_resumes_and_matches_uninterrupted_run(inputs, uninterrupted, tmp_path, batch, calls, snapshot):
    with pytest.raises(Interrupted):
        start(inputs, tmp_path, "--batch-size", batch, "--snapshot-every", snapshot, hook=interrupt_after(calls))
    before = replies(tmp_path)
    assert len(before) < 12
    with (tmp_path / "replies.jsonl").open("a") as f:
        f.write('{"arm": "A", "prompt_id": 1')  # a torn final line, as after a crash mid-write
    start(inputs, tmp_path, "--batch-size", batch, "--snapshot-every", snapshot)
    assert replies(tmp_path) == uninterrupted
    assert not (tmp_path / "inflight.json").exists()


def test_restart_with_other_settings_is_refused(inputs, tmp_path):
    with pytest.raises(Interrupted):
        start(inputs, tmp_path, hook=interrupt_after(2))
    saved = json.loads((tmp_path / "run_config.json").read_text())
    assert saved["cap"] == 24 and saved["arms"] == "ABCD"
    with pytest.raises(SystemExit, match="other settings"):
        start(inputs, tmp_path, "--seed", "6")


def test_waves_keep_the_arms_of_a_prompt_together():
    rows = [Row(a, p, [1]) for p, arms in ((1, "BCD"), (2, "ABCD"), (3, "ABCD"), (4, "AB")) for a in arms]
    waves = run_module.make_waves(rows, 8)
    assert [[(r.prompt_id, r.arm) for r in w] for w in waves] == [
        [(1, "B"), (1, "C"), (1, "D"), (2, "A"), (2, "B"), (2, "C"), (2, "D")],
        [(3, "A"), (3, "B"), (3, "C"), (3, "D"), (4, "A"), (4, "B")],
    ]
    assert [len(w) for w in run_module.make_waves(rows, 3)] == [3, 3, 1, 3, 3]  # 3 < 4 arms: prompts split


def test_prompts_with_saved_progress_come_first_with_all_their_arms():
    prompts = [{"idx": i, "prompt_ids": [1]} for i in (10, 11, 12)]
    inflight = {("C", 12): {"generated": [5, 6], "elapsed": 1.0}}
    rows = run_module.pending_rows(prompts, "ABCD", {("A", 12)}, inflight)
    assert [(r.prompt_id, r.arm) for r in rows[:3]] == [(12, "B"), (12, "C"), (12, "D")]
    assert rows[1].generated == [5, 6] and rows[1].elapsed == 1.0 and len(rows) == 11


def test_finished_run_is_a_no_op(inputs, tmp_path):
    start(inputs, tmp_path, "--arms", "AC", "--limit", "2")
    first = (tmp_path / "replies.jsonl").read_bytes()
    start(inputs, tmp_path, "--arms", "AC", "--limit", "2")
    assert (tmp_path / "replies.jsonl").read_bytes() == first and len(replies(tmp_path)) == 4
