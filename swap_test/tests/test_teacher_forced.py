"""Teacher-forced readout equals log-probabilities computed directly from a full forward."""

import json

import torch
from helpers import STOP_IDS, perturbed, random_weights, tiny_config, write_run_inputs

from swaptest import teacher_forced
from swaptest.checks import full_logits
from swaptest.config import ModelConfig
from swaptest.model import GrugModel, Overrides
from swaptest.patch import ARM_GROUPS, GROUP_PATTERNS, arm_index
from swaptest.records import read_jsonl
from swaptest.teacher_forced import score_reply

CFG = ModelConfig.from_dict(tiny_config())
WEIGHTS = random_weights(CFG, seed=4)
PATCH = perturbed(WEIGHTS, CFG, scale=1.0)
TABLES = {
    arm_index(a): {n: t for n, t in PATCH.items() if any(GROUP_PATTERNS[g].fullmatch(n) for g in ARM_GROUPS[a])}
    for a in "ABCD"
}
MODEL = GrugModel(CFG, WEIGHTS, Overrides.from_arm_tables(TABLES))
PROMPT = [256, 5, 6, 7, 8]
REPLY = [9, 10, 11, 12, 13, 14, 15, 260]


def direct(arm: str) -> torch.Tensor:
    seq = torch.tensor([PROMPT + REPLY])
    logp = torch.log_softmax(full_logits(MODEL, seq, torch.tensor([arm_index(arm)])).double(), -1)[0]
    positions = torch.arange(len(PROMPT) - 1, len(PROMPT) + len(REPLY) - 1)
    return logp[positions], logp[positions, torch.tensor(REPLY)]


def test_scores_match_direct_log_probabilities():
    for chunk in (3, 64):
        rows = score_reply(MODEL, PROMPT, REPLY, "ABCD", STOP_IDS, chunk)
        for row in rows:
            full, token_lp = direct(row["arm"])
            assert abs(row["nll_sum"] + float(token_lp.sum())) < 1e-4
            assert abs(row["stop_logprob"] - float(token_lp[-1])) < 1e-5
            assert abs(row["any_stop_logprob"] - float(torch.logsumexp(full[-1, STOP_IDS], -1))) < 1e-5
            assert row["stop_token"] == 260 and row["n_tokens"] == len(REPLY)
    assert len({round(r["nll_mean"], 6) for r in rows}) > 1  # the arms differ


def test_reply_at_the_cap_has_no_stop_fields():
    row = score_reply(MODEL, PROMPT, REPLY[:-1], "A", STOP_IDS, 4)[0]
    assert row["stop_logprob"] is None and row["any_stop_logprob"] is None and row["stop_token"] is None


def test_interrupted_output_is_completed_and_checked(tmp_path):
    ckpt, patch, prompts = write_run_inputs(tmp_path, n_prompts=3)
    replies = tmp_path / "replies.jsonl"
    rows = [
        {
            "arm": a,
            "prompt_id": 100 + i,
            "token_ids": [9 + i, 10, 11, *([260] if i != 1 else [12])],
            "stop_reason": 260 if i != 1 else "cap",
            "n_new": 4,
        }
        for i in range(3)
        for a in "AB"
    ]
    replies.write_text("".join(json.dumps(r) + "\n" for r in rows))
    base = ["--replies", str(replies), "--arms", "ABCD"]
    full, partial = tmp_path / "full.jsonl", tmp_path / "partial.jsonl"
    common = ["--checkpoint", str(ckpt), "--patch", str(patch), "--prompts", str(prompts), "--chunk", "4", *base]
    assert teacher_forced.main([*common, "--out", str(full)]) == 0
    complete = read_jsonl(full)
    assert len(complete) == 12 and teacher_forced.main(["--check", *base, "--out", str(full)]) == 0
    # an interrupted pass: prompt 100 complete, prompt 101 with two of its four rows, and a torn line
    partial.write_text("".join(json.dumps(r) + "\n" for r in complete[:6]) + '{"prompt_id": 101')
    assert teacher_forced.main(["--check", *base, "--out", str(partial)]) == 1
    assert teacher_forced.main([*common, "--out", str(partial)]) == 0
    key = lambda r: (r["prompt_id"], r["arm"])  # noqa: E731
    assert sorted(read_jsonl(partial), key=key) == sorted(complete, key=key)
    assert teacher_forced.main(["--check", *base, "--out", str(partial)]) == 0
