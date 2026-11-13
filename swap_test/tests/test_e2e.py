"""End to end on a tiny random bfloat16 checkpoint in shards with a fake patch: prompts, run, teacher-forced pass,
analysis."""

import json

import torch
from helpers import FakeTokenizer, perturbed, random_weights, tiny_config, write_checkpoint
from safetensors.torch import save_file

from swaptest import analyze, prompts, run, teacher_forced
from swaptest.config import ModelConfig
from swaptest.records import read_jsonl


def test_pipeline(tmp_path):
    raw = tiny_config()
    cfg = ModelConfig.from_dict(raw)
    weights = random_weights(cfg, seed=21, dtype=torch.bfloat16)
    write_checkpoint(tmp_path / "ckpt", raw, weights, shards=4)
    patch = tmp_path / "patch.safetensors"
    save_file({n: t.contiguous() for n, t in perturbed(weights, cfg, scale=1.0).items()}, str(patch))
    gsm8k = tmp_path / "test.jsonl"
    gsm8k.write_text(
        "".join(
            json.dumps({"question": f"What is {i} + {i}?", "answer": f"{i} + {i}\n#### {2 * i}"}) + "\n"
            for i in range(6)
        )
    )

    ckpt, prompt_file, out = tmp_path / "ckpt", tmp_path / "prompts.jsonl", tmp_path / "run"
    prompts.main(
        ["--gsm8k", str(gsm8k), "--checkpoint", str(ckpt), "--out", str(prompt_file), "--count", "4"],
        tokenizer=FakeTokenizer(),
    )
    run.main(
        [
            "--checkpoint",
            str(ckpt),
            "--patch",
            str(patch),
            "--prompts",
            str(prompt_file),
            "--out",
            str(out),
            "--cap",
            "32",
            "--batch-size",
            "16",
            "--prefill-chunk",
            "16",
        ],
        tokenizer=FakeTokenizer(),
    )
    replies = read_jsonl(out / "replies.jsonl")
    assert sorted((r["arm"], r["prompt_id"]) for r in replies) == sorted(
        (a, p["idx"]) for p in read_jsonl(prompt_file) for a in "ABCD"
    )

    tf = tmp_path / "tf.jsonl"
    teacher_forced.main(
        [
            "--checkpoint",
            str(ckpt),
            "--patch",
            str(patch),
            "--prompts",
            str(prompt_file),
            "--replies",
            str(out / "replies.jsonl"),
            "--out",
            str(tf),
            "--chunk",
            "8",
        ]
    )
    scores = read_jsonl(tf)
    assert len(scores) == 16 and all(r["nll_mean"] > 0 for r in scores)

    summary = tmp_path / "summary.json"
    code = analyze.main(
        [
            "--prompts",
            str(prompt_file),
            "--replies",
            str(out / "replies.jsonl"),
            "--teacher-forced",
            str(tf),
            "--json",
            str(summary),
        ]
    )
    assert code == 0
    s = json.loads(summary.read_text())
    for arm in "ABCD":
        rows = [r for r in replies if r["arm"] == arm]
        assert s["arms"][arm]["capped"] == sum(r["stop_reason"] == "cap" for r in rows)
    assert set(s["comparisons"]) == {"B", "C", "D"} and set(s["teacher_forced"]) == set("ABCD")
