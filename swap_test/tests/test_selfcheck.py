"""The 67B self-check and the small-model validation, run end to end on a tiny checkpoint with the fake tokenizer."""

import json

import pytest
import torch
from helpers import STOP_IDS, FakeTokenizer, perturbed, random_weights, tiny_config, write_checkpoint
from safetensors.torch import save_file

from swaptest import prompts as prompts_module
from swaptest import selfcheck, validate_small
from swaptest.chat import decode, render_user_turn
from swaptest.checks import full_logits
from swaptest.config import ModelConfig
from swaptest.engine import Finished, Row, generate
from swaptest.loading import load_model
from swaptest.sampling import Sampler


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("selfcheck")
    raw = tiny_config()
    cfg = ModelConfig.from_dict(raw)
    weights = random_weights(cfg, seed=31)
    write_checkpoint(root / "ckpt", raw, weights)
    patch = root / "patch.safetensors"
    save_file({n: t.contiguous() for n, t in perturbed(weights, cfg, scale=1.0).items()}, str(patch))
    gsm8k = root / "test.jsonl"
    gsm8k.write_text(
        "".join(
            json.dumps(
                {
                    "question": f"Sam has {i} pens and buys {i + 2} more. How many now?",
                    "answer": f"{i} + {i + 2} = {2 * i + 2}\n#### {2 * i + 2}",
                }
            )
            + "\n"
            for i in range(8)
        )
    )
    prompt_file = root / "prompts.jsonl"
    prompts_module.main(
        ["--gsm8k", str(gsm8k), "--checkpoint", str(root / "ckpt"), "--out", str(prompt_file), "--count", "4"],
        tokenizer=FakeTokenizer(),
    )
    return root, weights, cfg, patch, prompt_file


SMALL_ARGS = [
    "--long-len",
    "160",
    "--prefill-len",
    "130",
    "--chunk",
    "16",
    "--mixed-steps",
    "4",
    "--batches",
    "1,4,8",
    "--steps",
    "2",
    "--per-arm",
    "4",
    "--cap",
    "64",
    "--batch-size",
    "8",
    "--min-free-gib",
    "0",
]


def test_selfcheck_passes_and_writes_its_summary(inputs):
    root, _, _, patch, prompt_file = inputs
    out = root / "selfcheck.json"
    code = selfcheck.main(
        [
            "--checkpoint",
            str(root / "ckpt"),
            "--patch",
            str(patch),
            "--prompts",
            str(prompt_file),
            "--json",
            str(out),
            "--max-hours",
            "1000",
            *SMALL_ARGS,
        ],
        tokenizer=FakeTokenizer(),
    )
    summary = json.loads(out.read_text())
    assert code == 0 and summary["exit_code"] == 0
    assert set(summary["passed"]) == {"load", "template", "cached", "mixed", "speed"} and all(
        summary["passed"].values()
    )
    assert summary["cached"]["top1_agreement"] == 1.0 and summary["speed"]["batch_size"] == 8
    assert set(summary["speed"]["projected_hours"]) == set(selfcheck.SCENARIOS)


def test_selfcheck_stops_when_the_projection_is_too_long(inputs):
    root, _, _, patch, prompt_file = inputs
    code = selfcheck.main(
        [
            "--checkpoint",
            str(root / "ckpt"),
            "--patch",
            str(patch),
            "--prompts",
            str(prompt_file),
            "--max-hours",
            "0",
            *SMALL_ARGS,
        ],
        tokenizer=FakeTokenizer(),
    )
    assert code == 4


def test_selfcheck_reports_a_bad_patch_as_a_load_failure(inputs, tmp_path):
    root, weights, cfg, _, prompt_file = inputs
    tensors = perturbed(weights, cfg, scale=1.0)
    name = "model.layers.0.self_attn.attn_gate.weight"
    tensors[name] = weights[name].clone()  # a patch tensor equal to the checkpoint's
    bad = tmp_path / "bad.safetensors"
    save_file({n: t.contiguous() for n, t in tensors.items()}, str(bad))
    out = tmp_path / "selfcheck.json"
    code = selfcheck.main(
        [
            "--checkpoint",
            str(root / "ckpt"),
            "--patch",
            str(bad),
            "--prompts",
            str(prompt_file),
            "--json",
            str(out),
            *SMALL_ARGS,
        ],
        tokenizer=FakeTokenizer(),
    )
    assert code == 3 and json.loads(out.read_text())["passed"] == {"load": False}


def test_selfcheck_detects_a_prompt_file_from_another_template(inputs, tmp_path):
    root, _, _, patch, prompt_file = inputs
    rows = [json.loads(line) for line in prompt_file.read_text().splitlines()]
    rows[0]["prompt_ids"] = rows[0]["prompt_ids"][:-1]
    edited = tmp_path / "prompts.jsonl"
    edited.write_text("".join(json.dumps(r) + "\n" for r in rows))
    code = selfcheck.main(
        [
            "--checkpoint",
            str(root / "ckpt"),
            "--patch",
            str(patch),
            "--prompts",
            str(edited),
            "--max-hours",
            "1000",
            *SMALL_ARGS,
        ],
        tokenizer=FakeTokenizer(),
    )
    assert code == 1


def make_record(ckpt, tokenizer, max_tokens=40):
    """A validation record in the vLLM record's format, filled with this port's own outputs."""
    loaded = load_model(ckpt, None, "A", lambda _: None)
    prompt64 = [256, *torch.randint(0, 256, (63,), generator=torch.Generator().manual_seed(5)).tolist()]
    logits = full_logits(loaded.model, torch.tensor([prompt64]))[0, -1]
    token = int(logits.argmax())
    logprob = float(torch.log_softmax(logits.double(), -1)[token])
    texts = ["What is 2 + 2?", "Name a colour.", "Count to three.", "Say hello."]
    rows = [Row("A", i, render_user_turn(tokenizer, t, enable_thinking=False)[1]) for i, t in enumerate(texts)]
    done: dict[int, Finished] = {}
    generate(loaded.model, rows, Sampler(0, 0.0), max_tokens, STOP_IDS, lambda f: done.__setitem__(f.row.prompt_id, f))
    samples = []
    for i, t in enumerate(texts):
        f = done[i]
        body = f.row.generated if f.stop_reason == "cap" else f.row.generated[:-1]
        samples.append(
            {
                "name": f"s{i}",
                "prompt": t,
                "response": {
                    "choices": [
                        {
                            "message": {"content": decode(tokenizer, body, keep_special=False)},
                            "finish_reason": "length" if f.stop_reason == "cap" else "stop",
                            "stop_reason": None,
                        }
                    ],
                    "usage": {"prompt_tokens": len(rows[i].prompt)},
                },
            }
        )
    return {
        "response": {
            "choices": [{"prompt_token_ids": prompt64, "token_ids": [token], "logprobs": {"token_logprobs": [logprob]}}]
        },
        "samples": samples,
        "generation_settings": {"max_tokens": max_tokens},
    }


def test_validate_small_passes_on_a_record_of_its_own_outputs_and_fails_on_a_shifted_logprob(inputs, tmp_path):
    root, _, _, _, prompt_file = inputs
    tokenizer = FakeTokenizer()
    record = make_record(root / "ckpt", tokenizer)
    path = tmp_path / "vllm_validation.json"
    path.write_text(json.dumps(record))
    args = [
        "--checkpoint",
        str(root / "ckpt"),
        "--record",
        str(path),
        "--prompts",
        str(prompt_file),
        "--long-len",
        "160",
        "--prefill-len",
        "130",
        "--chunk",
        "16",
    ]
    assert validate_small.main(args, tokenizer=tokenizer) == 0
    record["response"]["choices"][0]["logprobs"]["token_logprobs"][0] -= 0.1
    path.write_text(json.dumps(record))
    assert validate_small.main(args, tokenizer=tokenizer) == 1
