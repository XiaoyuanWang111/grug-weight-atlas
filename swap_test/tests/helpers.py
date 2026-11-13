"""Tiny random Grug checkpoints, fake patches, made-up task data and (from `fake_tokenizer`) a byte-level tokenizer
with the real chat template, for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from fake_tokenizer import FIXTURES as FIXTURES
from fake_tokenizer import SPECIALS
from fake_tokenizer import FakeTokenizer as FakeTokenizer
from safetensors.torch import save_file

from swaptest.checkpoint import expected_shapes
from swaptest.config import ModelConfig
from swaptest.patch import GROUP_PATTERNS

STOP_IDS = [257, 260]


def tiny_config(**changes: Any) -> dict[str, Any]:
    """A 5-layer config (long layers 3 and 4) with 4 query / 2 KV heads and a 4-token window."""
    raw = {
        "model_type": "grug_moe",
        "architectures": ["GrugMoeForCausalLM"],
        "grugmoe_artifact_schema_version": 1,
        "vocab_size": 264,
        "hidden_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24,
        "sliding_window": 4,
        "qk_mult": 1.75,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "bos_token_id": 256,
        "eos_token_id": 257,
    }
    raw.update(changes)
    return raw


def random_weights(cfg: ModelConfig, seed: int = 0, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Random weights at scales that keep activations and gates in a useful range."""
    g = torch.Generator().manual_seed(seed)
    weights = {}
    for name, shape in sorted(expected_shapes(cfg).items()):
        t = torch.randn(shape, generator=g)
        if len(shape) == 1:
            t = 1.0 + 0.1 * t if name.endswith("norm.weight") else 0.1 * t
        elif "gated_norm" in name or "attn_gate" in name:
            t = 0.1 * t
        elif name != "model.embed_tokens.weight":
            t = t / shape[-1] ** 0.5
        weights[name] = t.to(torch.float32 if name.endswith(".mlp.router.bias") else dtype)
    return weights


def perturbed(
    weights: dict[str, torch.Tensor], cfg: ModelConfig, scale: float = 0.5, seed: int = 1
) -> dict[str, torch.Tensor]:
    """Patch tensors: the gate and router tensors of every layer with added noise."""
    g = torch.Generator().manual_seed(seed)
    names = [n for n in weights if any(p.fullmatch(n) for p in GROUP_PATTERNS.values())]
    out = {}
    for name in sorted(names):
        w = weights[name].float()
        out[name] = (w + scale * w.std() * torch.randn(w.shape, generator=g)).to(weights[name].dtype)
    assert len(out) == 4 * cfg.num_layers
    return out


def write_checkpoint(
    directory: Path, raw_config: dict[str, Any], weights: dict[str, torch.Tensor], shards: int = 3
) -> None:
    """Save as several safetensors shards with an index, plus config.json and generation_config.json."""
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(weights)
    weight_map = {}
    for s in range(shards):
        part = names[s::shards]
        file = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: weights[n].contiguous() for n in part}, str(directory / file))
        weight_map |= {n: file for n in part}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps(raw_config))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": STOP_IDS}))


def write_run_inputs(
    directory: Path, dtype: torch.dtype = torch.float32, n_prompts: int = 3, seed: int = 0
) -> tuple[Path, Path, Path]:
    """A tiny checkpoint, a patch file and a prompt file with random prompts; returns their paths."""
    raw = tiny_config()
    cfg = ModelConfig.from_dict(raw)
    weights = random_weights(cfg, seed=seed, dtype=dtype)
    write_checkpoint(directory / "ckpt", raw, weights)
    patch = directory / "patch.safetensors"
    save_file({n: t.contiguous() for n, t in perturbed(weights, cfg, scale=1.0).items()}, str(patch))
    g = torch.Generator().manual_seed(seed + 100)
    prompts = directory / "prompts.jsonl"
    rows = [
        {
            "idx": 100 + i,
            "question": f"q{i}",
            "answer": f"#### {i}",
            "gold": str(i),
            "prompt_ids": [SPECIALS["<|begin_of_text|>"], *torch.randint(0, 256, (3 + 2 * i,), generator=g).tolist()],
        }
        for i in range(n_prompts)
    ]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return directory / "ckpt", patch, prompts


CRUX_ROWS = [
    {"id": "sample_0", "code": "def f(x):\n    return x + 1", "input": "16", "output": "17"},
    {"id": "sample_1", "code": "def f(s):\n    return s + '=='", "input": "'a'", "output": "'a=='"},
    {"id": "sample_2", "code": "def f(a, b):\n    return a + sum(b)", "input": "2, [1, 3]", "output": "6"},
    {"id": "sample_3", "code": "def f(text):\n    return text.upper()", "input": "'hi'", "output": "'HI'"},
    {
        "id": "sample_4",
        "code": "def f(n):\n    return [i * i for i in range(n)]",
        "input": "4",
        "output": "[0, 1, 4, 9]",
    },
    {
        "id": "sample_5",
        "code": "def f(d):\n    return sorted(d.items())",
        "input": "{'b': 2, 'a': 1}",
        "output": "[('a', 1), ('b', 2)]",
    },
]


def mmlu_rows(n: int, with_cot: bool, first_id: int = 70) -> list[dict[str, Any]]:
    """Made-up MMLU-Pro rows in two categories; every third row pads its options with "N/A"."""
    rows = []
    for i in range(n):
        options = [f"option {k} of question {first_id + i}" for k in range(4 + i % 3)]
        if i % 3 == 0:
            options += ["N/A", "N/A"]
        answer = "ABCD"[i % 4]
        cot = f"A: Let's think step by step. Worked reasoning number {i}. The answer is ({answer})." if with_cot else ""
        rows.append(
            {
                "question_id": first_id + i,
                "question": f"Made-up question {first_id + i} about {'math' if i % 2 else 'law'}?",
                "options": options,
                "answer": answer,
                "answer_index": "ABCD".index(answer),
                "cot_content": cot,
                "category": "math" if i % 2 else "law",
                "src": "made-up",
            }
        )
    return rows


def write_task_data(directory: Path) -> tuple[Path, Path, Path]:
    """Made-up CRUXEval test file and MMLU-Pro test and validation files (JSONL); returns their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    files = (directory / "crux.jsonl", directory / "mmlu_test.jsonl", directory / "mmlu_validation.jsonl")
    for path, rows in zip(files, (CRUX_ROWS, mmlu_rows(8, False), mmlu_rows(6, True, first_id=1)), strict=True):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return files
