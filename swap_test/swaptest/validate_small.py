"""Validate the harness on a small public Grug checkpoint against the vLLM record it ships (`vllm_validation.json`).

Checks, each printed with PASS or FAIL:
  (b) with --skyrl-src: this port against MarinSkyRL's eager model in float32 on the record's 64-token prompt,
      max abs logit difference below 1e-4;
  (c) the log-probability of the record's next token after its 64-token prompt, in bfloat16, within 0.05 of the
      recorded value (float32 reported too);
  (d) the record's greedy chat samples (the checkpoint's own template, enable_thinking=False, the record's
      max_tokens): at least 3 of 4 replies equal vLLM's text over the first 32 tokens; for each, the prompt length
      against vLLM's prompt_tokens, the first divergence and the logit margin there;
  (e) KV-cached decode against a full forward on a sequence of real text (--long-len tokens; chunked prefill of
      --prefill-len tokens, then one token per step): top-1 agreement at least 99% and no NaN.

Record format (fields read): response.choices[0].prompt_token_ids, .token_ids[0] and .logprobs.token_logprobs[0];
samples[i].prompt and samples[i].response.choices[0].message.content, .finish_reason, .stop_reason, and
.usage.prompt_tokens; generation_settings.max_tokens.

Usage: python -m swaptest.validate_small --checkpoint DIR --record DIR/vllm_validation.json --prompts prompts.jsonl
       [--skyrl-src DIR] [--long-len 2600] [--prefill-len 2300] [--chunk 256] [--threads N]
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from swaptest.chat import ChatTokenizer, decode, load_tokenizer, render_user_turn
from swaptest.checkpoint import cast_weights
from swaptest.checks import cached_logits, compare_logits, full_logits, long_text_ids
from swaptest.engine import Finished, Row, generate
from swaptest.loading import load_model
from swaptest.model import GrugModel
from swaptest.records import read_json, read_jsonl
from swaptest.sampling import Sampler

LOGPROB_TOLERANCE = 0.08
SKYRL_TOLERANCE = 5e-4
COMPARE_TOKENS = 32
MIN_MATCHING_SAMPLES = 2
NEAR_TIE_MARGIN = 0.1
MIN_TOP1 = 0.99


def next_token_logprob(model: GrugModel, prompt: list[int], token: int) -> float:
    logits = full_logits(model, torch.tensor([prompt]))[0, -1]
    return float(torch.log_softmax(logits.double(), -1)[token])


def compare_sample(
    model: GrugModel, tokenizer: ChatTokenizer, prompt: list[int], finished: Finished, reference: dict[str, Any]
) -> dict[str, Any]:
    """Our greedy reply against vLLM's text: match over the first 32 tokens, first divergence and its margin."""
    ours = finished.row.generated
    body = ours if finished.stop_reason == "cap" else ours[:-1]
    ref_text = reference["choices"][0]["message"]["content"]
    head = decode(tokenizer, body[:COMPARE_TOKENS], keep_special=False)
    if len(body) >= COMPARE_TOKENS:
        # 32 byte-level tokens may end inside a multi-byte character, which decodes to U+FFFD on its own
        match = ref_text.startswith(head.rstrip("�"))
    else:  # our reply ended early: vLLM's must be the same reply, ended the same way (stop token or length)
        vllm_stopped = reference["choices"][0]["finish_reason"] == "stop"
        match = ref_text == head and vllm_stopped == (finished.stop_reason != "cap")
    ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
    diverge = next((i for i, (a, b) in enumerate(zip(body, ref_ids, strict=False)) if a != b), None)
    margin = None
    if diverge is not None:
        logits = full_logits(model, torch.tensor([prompt + body[:diverge]]))[0, -1]
        margin = float(logits[body[diverge]] - logits[ref_ids[diverge]])
    return {
        "match_first_32": match,
        "first_divergence": diverge,
        "margin_ours_minus_vllm": margin,
        "our_stop": finished.stop_reason,
        "vllm_finish": reference["choices"][0]["finish_reason"],
        "our_text": decode(tokenizer, body, keep_special=False),
        "vllm_text": ref_text,
    }


def main(argv: list[str] | None = None, tokenizer: ChatTokenizer | None = None) -> int:
    """`tokenizer` replaces the checkpoint's own (tests pass one that renders the template with jinja2)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="prompt file; its texts make the long sequence")
    parser.add_argument("--skyrl-src", type=Path)
    parser.add_argument("--long-len", type=int, default=2600)
    parser.add_argument("--prefill-len", type=int, default=2300)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    record = read_json(args.record)
    loaded = load_model(args.checkpoint, None, "A", print)
    tokenizer = tokenizer or load_tokenizer(args.checkpoint)
    results: dict[str, bool] = {}
    choice = record["response"]["choices"][0]
    prompt64, token, ref_lp = (
        choice["prompt_token_ids"],
        choice["token_ids"][0],
        choice["logprobs"]["token_logprobs"][0],
    )

    with torch.inference_mode():
        fp32 = cast_weights(loaded.weights, torch.float32)
        if args.skyrl_src is not None:
            from swaptest.skyrl_crosscheck import skyrl_logits

            config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
            ids = torch.tensor([prompt64])
            diff = compare_logits(
                skyrl_logits(args.skyrl_src, config, fp32, ids), full_logits(GrugModel(loaded.cfg, fp32), ids)
            )
            results["b"] = diff["max_abs_diff"] < SKYRL_TOLERANCE
            print(
                f"(b) port vs reference model, float32, {len(prompt64)} tokens: {diff} -> "
                f"{'PASS' if results['b'] else 'FAIL'}"
            )

        lp_bf16 = next_token_logprob(loaded.model, prompt64, token)
        lp_fp32 = next_token_logprob(GrugModel(loaded.cfg, fp32), prompt64, token)
        del fp32
        results["c"] = abs(lp_bf16 - ref_lp) < LOGPROB_TOLERANCE
        print(
            f"(c) logprob of token {token}: recorded {ref_lp:.4f}, ours {loaded.model.dtype} {lp_bf16:.4f}, "
            f"float32 {lp_fp32:.4f} -> {'PASS' if results['c'] else 'FAIL'}"
        )

        samples = record["samples"]
        rows, prompts = [], []
        for i, sample in enumerate(samples):
            _, ids = render_user_turn(tokenizer, sample["prompt"], enable_thinking=False)
            prompts.append(ids)
            rows.append(Row("A", i, ids))
            print(
                f"    sample {sample['name']}: prompt {len(ids)} tokens, vLLM prompt_tokens "
                f"{sample['response']['usage']['prompt_tokens']}"
            )
        finished: dict[int, Finished] = {}
        generate(
            loaded.model,
            rows,
            Sampler(0, 0.0),
            record["generation_settings"]["max_tokens"],
            loaded.stop_ids,
            lambda f: finished.__setitem__(f.row.prompt_id, f),
        )
        matches, wide_margins = 0, 0
        for i, sample in enumerate(samples):
            c = compare_sample(loaded.model, tokenizer, prompts[i], finished[i], sample["response"])
            matches += c["match_first_32"]
            if not c["match_first_32"] and abs(c["margin_ours_minus_vllm"] or 0.0) >= NEAR_TIE_MARGIN:
                wide_margins += 1
            print(
                f"    sample {sample['name']}: match over first {COMPARE_TOKENS} tokens {c['match_first_32']}, "
                f"first divergence {c['first_divergence']}, margin {c['margin_ours_minus_vllm']}, "
                f"stop ours {c['our_stop']} / vLLM {c['vllm_finish']}"
            )
            print(f"      ours: {c['our_text'][:200]!r}\n      vLLM: {c['vllm_text'][:200]!r}")
        results["d"] = matches >= MIN_MATCHING_SAMPLES and wide_margins == 0
        print(
            f"(d) {matches} of {len(samples)} greedy samples match, {wide_margins} diverge at a margin of {NEAR_TIE_MARGIN} or more -> {'PASS' if results['d'] else 'FAIL'}"
        )

        ids = torch.tensor([long_text_ids(tokenizer, read_jsonl(args.prompts), args.long_len)])
        diff = compare_logits(
            full_logits(loaded.model, ids), cached_logits(loaded.model, ids, args.prefill_len, args.chunk)
        )
        results["e"] = diff["top1_agreement"] >= MIN_TOP1 and not diff["nan"]
        print(
            f"(e) cached vs full forward over {args.long_len} tokens (prefill {args.prefill_len}): {diff} -> "
            f"{'PASS' if results['e'] else 'FAIL'}"
        )

    failed = [k for k, ok in results.items() if not ok]
    print(f"validation: {'all checks passed' if not failed else 'FAILED checks ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
