"""Teacher-forced readout: every arm's log-probabilities of the reference arm's sampled replies.

For each prompt with a reply of the reference arm (default A), the prompt plus that reply is fed to all arms in
one batch (prefill only). Per (prompt, arm) the output holds the summed and mean negative log-likelihood of the
reply tokens and, when the reference reply ended with a stop token, the log-probability of that token at that
position and the log-probability of any stop token there.

Usage: python -m swaptest.teacher_forced --checkpoint DIR [--patch FILE | --merge SPEC.json] --prompts prompts.jsonl
       --replies RUN_DIR/replies.jsonl --out tf.jsonl [--arms ABCD] [--reference A] [--chunk 256] [--threads N]
       python -m swaptest.teacher_forced --check --replies RUN_DIR/replies.jsonl --out tf.jsonl [--arms ABCD]
The rows of one prompt (one per arm) are appended with one write. Prompts already present in --out with a row
for every arm are skipped, so an interrupted pass can be restarted with the same command; rows of a prompt that
lacks some arms are removed first and that prompt is scored again.

--check verifies a finished output without loading the model: one row per arm for every reply of the reference
arm, every log-probability finite, and (reported, not enforced) the reference arm's mean NLL on its own samples
inside 0.05 .. 3 nats per token. Exit code 1 if a row is missing or a value is not finite.

With --merge SPEC.json (phase 2) the arms are those of a merge built in memory (`swaptest.run` explains the
option); the reference replies then come from another model's run (arm A of phase 1), so the reference arm need
not be among --arms.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch

from swaptest.engine import prefill
from swaptest.loading import load_merged_model, load_model
from swaptest.model import GrugModel
from swaptest.patch import ARM_ORDER, PHASE1_ARMS, arm_index
from swaptest.records import append_jsonl_many, drop_torn_tail, read_jsonl, write_jsonl_atomic

PLAUSIBLE_NLL = (0.05, 3.0)


def score_reply(
    model: GrugModel, prompt: list[int], reply: list[int], arms: str, stop_ids: Collection[int], chunk: int
) -> list[dict[str, Any]]:
    """Per arm: NLL of `reply` after `prompt`, and stop-token log-probabilities if `reply` ends with a stop."""
    seq, p, n = prompt + reply, len(prompt), len(reply)
    stopped = reply[-1] in stop_ids
    stop_pos = p + n - 2  # the position whose logits predict the final token
    ids = torch.tensor(seq).repeat(len(arms), 1)
    arm = torch.tensor([arm_index(a) for a in arms])
    token_lp = torch.empty(len(arms), n, dtype=torch.float64)
    any_stop = torch.full((len(arms),), float("nan"), dtype=torch.float64)
    stops = torch.tensor(sorted(stop_ids))

    def on_chunk(rows: slice, start: int, hidden: torch.Tensor) -> None:
        lo, hi = max(start, p - 1), min(start + hidden.shape[1], p + n - 1)  # predictor positions p-1 .. p+n-2
        if lo >= hi:
            return
        logp = torch.log_softmax(model.logits(hidden[:, lo - start : hi - start]).double(), dim=-1)
        targets = torch.tensor(seq[lo + 1 : hi + 1]).view(1, -1, 1).expand(len(arms), -1, 1)
        token_lp[:, lo - p + 1 : hi - p + 1] = logp.gather(-1, targets).squeeze(-1)
        if stopped and lo <= stop_pos < hi:
            any_stop[:] = torch.logsumexp(logp[:, stop_pos - lo, stops], dim=-1)

    cache = model.new_cache(len(arms), len(seq))
    prefill(model, cache, ids, torch.zeros(len(arms), dtype=torch.long), arm, chunk, len(arms), on_chunk)
    return [
        {
            "arm": a,
            "n_tokens": n,
            "nll_sum": float(-token_lp[i].sum()),
            "nll_mean": float(-token_lp[i].mean()),
            "stop_token": reply[-1] if stopped else None,
            "stop_logprob": float(token_lp[i, -1]) if stopped else None,
            "any_stop_logprob": float(any_stop[i]) if stopped else None,
        }
        for i, a in enumerate(arms)
    ]


def complete_prompts(rows: list[dict[str, Any]], arms: str) -> set[int]:
    """Prompt ids that have exactly one row for each of `arms`."""
    seen: dict[int, list[str]] = {}
    for r in rows:
        seen.setdefault(r["prompt_id"], []).append(r["arm"])
    return {p for p, a in seen.items() if sorted(a) == sorted(arms)}


def check_output(replies: list[dict[str, Any]], rows: list[dict[str, Any]], arms: str, reference: str) -> list[str]:
    """Problems of a finished output: missing or duplicate rows, values that are not finite."""
    problems = []
    complete = complete_prompts(rows, arms)
    missing = sorted(r["prompt_id"] for r in replies if r["prompt_id"] not in complete)
    if missing:
        problems.append(f"{len(missing)} replies of arm {reference} lack a row for every arm of {arms}: {missing[:10]}")
    for r in rows:
        values = [r["nll_sum"], r["nll_mean"]] + [
            r[k] for k in ("stop_logprob", "any_stop_logprob") if r[k] is not None
        ]
        if not all(math.isfinite(v) for v in values):
            problems.append(f"prompt {r['prompt_id']}, arm {r['arm']}: a value is not finite")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--merge", type=Path, help="spec file of a merge built in memory (arms P, N, G)")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--replies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=PHASE1_ARMS)
    parser.add_argument("--reference", default="A")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--check", action="store_true", help="only check a finished output")
    args = parser.parse_args(argv)
    if set(args.arms) - set(ARM_ORDER) or len(set(args.arms)) != len(args.arms):
        parser.error(f"--arms must be distinct letters from {ARM_ORDER}")
    if args.merge is None and args.reference not in args.arms:
        parser.error("--arms must include the reference arm")
    replies = sorted((r for r in read_jsonl(args.replies) if r["arm"] == args.reference), key=lambda r: r["prompt_id"])

    if args.check:
        rows = read_jsonl(args.out)
        problems = check_output(replies, rows, args.arms, args.reference)
        for problem in problems:
            print(problem)
        own = [r["nll_mean"] for r in rows if r["arm"] == args.reference and math.isfinite(r["nll_mean"])]
        if own:
            mean = statistics.fmean(own)
            lo, hi = PLAUSIBLE_NLL
            note = "inside" if lo <= mean <= hi else "OUTSIDE"
            print(f"arm {args.reference} mean NLL on its own replies: {mean:.4f} nats per token ({note} {lo} .. {hi})")
        print(
            f"{len(replies)} replies of arm {args.reference}, {len(rows)} rows: "
            f"{'complete' if not problems else f'{len(problems)} problems'}"
        )
        return 1 if problems else 0

    if args.checkpoint is None or args.prompts is None:
        parser.error("--checkpoint and --prompts are required unless --check is given")
    if args.threads:
        torch.set_num_threads(args.threads)
    prompts = {p["idx"]: p for p in read_jsonl(args.prompts)}
    drop_torn_tail(args.out)
    rows = read_jsonl(args.out)
    done = complete_prompts(rows, args.arms)
    if len(rows) != len(done) * len(args.arms):  # rows of an interrupted prompt: remove them, score it again
        write_jsonl_atomic(args.out, [r for r in rows if r["prompt_id"] in done])
    todo = [r for r in replies if r["prompt_id"] not in done]
    print(f"{len(replies)} replies of arm {args.reference}; {len(done)} prompts scored before, {len(todo)} to score")
    if not todo:
        return 0
    if args.merge:
        loaded = load_merged_model(args.checkpoint, args.merge, args.arms, print)
    else:
        loaded = load_model(args.checkpoint, args.patch, args.arms, print)
    with torch.inference_mode():
        for i, reply in enumerate(todo):
            scored = score_reply(
                loaded.model,
                prompts[reply["prompt_id"]]["prompt_ids"],
                reply["token_ids"],
                args.arms,
                loaded.stop_ids,
                args.chunk,
            )
            append_jsonl_many(
                args.out,
                [
                    {
                        "prompt_id": reply["prompt_id"],
                        "reference": args.reference,
                        "reference_stop_reason": reply["stop_reason"],
                        **row,
                    }
                    for row in scored
                ],
            )
            summary = ", ".join(f"{r['arm']} nll {r['nll_mean']:.3f}" for r in scored)
            print(f"[{i + 1}/{len(todo)}] prompt {reply['prompt_id']} ({reply['n_new']} tokens): {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
