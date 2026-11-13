"""Load the large checkpoint with the patch overrides and check the harness on it before the long run.

Checks, each printed with PASS or FAIL (a JSON summary is written with --json):
  load      the patch groups, shapes and dtypes match the checkpoint and every patch tensor differs from it; the
            process holds at least 95% of the weight bytes (the model is resident) and the machine keeps at
            least --min-free-gib available (Linux; reported as not measured elsewhere);
  template  the first three prompts of the prompt file render again to the same text and token ids with the
            checkpoint's tokenizer and template, and hold exactly one start token, the thinking line and the
            assistant header at the end;
  skyrl     with --skyrl-src: this port against MarinSkyRL's eager model in the checkpoint dtype on one prompt of
            at least 64 tokens (its first 64), top-1 equal at >= 63 of 64 positions;
  cached    KV-cached decode against a full forward on --long-len tokens of real text, top-1 agreement >= 98%;
  mixed     arm A inside a batch of arms A-D against arm A alone over --mixed-steps decode steps, top-1 agreement
            >= 95% (the largest difference of every other arm from A is reported);
  speed     sampled decode steps (the run's sampler included) timed at each --batches size; bandwidth and
            per-expert-call overhead fitted; projected duration of the main run (--per-arm prompts for each of
            --arms, cap --cap, waves of --batch-size rows) when arms B and D run away (80%) and when nothing does.
Exit codes: 0 all checks pass; 3 the load check fails (patch, shapes or memory); 1 another check fails; 4 the
checks pass but the projected run (B and D run away) exceeds --max-hours.

Usage: python -m swaptest.selfcheck --checkpoint DIR --patch FILE --prompts prompts.jsonl [--skyrl-src DIR]
       [--long-len 2600] [--prefill-len 2300] [--mixed-steps 32] [--batches 1,16,64,192] [--steps 20]
       [--arms ABCD] [--per-arm 48] [--cap 4096] [--batch-size 192] [--max-hours 8] [--min-free-gib 80]
       [--seed 9984] [--json out.json] [--threads N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from swaptest.chat import ChatTokenizer, load_tokenizer, render_user_turn
from swaptest.checks import (
    SpeedModel,
    cached_logits,
    compare_logits,
    full_logits,
    long_text_ids,
    mixed_vs_alone,
    time_decode,
)
from swaptest.loading import load_model, memory_gib, memory_report, weight_gib
from swaptest.patch import PHASE1_ARMS
from swaptest.prompts import check_prompt
from swaptest.records import read_jsonl, write_json_atomic
from swaptest.sampling import Sampler

CACHED_MIN_TOP1 = 0.98
MIXED_MIN_TOP1 = 0.95
SKYRL_TOKENS = 64
SKYRL_MIN_EQUAL = 63
MIN_RESIDENT_SHARE = 0.95
NORMAL_RATE, RUNAWAY_RATE = 0.03, 0.8
SCENARIOS = {
    "B and D run away (80%)": {"A": NORMAL_RATE, "B": RUNAWAY_RATE, "C": NORMAL_RATE, "D": RUNAWAY_RATE},
    "nothing runs away": dict.fromkeys(PHASE1_ARMS, NORMAL_RATE),
}


def memory_check(weights_gib: float, min_free_gib: float) -> tuple[bool, dict]:
    """The model is resident and enough memory stays available (always passes where it cannot be measured)."""
    m = memory_gib()
    if m is None:
        return True, {"weights_gib": weights_gib, "measured": False}
    ok = m["rss"] >= MIN_RESIDENT_SHARE * weights_gib and m["available"] >= min_free_gib
    return ok, {"weights_gib": weights_gib, "measured": True, **m, "min_free_gib": min_free_gib}


def main(argv: list[str] | None = None, tokenizer: ChatTokenizer | None = None) -> int:
    """`tokenizer` replaces the checkpoint's own (tests pass one that renders the template with jinja2)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--skyrl-src", type=Path)
    parser.add_argument("--long-len", type=int, default=2600)
    parser.add_argument("--prefill-len", type=int, default=2300)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--mixed-steps", type=int, default=32)
    parser.add_argument("--batches", default="1,16,64,192")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--arms", default=PHASE1_ARMS, help="arms of the main run, for the projection")
    parser.add_argument("--per-arm", type=int, default=48, help="prompts per arm in the main run")
    parser.add_argument("--cap", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=192, help="rows per wave in the main run")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=9984, help="sampling seed used while timing")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if not args.arms or set(args.arms) - set(PHASE1_ARMS) or len(set(args.arms)) != len(args.arms):
        parser.error(f"--arms must be distinct letters from {PHASE1_ARMS}")
    if args.threads:
        torch.set_num_threads(args.threads)
    print(
        f"torch {torch.__version__}, {torch.get_num_threads()} threads, "
        f"oneDNN available: {torch.backends.mkldnn.is_available()}, "
        f"CPU capability: {torch.backends.cpu.get_cpu_capability()}"
    )

    summary: dict = {}
    passed: dict[str, bool] = {}

    def report(name: str, ok: bool, detail: object) -> None:
        passed[name] = ok
        summary[name] = detail
        print(f"[{name}] {detail} -> {'PASS' if ok else 'FAIL'}")

    def finish(code: int) -> int:
        summary["passed"] = passed
        summary["exit_code"] = code
        if args.json:
            write_json_atomic(args.json, summary)
        return code

    try:
        loaded = load_model(args.checkpoint, args.patch, "ABCD", print)
    except ValueError as error:
        report("load", False, f"loading failed: {error}")
        print("self-check FAILED: load")
        return finish(3)
    ok, detail = memory_check(weight_gib(loaded.weights), args.min_free_gib)
    report("load", ok, {"patch": "groups, shapes and dtypes match; every tensor differs", "memory": detail})
    print(memory_report())
    if not ok:
        print("self-check FAILED: load")
        return finish(3)

    model = loaded.model
    tokenizer = tokenizer or load_tokenizer(args.checkpoint)
    prompts = read_jsonl(args.prompts)

    with torch.inference_mode():
        problems = []
        for p in prompts[:3]:
            text, ids = render_user_turn(tokenizer, p["user_text"])
            if text != p["prompt_text"] or ids != p["prompt_ids"]:
                problems.append(f"prompt {p['idx']}: renders differently now than in the prompt file")
            try:
                check_prompt(text, ids, tokenizer.bos_token_id)
            except ValueError as error:
                problems.append(f"prompt {p['idx']}: {error}")
        report("template", not problems, problems or "first three prompts render as expected")

        if args.skyrl_src is not None:
            from swaptest.skyrl_crosscheck import skyrl_logits

            prompt = next(p["prompt_ids"] for p in prompts if len(p["prompt_ids"]) >= SKYRL_TOKENS)
            ids = torch.tensor([prompt[:SKYRL_TOKENS]])
            config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
            ref, ours = skyrl_logits(args.skyrl_src, config, loaded.weights, ids), full_logits(model, ids)
            equal = int((ref.argmax(-1) == ours.argmax(-1)).sum())
            report(
                "skyrl",
                equal >= SKYRL_MIN_EQUAL,
                {"top1_equal": f"{equal}/{SKYRL_TOKENS}", **compare_logits(ref, ours)},
            )

        ids = torch.tensor([long_text_ids(tokenizer, prompts, args.long_len)])
        diff = compare_logits(full_logits(model, ids), cached_logits(model, ids, args.prefill_len, args.chunk))
        report("cached", diff["top1_agreement"] >= CACHED_MIN_TOP1 and not diff["nan"], diff)

        diff = mixed_vs_alone(model, prompts[0]["prompt_ids"], args.mixed_steps, "ABCD")
        report("mixed", diff["top1_agreement"] >= MIXED_MIN_TOP1 and not diff["nan"], diff)

        speed = SpeedModel(model)
        sampler = Sampler(args.seed, 1.0)
        timings = [
            time_decode(
                model, [p["prompt_ids"] for p in prompts], "ABCD", int(b), args.steps, args.chunk, sampler=sampler
            )
            for b in args.batches.split(",")
        ]
        for t in timings:
            print(
                f"    batch {t['batch']}: {t['step_s']:.3f} s per step, {t['tokens_per_s']:.1f} tokens/s; "
                f"prefill {t['prefill_tokens']} tokens in {t['prefill_s']:.1f} s"
            )
        speed.fit([(t["batch"], t["step_s"]) for t in timings])
        prefill_rate = max(t["prefill_tokens"] / t["prefill_s"] for t in timings)
        prompt_len = round(statistics.fmean(len(p["prompt_ids"]) for p in prompts))
        hours = {
            name: speed.project_hours(
                [rates[a] for a in args.arms],
                args.per_arm,
                args.cap,
                prompt_len,
                prefill_rate,
                batch_size=args.batch_size,
            )
            for name, rates in SCENARIOS.items()
        }
        report(
            "speed",
            True,
            {
                "timings": timings,
                "bandwidth_GBps": speed.bandwidth / 1e9,
                "per_expert_call_ms": speed.call_s * 1e3,
                "prefill_tokens_per_s": prefill_rate,
                "arms": args.arms,
                "per_arm": args.per_arm,
                "cap": args.cap,
                "batch_size": args.batch_size,
                "projected_hours": hours,
            },
        )

    summary["memory_after_checks"] = memory_report()
    failed = [name for name, ok in passed.items() if not ok]
    worst = hours[next(iter(SCENARIOS))]
    print("projected main run: " + ", ".join(f"{k} {v:.2f} h" for k, v in hours.items()))
    if failed:
        print(f"self-check FAILED: {', '.join(failed)}")
        return finish(1)
    if worst > args.max_hours:
        print(f"stop: the projected main run ({worst:.2f} h) exceeds {args.max_hours} h")
        return finish(4)
    print("self-check passed")
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
