"""Build the phase-2 merge in memory with its arms and check the harness on it before the long runs.

Checks, each printed with PASS or FAIL (a JSON summary is written with --json):
  load      the merge of the spec builds, every arm override matches the merged tensor's shape and dtype and
            differs from it, the process holds at least 95% of the weight bytes and the machine keeps at least
            --min-free-gib available (Linux; reported as not measured elsewhere);
  kv        the key/value caches of the planned waves fit: the runaway run (--batch-size rows of the longest
            GSM8K prompt plus --cap) and a capability run (--task-batch-size rows of the longest task prompt plus
            --task-cap) each leave at least --kv-headroom-gib of the available memory (Linux);
  template  the first three GSM8K prompts and the first three task prompts render again to the same text and token
            ids, with one start token, the thinking line and the assistant header;
  cached    KV-cached decode of arm P against a full forward on --long-len tokens of real text, top-1 >= 99%;
  mixed     arm P inside a batch of all --arms against arm P alone over --mixed-steps steps, top-1 >= 99%;
  speed     sampled decode steps timed at each --batches size; bandwidth and per-expert-call overhead fitted;
            projected hours of the runaway run (--per-arm GSM8K prompts for each of --arms, cap --cap, waves of
            --batch-size rows) when P runs away (80%) and when nothing does, and of a capability run of one arm
            and of two arms over the task prompts in file order (waves of --task-batch-size rows; replies ASSUMED
            log-normal with median --task-median tokens and sigma --task-sigma, capped at --task-cap).
Exit codes: 0 all checks pass; 3 the load check fails; 1 another check fails; 4 the checks pass but the projected
runaway run (P runs away) exceeds --max-hours.

Usage: python -m swaptest.selfcheck_phase2 --checkpoint DIR --merge SPEC.json --prompts prompts.jsonl
       --task-prompts tasks.jsonl [--arms PNG] [--long-len 2600] [--prefill-len 2300] [--mixed-steps 32]
       [--batches 1,16,64,144] [--steps 20] [--per-arm 48] [--cap 4096] [--batch-size 144] [--task-cap 4096]
       [--task-batch-size 384] [--task-median 1000] [--task-sigma 0.8] [--max-hours 8] [--min-free-gib 60]
       [--kv-headroom-gib 16] [--seed 9984] [--json out.json] [--threads N]
"""

from __future__ import annotations

import argparse
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
from swaptest.loading import load_merged_model, memory_gib, memory_report, weight_gib
from swaptest.patch import arm_index
from swaptest.prompts import check_prompt
from swaptest.records import read_jsonl, write_json_atomic
from swaptest.sampling import Sampler
from swaptest.selfcheck import CACHED_MIN_TOP1, MIXED_MIN_TOP1, NORMAL_RATE, RUNAWAY_RATE, memory_check

PHASE2_ARMS = "PNG"


def main(argv: list[str] | None = None, tokenizer: ChatTokenizer | None = None) -> int:
    """`tokenizer` replaces the checkpoint's own (tests pass one that renders the template with jinja2)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="configuration, tokenizer and template")
    parser.add_argument("--merge", type=Path, required=True, help="spec file of the merge")
    parser.add_argument("--prompts", type=Path, required=True, help="GSM8K prompt file of the runaway run")
    parser.add_argument("--task-prompts", type=Path, required=True)
    parser.add_argument("--arms", default=PHASE2_ARMS)
    parser.add_argument("--long-len", type=int, default=2600)
    parser.add_argument("--prefill-len", type=int, default=2300)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--mixed-steps", type=int, default=32)
    parser.add_argument("--batches", default="1,16,64,144")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--per-arm", type=int, default=48)
    parser.add_argument("--cap", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--task-cap", type=int, default=4096)
    parser.add_argument("--task-batch-size", type=int, default=384)
    parser.add_argument("--task-median", type=float, default=1000.0, help="ASSUMED median task reply length")
    parser.add_argument("--task-sigma", type=float, default=0.8, help="ASSUMED sigma of log task reply length")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--min-free-gib", type=float, default=60.0)
    parser.add_argument("--kv-headroom-gib", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=9984)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if "P" not in args.arms or set(args.arms) - set(PHASE2_ARMS) or len(set(args.arms)) != len(args.arms):
        parser.error(f"--arms must be distinct letters from {PHASE2_ARMS} including P")
    if args.threads:
        torch.set_num_threads(args.threads)
    print(
        f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU {torch.backends.cpu.get_cpu_capability()}"
    )

    summary: dict = {}
    passed: dict[str, bool] = {}

    def report(name: str, ok: bool, detail: object) -> None:
        passed[name] = ok
        summary[name] = detail
        print(f"[{name}] {detail} -> {'PASS' if ok else 'FAIL'}", flush=True)

    def finish(code: int) -> int:
        summary["passed"] = passed
        summary["exit_code"] = code
        if args.json:
            write_json_atomic(args.json, summary)
        return code

    try:
        loaded = load_merged_model(args.checkpoint, args.merge, args.arms, lambda m: print(m, flush=True))
    except (ValueError, OSError) as error:
        report("load", False, f"building the merge or its arms failed: {error}")
        return finish(3)
    ok, detail = memory_check(weight_gib(loaded.weights), args.min_free_gib)
    report("load", ok, {"arms": args.arms, "spec_sha256": loaded.spec_sha256, "memory": detail})
    if not ok:
        return finish(3)

    model = loaded.model
    speed = SpeedModel(model)
    prompts, tasks = read_jsonl(args.prompts), read_jsonl(args.task_prompts)
    plans = {
        "runaway": (args.batch_size, max(len(p["prompt_ids"]) for p in prompts) + args.cap),
        "capability": (args.task_batch_size, max(len(p["prompt_ids"]) for p in tasks) + args.task_cap),
    }
    kv = {name: rows * speed.kv_bytes(length) / 2**30 for name, (rows, length) in plans.items()}
    memory = memory_gib()
    if memory is None:
        report("kv", True, {"planned_gib": kv, "measured": False})
    else:
        spare = {name: memory["available"] - gib for name, gib in kv.items()}
        report("kv", min(spare.values()) >= args.kv_headroom_gib, {"planned_gib": kv, "spare_gib": spare})

    tokenizer = tokenizer or load_tokenizer(args.checkpoint)
    with torch.inference_mode():
        problems = []
        for p in prompts[:3] + tasks[:3]:
            text, ids = render_user_turn(tokenizer, p["user_text"])
            if text != p["prompt_text"] or ids != p["prompt_ids"]:
                problems.append(f"prompt {p['idx']}: renders differently now than in its prompt file")
            try:
                check_prompt(text, ids, tokenizer.bos_token_id)
            except ValueError as error:
                problems.append(f"prompt {p['idx']}: {error}")
        report("template", not problems, problems or "prompts render as expected")

        ids = torch.tensor([long_text_ids(tokenizer, prompts, args.long_len)])
        arm_p = torch.tensor([arm_index("P")])
        full = full_logits(model, ids, arm_p)
        diff = compare_logits(full, cached_logits(model, ids, args.prefill_len, args.chunk, arm_p))
        report("cached", diff["top1_agreement"] >= CACHED_MIN_TOP1 and not diff["nan"], diff)

        diff = mixed_vs_alone(model, prompts[0]["prompt_ids"], args.mixed_steps, args.arms, reference="P")
        report("mixed", diff["top1_agreement"] >= MIXED_MIN_TOP1 and not diff["nan"], diff)

        sampler = Sampler(args.seed, 1.0)
        timings = [
            time_decode(
                model, [p["prompt_ids"] for p in prompts], args.arms, int(b), args.steps, args.chunk, sampler=sampler
            )
            for b in args.batches.split(",")
        ]
        for t in timings:
            print(f"    batch {t['batch']}: {t['step_s']:.3f} s per step, {t['tokens_per_s']:.1f} tokens/s")
        speed.fit([(t["batch"], t["step_s"]) for t in timings])
        prefill_rate = max(t["prefill_tokens"] / t["prefill_s"] for t in timings)
        prompt_len = round(statistics.fmean(len(p["prompt_ids"]) for p in prompts))
        scenarios = {
            "P_runs_away": [RUNAWAY_RATE if a == "P" else NORMAL_RATE for a in args.arms],
            "nothing_runs_away": [NORMAL_RATE] * len(args.arms),
        }
        hours = {
            name: speed.project_hours(rates, args.per_arm, args.cap, prompt_len, prefill_rate, args.batch_size)
            for name, rates in scenarios.items()
        }
        task_lens = [len(p["prompt_ids"]) for p in tasks]
        task_hours = {
            f"{n}_arm{'s' if n > 1 else ''}": speed.project_hours(
                [NORMAL_RATE] * n,
                len(task_lens),
                args.task_cap,
                task_lens,
                prefill_rate,
                args.task_batch_size,
                median=args.task_median,
                sigma=args.task_sigma,
            )
            for n in (1, 2)
        }
        report(
            "speed",
            True,
            {
                "timings": timings,
                "bandwidth_GBps": speed.bandwidth / 1e9,
                "per_expert_call_ms": speed.call_s * 1e3,
                "prefill_tokens_per_s": prefill_rate,
                "projected_hours": hours,
                "projected_task_hours": task_hours,
                "task_reply_model": {"median": args.task_median, "sigma": args.task_sigma, "assumed": True},
            },
        )

    summary["memory_after_checks"] = memory_report()
    failed = [name for name, ok in passed.items() if not ok]
    worst = hours[next(iter(scenarios))]
    print("projected runaway run: " + ", ".join(f"{k} {v:.2f} h" for k, v in hours.items()))
    print("projected capability run: " + ", ".join(f"{k} {v:.2f} h" for k, v in task_hours.items()))
    if failed:
        print(f"self-check FAILED: {', '.join(failed)}")
        return finish(1)
    if worst > args.max_hours:
        print(f"stop: the projected runaway run ({worst:.2f} h) exceeds {args.max_hours} h")
        return finish(4)
    print("self-check passed")
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
