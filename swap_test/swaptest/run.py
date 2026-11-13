"""Generate one reply for every (arm, prompt) pair and append it to a JSONL file; resumable.

Usage:
  python -m swaptest.run --checkpoint DIR [--patch FILE | --merge SPEC.json] --prompts prompts.jsonl --out RUN_DIR
      [--arms ABCD] [--cap 4096] [--batch-size 192] [--seed 9984] [--temperature 1.0] [--limit N]
      [--threads N] [--prefill-chunk 256] [--prefill-rows 16] [--log-every 100] [--snapshot-every 50]

Files in RUN_DIR:
  replies.jsonl    one line per finished pair: arm, prompt_id, seed, temperature, prompt_tokens, n_new (generated
                   tokens, a final stop token included), stop_reason (the stop token id, or "cap"), token_ids,
                   text (decoded with special tokens kept, final stop token excluded) and wall_s
  inflight.json    tokens generated so far by the rows of the running wave (rewritten every --snapshot-every steps)
  run_config.json  the settings a restart must repeat
  run.log          progress, with tokens per second

Pairs are decoded in waves of at most --batch-size rows, prompt by prompt with the arms of one prompt next to each
other and always in the same wave. Restarted with the same RUN_DIR, the run skips finished pairs and continues
rows in flight from their snapshot: their prefix is prefilled again and decoding resumes at the next position with
the same noise. Stop tokens come from the checkpoint's generation_config.json.

Without --merge the weights are the checkpoint's and the arms are A to D (the patch supplies B, C and D). With
--merge the weights are the merge of the spec file, built in memory (`swaptest.merge`), the arms are those the spec
defines (P, N, G in phase 2), and the checkpoint directory supplies only the configuration, the tokenizer, the chat
template and the stop tokens.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from swaptest.chat import ChatTokenizer, decode, load_tokenizer
from swaptest.engine import Finished, Row, generate
from swaptest.loading import load_merged_model, load_model
from swaptest.patch import ARM_ORDER, PHASE1_ARMS
from swaptest.records import append_jsonl, drop_torn_tail, read_json, read_jsonl, sha256_file, write_json_atomic
from swaptest.sampling import Sampler

REPLIES, INFLIGHT, CONFIG, LOG = "replies.jsonl", "inflight.json", "run_config.json", "run.log"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch", type=Path, help="patch file; needed by arms B, C and D")
    parser.add_argument("--merge", type=Path, help="spec file of a merge built in memory (arms P, N, G)")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=PHASE1_ARMS)
    parser.add_argument("--cap", type=int, default=4096, help="maximum new tokens per reply")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=9984)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 means greedy")
    parser.add_argument("--limit", type=int, help="use only the first N prompts")
    parser.add_argument("--threads", type=int, help="torch threads (default: torch's choice)")
    parser.add_argument("--prefill-chunk", type=int, default=256)
    parser.add_argument("--prefill-rows", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=50)
    args = parser.parse_args(argv)
    allowed = ARM_ORDER if args.merge else PHASE1_ARMS
    if not args.arms or len(set(args.arms)) != len(args.arms) or set(args.arms) - set(allowed):
        parser.error(f"--arms must be distinct letters from {allowed}")
    if args.merge and args.patch:
        parser.error("--patch and --merge exclude each other")
    return args


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("swaptest.run")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old in list(logger.handlers):
        old.close()
        logger.removeHandler(old)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = time.gmtime  # UTC, as in the scripts' logs
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def settings_of(args: argparse.Namespace, n_prompts: int) -> dict:
    return {
        "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
        "patch_sha256": sha256_file(args.patch) if args.patch else None,
        "merge_spec_sha256": sha256_file(args.merge) if args.merge else None,
        "prompts_sha256": sha256_file(args.prompts),
        "prompts_used": n_prompts,
        "arms": args.arms,
        "cap": args.cap,
        "seed": args.seed,
        "temperature": args.temperature,
    }


def pending_rows(prompts: list[dict], arms: str, done: set, inflight: dict) -> list[Row]:
    """Unfinished pairs, prompt by prompt with arms adjacent; prompts with saved progress first."""
    rows = []
    for prompt in prompts:
        for arm in arms:
            key = (arm, prompt["idx"])
            if key not in done:
                saved = inflight.get(key, {})
                rows.append(
                    Row(
                        arm,
                        prompt["idx"],
                        prompt["prompt_ids"],
                        list(saved.get("generated", [])),
                        saved.get("elapsed", 0.0),
                    )
                )
    resumed = {prompt_id for _, prompt_id in inflight}
    return sorted(rows, key=lambda r: r.prompt_id not in resumed)  # stable: the arms of a prompt stay adjacent


def make_waves(rows: list[Row], batch_size: int) -> list[list[Row]]:
    """Consecutive waves of at most `batch_size` rows that never split the rows of one prompt between waves, so
    the arms of a prompt always share a batch (a prompt is split only if it alone has more rows than a wave)."""
    waves: list[list[Row]] = []
    current: list[Row] = []
    for _, grouped in itertools.groupby(rows, key=lambda r: r.prompt_id):
        group = list(grouped)
        if current and len(current) + len(group) > batch_size:
            waves.append(current)
            current = []
        while len(group) > batch_size:
            waves.append(group[:batch_size])
            group = group[batch_size:]
        current += group
    if current:
        waves.append(current)
    return waves


def run(
    args: argparse.Namespace, tokenizer: ChatTokenizer | None = None, step_hook: Callable[[int], None] | None = None
) -> None:
    """Generate all pending pairs. `step_hook(step)` runs after every decode step (tests use it to interrupt)."""
    args.out.mkdir(parents=True, exist_ok=True)
    log = make_logger(args.out / LOG)
    prompts = read_jsonl(args.prompts)[: args.limit]
    settings = settings_of(args, len(prompts))
    config_path = args.out / CONFIG
    if config_path.exists():
        saved = read_json(config_path)
        if saved != settings:
            changed = sorted(k for k in settings if saved.get(k) != settings[k])
            raise SystemExit(f"{args.out} holds a run with other settings ({', '.join(changed)}); use a new --out")
    else:
        write_json_atomic(config_path, settings)

    replies_path, inflight_path = args.out / REPLIES, args.out / INFLIGHT
    drop_torn_tail(replies_path)
    done = {(r["arm"], r["prompt_id"]) for r in read_jsonl(replies_path)}
    inflight = {}
    if inflight_path.exists():
        for r in read_json(inflight_path)["rows"]:
            inflight[(r["arm"], r["prompt_id"])] = r
    rows = pending_rows(prompts, args.arms, done, inflight)
    log.info(f"{len(done)} pairs done, {len(rows)} to generate ({sum(r.key in inflight for r in rows)} resumed)")
    if not rows:
        inflight_path.unlink(missing_ok=True)
        return

    if args.threads:
        torch.set_num_threads(args.threads)
    log.info(f"torch {torch.__version__}, {torch.get_num_threads()} threads, batch size {args.batch_size}")
    if args.batch_size < len(args.arms):
        log.info(
            f"warning: batch size {args.batch_size} is smaller than the {len(args.arms)} arms, so the arms of a "
            "prompt fall into different waves (different batches change low-order bits)"
        )
    if args.merge:
        loaded = load_merged_model(args.checkpoint, args.merge, args.arms, log.info)
    else:
        loaded = load_model(args.checkpoint, args.patch, args.arms, log.info)
    tokenizer = tokenizer or load_tokenizer(args.checkpoint)
    sampler = Sampler(args.seed, args.temperature)
    log.info(f"stop tokens {loaded.stop_ids}, cap {args.cap}, temperature {args.temperature}, seed {args.seed}")

    def on_finish(f: Finished) -> None:
        ids = f.row.generated
        body = ids if f.stop_reason == "cap" else ids[:-1]
        append_jsonl(
            replies_path,
            {
                "arm": f.row.arm,
                "prompt_id": f.row.prompt_id,
                "seed": args.seed,
                "temperature": args.temperature,
                "prompt_tokens": len(f.row.prompt),
                "n_new": len(ids),
                "stop_reason": f.stop_reason,
                "token_ids": ids,
                "text": decode(tokenizer, body),
                "wall_s": round(f.wall_s, 3),
            },
        )

    waves = make_waves(rows, args.batch_size)
    n_waves = len(waves)
    with torch.inference_mode():
        for w, wave in enumerate(waves):
            base = sum(len(r.generated) for r in wave)
            mark = {"t": time.monotonic(), "tokens": 0}
            log.info(
                f"wave {w + 1}/{n_waves}: {len(wave)} rows, longest prefix "
                f"{max(len(r.prompt) + len(r.generated) for r in wave)} tokens"
            )

            def on_step(step: int, active: list[Row], seconds: float, wave=wave, base=base, mark=mark) -> None:
                if step % args.snapshot_every == 0 or not active:
                    write_json_atomic(
                        inflight_path,
                        {
                            "rows": [
                                {
                                    "arm": r.arm,
                                    "prompt_id": r.prompt_id,
                                    "generated": r.generated,
                                    "elapsed": round(r.elapsed + seconds, 3),
                                }
                                for r in active
                            ]
                        },
                    )
                if step % args.log_every == 0 or not active:
                    tokens = sum(len(r.generated) for r in wave) - base
                    now = time.monotonic()
                    rate = (tokens - mark["tokens"]) / max(now - mark["t"], 1e-9)
                    log.info(
                        f"step {step}: {len(active)} rows active, {tokens} tokens this wave, "
                        f"{rate:.1f} tok/s (last interval), {tokens / max(seconds, 1e-9):.1f} tok/s (wave), "
                        f"{seconds / 60:.1f} min"
                    )
                    mark.update(t=now, tokens=tokens)
                if step_hook is not None:
                    step_hook(step)

            generate(
                loaded.model,
                wave,
                sampler,
                args.cap,
                loaded.stop_ids,
                on_finish,
                on_step,
                chunk=args.prefill_chunk,
                row_block=args.prefill_rows,
            )
    inflight_path.unlink(missing_ok=True)
    log.info(f"done: {len(read_jsonl(replies_path))} replies in {replies_path}")


def main(argv: list[str] | None = None, tokenizer: ChatTokenizer | None = None) -> None:
    run(parse_args(argv), tokenizer)


if __name__ == "__main__":
    main()
