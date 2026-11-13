"""Statistics and the pre-registered decision for a swap-test run.

Primary statistic per arm: R, the share of replies that reach the cap without a stop token (stop_reason "cap").
Each arm X is compared with the reference arm A by the rule in `swaptest.stats`. Validity gate V1: R_A <= 0.15
and arm A's GSM8K flexible accuracy >= 0.80; when V1 fails no arm is interpreted.

Secondary statistics: length quartiles (replies at the cap are censored), stop tokens, whether `<|end_think|>`
was emitted, GSM8K flexible accuracy, repetition in the last 1,024 tokens (distinct 4-gram ratio and zlib ratio of
the token ids), the first position where each arm leaves the reference arm's tokens, and the teacher-forced
readout when given.

With --interim, pairs not finished yet count as pending: an arm is called "no effect" early only if the call
holds when every pending reply of that arm reaches the cap and every pending reference reply stops.

Usage: python -m swaptest.analyze --prompts prompts.jsonl [more.jsonl] --replies RUN/replies.jsonl [more]
       [--teacher-forced tf.jsonl] [--reference A] [--limit N] [--interim] [--min-stopped K] [--json out.json]
       [--show-replies ARMS] [--arms ARMS]
With several prompt and reply files (an extension drawn later), all prompts count; --arms restricts the analysis
to the reference arm and the arms that were extended.
Exit codes: 0 done; 2 pairs missing (without --interim); 3 fewer than --min-stopped reference replies stopped.
The exit code does not depend on V1 or on the decisions; read them from the report or the JSON summary.
"""

from __future__ import annotations

import argparse
import statistics
import struct
import sys
import zlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swaptest.gsm8k import extract_answer, is_correct
from swaptest.patch import ARM_ORDER
from swaptest.records import read_jsonl, write_json_atomic
from swaptest.stats import compare

V1_MAX_RATE = 0.15
V1_MIN_ACCURACY = 0.80
END_THINK = "<|end_think|>"
TAIL = 1024


def repetition(ids: Sequence[int]) -> tuple[float, float]:
    """Distinct 4-gram ratio and zlib compression ratio of the last 1,024 token ids (lower = more repetitive)."""
    tail = list(ids[-TAIL:])
    grams = [tuple(tail[i : i + 4]) for i in range(len(tail) - 3)]
    distinct = len(set(grams)) / len(grams) if grams else 1.0
    raw = struct.pack(f"<{len(tail)}i", *tail)
    return distinct, len(zlib.compress(raw, 9)) / len(raw) if raw else 1.0


def first_divergence(ids: Sequence[int], ref: Sequence[int]) -> int | None:
    """First position where two replies differ (a prefix differs at its end); None if identical."""
    for i, (a, b) in enumerate(zip(ids, ref, strict=False)):
        if a != b:
            return i
    return None if len(ids) == len(ref) else min(len(ids), len(ref))


def quartiles(lengths: list[int], censored_from: int | None) -> list[str]:
    """Quartiles of reply lengths; a value at or above the shortest capped reply is shown as censored."""
    q = statistics.quantiles(lengths, n=4, method="inclusive") if len(lengths) > 1 else lengths * 3
    return [f">={v:.0f}" if censored_from is not None and v >= censored_from else f"{v:.0f}" for v in q]


def load_replies(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    replies: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for r in read_jsonl(path):
            key = (r["arm"], r["prompt_id"])
            if key in replies and replies[key]["token_ids"] != r["token_ids"]:
                raise SystemExit(f"two different replies for arm {key[0]}, prompt {key[1]}")
            replies[key] = r
    return replies


def arm_summary(rows: list[dict[str, Any]], gold: dict[int, str], ref: dict[int, dict[str, Any]]) -> dict[str, Any]:
    capped = [r for r in rows if r["stop_reason"] == "cap"]
    lengths = sorted(r["n_new"] for r in rows)
    rep_all = [repetition(r["token_ids"]) for r in rows]
    rep_cap = [repetition(r["token_ids"]) for r in capped]
    divergence = [
        first_divergence(r["token_ids"], ref[r["prompt_id"]]["token_ids"]) for r in rows if r["prompt_id"] in ref
    ]
    diverged = [d for d in divergence if d is not None]
    return {
        "n": len(rows),
        "capped": len(capped),
        "rate": len(capped) / len(rows),
        "accuracy": sum(is_correct(r["text"], gold[r["prompt_id"]]) for r in rows) / len(rows),
        "stop_reasons": dict(Counter(str(r["stop_reason"]) for r in rows)),
        "end_think": sum(END_THINK in r["text"] for r in rows),
        "length_quartiles": quartiles(lengths, min((r["n_new"] for r in capped), default=None)),
        "distinct4_median": statistics.median(d for d, _ in rep_all),
        "zlib_median": statistics.median(z for _, z in rep_all),
        "distinct4_median_capped": statistics.median(d for d, _ in rep_cap) if rep_cap else None,
        "zlib_median_capped": statistics.median(z for _, z in rep_cap) if rep_cap else None,
        "identical_to_reference": divergence.count(None),
        "first_divergence_median": statistics.median(diverged) if diverged else None,
    }


def teacher_forced_summary(rows: list[dict[str, Any]], reference: str) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    ref_stop = {r["prompt_id"]: r["stop_logprob"] for r in by_arm.get(reference, []) if r["stop_logprob"] is not None}
    out = {}
    for arm, items in sorted(by_arm.items(), key=lambda kv: ARM_ORDER.index(kv[0])):
        stopped = [r for r in items if r["stop_logprob"] is not None]
        paired = [r["stop_logprob"] - ref_stop[r["prompt_id"]] for r in stopped if r["prompt_id"] in ref_stop]
        out[arm] = {
            "prompts": len(items),
            "nll_mean": statistics.fmean(r["nll_mean"] for r in items),
            "stop_logprob_median": statistics.median(r["stop_logprob"] for r in stopped) if stopped else None,
            "any_stop_logprob_median": statistics.median(r["any_stop_logprob"] for r in stopped) if stopped else None,
            "stop_logprob_minus_reference_mean": statistics.fmean(paired) if paired else None,
        }
    return out


def analyze(
    prompts: list[dict[str, Any]],
    replies: dict[tuple[str, int], dict[str, Any]],
    reference: str,
    interim: bool,
    teacher_forced: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    gold = {p["idx"]: p["gold"] for p in prompts}
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in replies)]
    if reference not in arms:
        raise SystemExit(f"no replies of the reference arm {reference}")
    missing = {a: [p["idx"] for p in prompts if (a, p["idx"]) not in replies] for a in arms}
    rows = {a: [replies[(a, p["idx"])] for p in prompts if (a, p["idx"]) in replies] for a in arms}
    ref_rows = {r["prompt_id"]: r for r in rows[reference]}
    summary: dict[str, Any] = {
        "reference": reference,
        "prompts": len(prompts),
        "missing": {a: len(m) for a, m in missing.items()},
        "arms": {a: arm_summary(rows[a], gold, ref_rows) for a in arms if rows[a]},
    }
    ref = summary["arms"][reference]
    summary["v1"] = {
        "rate": ref["rate"],
        "accuracy": ref["accuracy"],
        "passed": ref["rate"] <= V1_MAX_RATE and ref["accuracy"] >= V1_MIN_ACCURACY,
    }
    comparisons = {}
    for arm in arms:
        if arm == reference or arm not in summary["arms"]:
            continue
        s = summary["arms"][arm]
        c = compare(s["capped"], s["n"], ref["capped"], ref["n"])
        entry = {"diff": c.diff, "p_value": c.p_value, "newcombe95": [c.lower, c.upper], "outcome": c.outcome}
        if interim:
            worst = compare(
                s["capped"] + len(missing[arm]),
                s["n"] + len(missing[arm]),
                ref["capped"],
                ref["n"] + len(missing[reference]),
            )
            entry["early_no_effect"] = worst.outcome == "no effect"
        if not summary["v1"]["passed"]:
            entry["outcome"] = "not interpreted (V1 failed)"
        comparisons[arm] = entry
    summary["comparisons"] = comparisons
    if teacher_forced is not None:
        summary["teacher_forced"] = teacher_forced_summary(teacher_forced, reference)
    return summary


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def print_report(s: dict[str, Any]) -> None:
    print(f"reference arm {s['reference']}; {s['prompts']} prompts; missing pairs {s['missing']}")
    print("\narm  n   capped  R      accuracy  </think>  length quartiles (censored >=)  stop reasons")
    for arm, a in s["arms"].items():
        print(
            f"{arm:4s} {a['n']:<3d} {a['capped']:<7d} {a['rate']:.3f}  {a['accuracy']:.3f}     "
            f"{a['end_think']:<9d} {' / '.join(a['length_quartiles']):31s} {a['stop_reasons']}"
        )
    print("\narm  distinct-4 (all / capped)  zlib (all / capped)  identical to ref  first divergence (median)")
    for arm, a in s["arms"].items():
        print(
            f"{arm:4s} {_fmt(a['distinct4_median'])} / {_fmt(a['distinct4_median_capped']):9s}       "
            f"{_fmt(a['zlib_median'])} / {_fmt(a['zlib_median_capped']):9s}  {a['identical_to_reference']:<16d} "
            f"{_fmt(a['first_divergence_median'], '.0f')}"
        )
    v1 = s["v1"]
    print(
        f"\nV1: R_{s['reference']} = {v1['rate']:.3f} (<= {V1_MAX_RATE}), accuracy = {v1['accuracy']:.3f} "
        f"(>= {V1_MIN_ACCURACY}): {'passed' if v1['passed'] else 'FAILED - no arm is interpreted'}"
    )
    print(f"\narm vs {s['reference']}  R diff   one-sided Fisher p  Newcombe 95%       decision")
    for arm, c in s["comparisons"].items():
        early = "" if "early_no_effect" not in c else f"  (early no-effect call allowed: {c['early_no_effect']})"
        print(
            f"{arm:10s} {c['diff']:+.3f}   {c['p_value']:<19.3g} [{c['newcombe95'][0]:+.3f}, "
            f"{c['newcombe95'][1]:+.3f}]  {c['outcome']}{early}"
        )
    if "teacher_forced" in s:
        print(f"\nteacher-forced on arm {s['reference']}'s replies")
        print("arm  prompts  mean NLL  median stop logp  median any-stop logp  mean stop logp - reference")
        for arm, t in s["teacher_forced"].items():
            print(
                f"{arm:4s} {t['prompts']:<8d} {t['nll_mean']:.4f}    {_fmt(t['stop_logprob_median']):17s} "
                f"{_fmt(t['any_stop_logprob_median']):21s} {_fmt(t['stop_logprob_minus_reference_mean'], '+.3f')}"
            )


def print_replies(
    prompts: list[dict[str, Any]], replies: dict[tuple[str, int], dict[str, Any]], arms: str, tail: int = 600
) -> None:
    """For reading replies by eye: per (prompt, arm) the stop reason, length, extracted and reference answer, and
    the last `tail` characters of the text."""
    for p in prompts:
        for arm in arms:
            r = replies.get((arm, p["idx"]))
            if r is None:
                continue
            answer = extract_answer(r["text"])
            print(
                f"\n=== prompt {p['idx']}, arm {arm}: stop {r['stop_reason']}, {r['n_new']} tokens, answer "
                f"{answer!r}, reference {p['gold']!r}, {'correct' if is_correct(r['text'], p['gold']) else 'wrong'}"
            )
            if p.get("question"):
                print(f"question: {p['question']}")
            print(f"reply (last {tail} characters): {r['text'][-tail:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--replies", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-forced", type=Path)
    parser.add_argument("--reference", default="A")
    parser.add_argument("--limit", type=int, help="use only the first N prompts")
    parser.add_argument("--interim", action="store_true", help="unfinished pairs count as pending")
    parser.add_argument("--min-stopped", type=int, help="exit 3 if fewer reference replies stopped")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    parser.add_argument("--show-replies", metavar="ARMS", help="also print the replies of these arms for reading")
    parser.add_argument("--arms", help="analyse only these arms (default: every arm with replies)")
    args = parser.parse_args(argv)
    if args.arms is not None and args.reference not in args.arms:
        parser.error("--arms must include the reference arm")

    prompts = [p for path in args.prompts for p in read_jsonl(path)][: args.limit]
    replies = load_replies(args.replies)
    tf = read_jsonl(args.teacher_forced) if args.teacher_forced else None
    if args.arms is not None:
        replies = {k: r for k, r in replies.items() if k[0] in args.arms}
        tf = None if tf is None else [r for r in tf if r["arm"] in args.arms]
    summary = analyze(prompts, replies, args.reference, args.interim, tf)
    print_report(summary)
    if args.show_replies:
        print_replies(prompts, replies, args.show_replies)
    if args.json:
        write_json_atomic(args.json, summary)
    if any(summary["missing"].values()) and not args.interim:
        print("pairs are missing; rerun with --interim for an interim look", file=sys.stderr)
        return 2
    ref = summary["arms"][args.reference]
    if args.min_stopped is not None and ref["n"] - ref["capped"] < args.min_stopped:
        print(f"only {ref['n'] - ref['capped']} reference replies stopped (need {args.min_stopped})", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
