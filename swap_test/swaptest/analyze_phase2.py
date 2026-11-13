"""Phase-2 statistics and the pre-registered decisions: the runaway test (necessity) and the capability test.

`runaway` reads arm A's replies (Merge-TA0.5, phase 1) and those of arms P, N and G (phase 2) on the same GSM8K
prompts and noise:
  - V1 as in phase 1: R_A <= 0.15 and arm A's GSM8K accuracy >= 0.80.
  - V2: R_P >= 0.50 passes; R_P < 0.30 fails (the harness does not reproduce the failure); in between "partial".
  - every arm against A with the phase-1 rule (`swaptest.stats.compare`), and P against N and against G with the
    same rule, where "effect" means that P runs away more often than the arm.
  - necessity, for N and G, tested only when V1 and V2 pass and P shows an effect against A: "necessary" when the
    arm shows no effect against A and P an effect against the arm; "not_necessary" when the arm shows an effect
    against A and P no effect against the arm; "partial" when P shows an effect against the arm but the arm is
    not "no effect" against A; otherwise "inconclusive".
  - eligibility for capability scoring (V1 passed, R_X <= 0.15 and no effect of X against A): the candidate is N
    if eligible, else G if eligible, else none; P is scored as well when eligible. `task_arms` lists the arms to
    score, candidate first.

`capability` grades the replies of arm A and of the scored arms on the task prompts (`swaptest.taskprompts`) and
compares each arm with A on the same prompts:
  - per target (CRUXEval input, CRUXEval output, MMLU-Pro) the paired accuracy difference, its 95% interval from a
    bootstrap that resamples the prompt groups (CRUXEval problems, shared by both directions; MMLU-Pro question
    texts) within each domain, and the exact two-sided McNemar p;
  - the composite D = 0.25 d(CRUXEval input) + 0.25 d(CRUXEval output) + 0.5 d(MMLU-Pro), each expert's domain
    weighted equally, with its interval at --level (pre-registered 0.95) from the same bootstrap draws;
  - V3 on arm A: on every target at most 10 points below #9984's Merge-TA0.5 score (CRUXEval 76.5 / 67.6;
    MMLU-Pro 61.83 on all documents, 63.2 on a listed subset), and at most 10% of its task replies at the cap;
  - stopping normally: eligible in the runaway decisions and the upper end of the Newcombe 95% interval of the
    difference in capped shares (arm minus A, all task prompts) below 0.05;
  - decision: "better" when V3 passes, the arm stops normally, the lower end of D's interval is above 0 and no
    target's 95% interval lies wholly below 0; "trade_off" when D's lower end is above 0 but a target's interval
    lies below 0; "worse" when D's upper end is below 0; otherwise "not_better"; "not_interpreted_v3" or
    "not_interpreted_stopping" when those conditions fail. `extend_X` says whether D's upper end reaches +0.02.

`quicklook` grades arm A's replies to a few prompts of each task and prints them for reading; it fails (exit code
3) when more than --max-capped replies reach the cap, the median length exceeds --max-median tokens, or a task has
fewer than --min-correct correct replies (a sign of a broken template, extraction or grader).

Outputs: a report on stdout, --json (the full summary) and --decisions (one "key value" line per decision, read
by the run script). Grades are kept in --grades, so a rerun does not execute the CRUXEval programs again. Exit
codes: 0 done; 2 pairs missing; 3 quick look outside its limits.

Usage: python -m swaptest.analyze_phase2 runaway --prompts PROMPTS --replies FILE [FILE ...]
       [--teacher-forced FILE] [--json OUT] [--decisions OUT]
       python -m swaptest.analyze_phase2 capability --prompts TASK_PROMPTS --replies FILE [FILE ...]
       --runaway RUNAWAY.json --grades GRADES.jsonl [--json OUT] [--decisions OUT] [--level 0.95]
       [--draws 10000] [--seed 9984] [--workers 8] [--timeout 10]
       python -m swaptest.analyze_phase2 quicklook --prompts QUICK_PROMPTS --replies FILE --grades GRADES.jsonl
       [--json OUT] [--max-capped 2] [--max-median 2000] [--min-correct 2]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from swaptest import cruxeval, mmlu_pro
from swaptest.analyze import analyze, load_replies, print_report
from swaptest.answers import answer_text
from swaptest.records import append_jsonl_many, read_json, read_jsonl, write_json_atomic
from swaptest.stats import compare, newcombe
from swaptest.taskprompts import TASKS

RUNAWAY_ARMS = "APNG"
V2_PASS, V2_FAIL = 0.50, 0.30
ELIGIBLE_MAX_RATE = 0.15
WEIGHTS = {"cruxeval_input": 0.25, "cruxeval_output": 0.25, "mmlu_pro": 0.5}
STRATUM = {"cruxeval_input": "cruxeval", "cruxeval_output": "cruxeval", "mmlu_pro": "mmlu_pro"}
REFERENCE_9984 = {"cruxeval_input": 0.765, "cruxeval_output": 0.676, "mmlu_pro": {"all": 0.6183, "listed": 0.632}}
V3_MARGIN, V3_MAX_CAPPED = 0.10, 0.10
STOP_MAX_UPPER = 0.05
FUTILITY = 0.02


# --------------------------------------------------------------------------------------------------------------
# runaway


def necessity(arm: str, summary: dict[str, Any], rescue: dict[str, dict[str, Any]], v2: str) -> str:
    comparisons = summary["comparisons"]
    if not summary["v1"]["passed"] or v2 != "pass" or comparisons.get("P", {}).get("outcome") != "effect":
        return "not_tested"
    against_a, p_against = comparisons[arm]["outcome"], rescue[arm]["outcome"]
    if against_a == "no effect" and p_against == "effect":
        return "necessary"
    if against_a == "effect" and p_against == "no effect":
        return "not_necessary"
    if p_against == "effect":
        return "partial"
    return "inconclusive"


def analyze_runaway(
    prompts: list[dict[str, Any]], replies: dict[tuple[str, int], dict[str, Any]], tf: list[dict[str, Any]] | None
) -> dict[str, Any]:
    summary = analyze(prompts, replies, "A", False, tf)
    arms = summary["arms"]
    if "P" not in arms:
        raise SystemExit("no replies of arm P")
    rate_p = arms["P"]["rate"]
    v2 = "pass" if rate_p >= V2_PASS else "fail" if rate_p < V2_FAIL else "partial"
    rescue = {}
    for arm in ("N", "G"):
        if arm in arms:
            c = compare(arms["P"]["capped"], arms["P"]["n"], arms[arm]["capped"], arms[arm]["n"])
            rescue[arm] = {"diff": c.diff, "p_value": c.p_value, "newcombe95": [c.lower, c.upper], "outcome": c.outcome}
    eligible = {
        arm: bool(
            summary["v1"]["passed"]
            and arms[arm]["rate"] <= ELIGIBLE_MAX_RATE
            and summary["comparisons"][arm]["outcome"] != "effect"
        )
        for arm in ("P", "N", "G")
        if arm in arms
    }
    candidate = next((arm for arm in ("N", "G") if eligible.get(arm)), "none")
    task_arms = [candidate] if candidate != "none" else []
    if eligible.get("P"):
        task_arms.append("P")
    summary["phase2"] = {
        "v2": {"rate": rate_p, "outcome": v2},
        "p_against": rescue,
        "necessity": {arm: necessity(arm, summary, rescue, v2) for arm in rescue},
        "eligible": eligible,
        "candidate": candidate,
        "task_arms": task_arms,
    }
    return summary


def runaway_decisions(summary: dict[str, Any]) -> dict[str, str]:
    p2 = summary["phase2"]
    out = {
        "v1": "passed" if summary["v1"]["passed"] else "failed",
        "v2": p2["v2"]["outcome"],
        "rate_P": f"{p2['v2']['rate']:.4f}",
        "candidate": p2["candidate"],
        "task_arms": ",".join(p2["task_arms"]) or "none",
    }
    for arm, c in summary["comparisons"].items():
        out[f"vs_A_{arm}"] = c["outcome"].replace(" ", "_")
    for arm, label in p2["necessity"].items():
        out[f"necessity_{arm}"] = label
    for arm, ok in p2["eligible"].items():
        out[f"eligible_{arm}"] = "yes" if ok else "no"
    return out


def print_runaway(summary: dict[str, Any]) -> None:
    print_report(summary)
    p2 = summary["phase2"]
    print(f"\nV2: R_P = {p2['v2']['rate']:.3f} (pass >= {V2_PASS}, fail < {V2_FAIL}): {p2['v2']['outcome']}")
    print("\nP vs arm  R diff   one-sided Fisher p  Newcombe 95%       outcome   necessity")
    for arm, c in p2["p_against"].items():
        print(
            f"P vs {arm:4s} {c['diff']:+.3f}   {c['p_value']:<19.3g} [{c['newcombe95'][0]:+.3f}, "
            f"{c['newcombe95'][1]:+.3f}]  {c['outcome']:9s} {p2['necessity'][arm]}"
        )
    print(
        f"\neligible for capability scoring: {p2['eligible']}; candidate {p2['candidate']}; scored arms "
        f"{p2['task_arms'] or 'none'}"
    )


# --------------------------------------------------------------------------------------------------------------
# capability


def text_sha(reply: dict[str, Any]) -> str:
    return hashlib.sha256(reply["text"].encode("utf-8")).hexdigest()


def grade_reply(prompt: dict[str, Any], reply: dict[str, Any], timeout: float) -> dict[str, Any]:
    text = answer_text(reply["text"], reply["stop_reason"] != "cap")
    task = prompt["task"]
    if task == "mmlu_pro":
        result = mmlu_pro.grade(prompt, text)
    else:
        result = cruxeval.grade(prompt, task.removeprefix("cruxeval_"), text, timeout)
    return {
        "arm": reply["arm"],
        "prompt_id": reply["prompt_id"],
        "task": task,
        "text_sha256": text_sha(reply),
        "capped": reply["stop_reason"] == "cap",
        "n_new": reply["n_new"],
        **result,
    }


def grade_all(
    prompts: dict[int, dict[str, Any]],
    replies: dict[tuple[str, int], dict[str, Any]],
    grades_path: Path,
    workers: int,
    timeout: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Grades of every reply; grades already in `grades_path` for an unchanged reply are reused, new ones are
    appended (CRUXEval programs run in parallel worker processes)."""
    cached = {(g["arm"], g["prompt_id"]): g for g in read_jsonl(grades_path)}
    grades, todo = {}, []
    for key, reply in replies.items():
        old = cached.get(key)
        if old is not None and old["text_sha256"] == text_sha(reply):
            grades[key] = old
        else:
            todo.append(key)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        new = list(pool.map(lambda k: grade_reply(prompts[k[1]], replies[k], timeout), todo))
    if new:
        append_jsonl_many(grades_path, new)
    grades |= {(g["arm"], g["prompt_id"]): g for g in new}
    return grades


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_task_differences(items: Sequence[tuple[str, str, int]], draws: int, seed: int) -> np.ndarray:
    """[draws, len(TASKS)] mean paired differences per task, resampling groups with replacement within each
    domain. `items` holds (task, group, difference in {-1, 0, 1})."""
    rng = np.random.default_rng(seed)
    column = {task: i for i, task in enumerate(TASKS)}
    num, den = np.zeros((draws, len(TASKS))), np.zeros((draws, len(TASKS)))
    for stratum in sorted(set(STRATUM.values())):
        groups = sorted({g for t, g, _ in items if STRATUM[t] == stratum})
        if not groups:
            continue
        row = {g: i for i, g in enumerate(groups)}
        sums, counts = np.zeros((len(groups), len(TASKS))), np.zeros((len(groups), len(TASKS)))
        for task, group, diff in items:
            if STRATUM[task] == stratum:
                sums[row[group], column[task]] += diff
                counts[row[group], column[task]] += 1
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
        num += weights @ sums
        den += weights @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def interval(values: np.ndarray, level: float) -> list[float]:
    return [float(v) for v in np.quantile(values, [(1 - level) / 2, (1 + level) / 2])]


def arm_scores(grades: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for task in TASKS:
        rows = [g for g in grades if g["task"] == task]
        if rows:
            out[task] = {
                "n": len(rows),
                "accuracy": sum(g["correct"] for g in rows) / len(rows),
                "capped": sum(g["capped"] for g in rows),
                "no_answer": sum(not g["answer"] for g in rows),
                "length_median": statistics.median(g["n_new"] for g in rows),
            }
    out["all"] = {"n": len(grades), "capped": sum(g["capped"] for g in grades)}
    return out


def compare_arms(
    arm: str,
    prompts: dict[int, dict[str, Any]],
    grades: dict[tuple[str, int], dict[str, Any]],
    level: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(pid for a, pid in grades if a == "A" and (arm, pid) in grades)
    items = [
        (
            prompts[pid]["task"],
            prompts[pid]["group"],
            int(grades[(arm, pid)]["correct"]) - int(grades[("A", pid)]["correct"]),
        )
        for pid in ids
    ]
    boot = bootstrap_task_differences(items, draws, seed)
    present = [t for t in TASKS if any(it[0] == t for it in items)]
    weight = {t: WEIGHTS[t] / sum(WEIGHTS[u] for u in present) for t in present}
    targets = {}
    for task in present:
        pairs = [
            (grades[(arm, pid)]["correct"], grades[("A", pid)]["correct"])
            for pid in ids
            if prompts[pid]["task"] == task
        ]
        b = sum(x and not y for x, y in pairs)
        c = sum(y and not x for x, y in pairs)
        targets[task] = {
            "n": len(pairs),
            "diff": (b - c) / len(pairs),
            "ci95": interval(boot[:, TASKS.index(task)], 0.95),
            "arm_right_reference_wrong": b,
            "arm_wrong_reference_right": c,
            "mcnemar_p": mcnemar_p(b, c),
        }
    composite = sum(weight[t] * targets[t]["diff"] for t in present)
    composite_boot = sum(weight[t] * boot[:, TASKS.index(t)] for t in present)
    capped_arm = sum(grades[(arm, pid)]["capped"] for pid in ids)
    capped_ref = sum(grades[("A", pid)]["capped"] for pid in ids)
    lower, upper = newcombe(capped_arm, len(ids), capped_ref, len(ids))
    return {
        "pairs": len(ids),
        "weights": weight,
        "targets": targets,
        "composite": {"diff": composite, "interval": interval(composite_boot, level), "level": level},
        "capped": {"arm": capped_arm, "reference": capped_ref, "newcombe95": [lower, upper]},
    }


def v3_check(scores: dict[str, Any], pool: str) -> dict[str, Any]:
    floors = {
        t: (REFERENCE_9984[t][pool] if t == "mmlu_pro" else REFERENCE_9984[t]) - V3_MARGIN for t in TASKS if t in scores
    }
    capped_share = scores["all"]["capped"] / max(1, scores["all"]["n"])
    passed = all(scores[t]["accuracy"] >= f for t, f in floors.items()) and capped_share <= V3_MAX_CAPPED
    return {"floors": floors, "capped_share": capped_share, "passed": passed}


def decide(comparison: dict[str, Any], v3: bool, stops: bool) -> str:
    lower, upper = comparison["composite"]["interval"]
    if not v3:
        return "not_interpreted_v3"
    if not stops:
        return "not_interpreted_stopping"
    if lower > 0:
        worse = any(t["ci95"][1] < 0 for t in comparison["targets"].values())
        return "trade_off" if worse else "better"
    return "worse" if upper < 0 else "not_better"


def analyze_capability(
    prompts: dict[int, dict[str, Any]],
    grades: dict[tuple[str, int], dict[str, Any]],
    runaway: dict[str, Any],
    level: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    arms = sorted({a for a, _ in grades}, key=lambda a: (a != "A", a))
    if "A" not in arms:
        raise SystemExit("no grades of arm A")
    pools = {p["pool"] for p in prompts.values() if p["task"] == "mmlu_pro"}
    pool = pools.pop() if len(pools) == 1 else "all"
    scores = {a: arm_scores([g for (b, _), g in grades.items() if b == a]) for a in arms}
    v3 = v3_check(scores["A"], pool)
    eligible = runaway["phase2"]["eligible"]
    comparisons = {}
    for arm in arms[1:]:
        c = compare_arms(arm, prompts, grades, level, draws, seed)
        c["stops_normally"] = bool(eligible.get(arm)) and c["capped"]["newcombe95"][1] < STOP_MAX_UPPER
        c["decision"] = decide(c, v3["passed"], c["stops_normally"])
        c["extend"] = c["decision"] == "not_better" and c["composite"]["interval"][1] >= FUTILITY
        comparisons[arm] = c
    return {
        "mmlu_pro_pool": pool,
        "scores": scores,
        "v3": v3,
        "comparisons": comparisons,
        "reference_9984": REFERENCE_9984,
    }


def capability_decisions(summary: dict[str, Any]) -> dict[str, str]:
    out = {"v3": "passed" if summary["v3"]["passed"] else "failed"}
    for arm, c in summary["comparisons"].items():
        low, high = c["composite"]["interval"]
        out[f"decision_{arm}"] = c["decision"]
        out[f"composite_{arm}"] = f"{c['composite']['diff']:+.4f} [{low:+.4f},{high:+.4f}]"
        out[f"stops_{arm}"] = "yes" if c["stops_normally"] else "no"
        out[f"extend_{arm}"] = "yes" if c["extend"] else "no"
    return out


def print_capability(summary: dict[str, Any]) -> None:
    print(f"MMLU-Pro pool: {summary['mmlu_pro_pool']}")
    print("\narm  task              n    accuracy  capped  no answer  median length")
    for arm, scores in summary["scores"].items():
        for task in TASKS:
            if task in scores:
                s = scores[task]
                print(
                    f"{arm:4s} {task:17s} {s['n']:<4d} {s['accuracy']:.3f}     {s['capped']:<7d} {s['no_answer']:<10d} "
                    f"{s['length_median']:.0f}"
                )
    v3 = summary["v3"]
    floors = ", ".join(f"{t} >= {f:.3f}" for t, f in v3["floors"].items())
    print(
        f"\nV3 (arm A): {floors}; capped share {v3['capped_share']:.3f} (<= {V3_MAX_CAPPED}): "
        f"{'passed' if v3['passed'] else 'FAILED - no capability claim'}"
    )
    for arm, c in summary["comparisons"].items():
        print(f"\narm {arm} against A on {c['pairs']} paired prompts")
        for task, t in c["targets"].items():
            print(
                f"  {task:17s} diff {t['diff']:+.3f}  95% [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  "
                f"discordant {t['arm_right_reference_wrong']}/{t['arm_wrong_reference_right']}  "
                f"McNemar p {t['mcnemar_p']:.3g}"
            )
        comp = c["composite"]
        print(
            f"  composite D {comp['diff']:+.4f}  {comp['level']:.1%} [{comp['interval'][0]:+.4f}, "
            f"{comp['interval'][1]:+.4f}]  (weights {c['weights']})"
        )
        cap = c["capped"]
        print(
            f"  capped {cap['arm']} vs {cap['reference']}, Newcombe 95% upper {cap['newcombe95'][1]:+.3f} "
            f"(< {STOP_MAX_UPPER}); stops normally: {c['stops_normally']}"
        )
        print(f"  decision: {c['decision']}; extension worthwhile: {c['extend']}")


def analyze_quicklook(
    prompts: dict[int, dict[str, Any]],
    replies: dict[tuple[str, int], dict[str, Any]],
    grades: dict[tuple[str, int], dict[str, Any]],
    limits: argparse.Namespace,
) -> dict[str, Any]:
    """Arm A's quick-look replies: stopped count, median length and correct answers per task, against the limits."""
    rows = list(grades.values())
    lengths = [g["n_new"] for g in rows]
    correct = {t: sum(g["correct"] for g in rows if g["task"] == t) for t in TASKS}
    stopped = sum(not g["capped"] for g in rows)
    median = statistics.median(lengths)
    problems = []
    if stopped < len(rows) - limits.max_capped:
        problems.append(f"{len(rows) - stopped} replies reached the cap (at most {limits.max_capped})")
    if median > limits.max_median:
        problems.append(f"median length {median:.0f} tokens exceeds {limits.max_median}")
    problems += [f"{t}: {c} correct (need {limits.min_correct})" for t, c in correct.items() if c < limits.min_correct]
    for (arm, pid), reply in sorted(replies.items(), key=lambda kv: (prompts[kv[0][1]]["task"], kv[0][1])):
        g, p = grades[(arm, pid)], prompts[pid]
        print(
            f"\n=== {p['task']} prompt {pid}, arm {arm}: stop {reply['stop_reason']}, {reply['n_new']} tokens, "
            f"answer {g['answer']!r}, reference {p['gold']!r}, {'correct' if g['correct'] else 'wrong'}"
        )
        print(f"reply (last 500 characters): {reply['text'][-500:]}")
    return {
        "replies": len(rows),
        "stopped": stopped,
        "median_length": median,
        "correct": correct,
        "problems": problems,
    }


# --------------------------------------------------------------------------------------------------------------


def write_decisions(path: Path, decisions: dict[str, str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{k} {v}\n" for k, v in decisions.items()), encoding="utf-8", newline="\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("runaway")
    cap = sub.add_parser("capability")
    quick = sub.add_parser("quicklook")
    for p in (run, cap, quick):
        p.add_argument("--prompts", type=Path, required=True)
        p.add_argument("--replies", type=Path, nargs="+", required=True)
        p.add_argument("--json", type=Path)
        p.add_argument("--decisions", type=Path)
    run.add_argument("--teacher-forced", type=Path)
    cap.add_argument("--runaway", type=Path, required=True, help="JSON summary of the runaway analysis")
    cap.add_argument("--level", type=float, default=0.95)
    cap.add_argument("--draws", type=int, default=10000)
    cap.add_argument("--seed", type=int, default=9984)
    for p in (cap, quick):
        p.add_argument("--grades", type=Path, required=True, help="grades file (created, reused, appended)")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--timeout", type=float, default=cruxeval.TIMEOUT_S)
    quick.add_argument("--max-capped", type=int, default=2, help="replies at the cap allowed")
    quick.add_argument("--max-median", type=float, default=2000, help="largest median reply length in tokens")
    quick.add_argument("--min-correct", type=int, default=2, help="correct replies needed in each task")
    args = parser.parse_args(argv)
    replies = load_replies(args.replies)
    if args.command == "quicklook":
        prompts_by_id = {p["idx"]: p for p in read_jsonl(args.prompts)}
        if set(prompts_by_id) - {pid for _, pid in replies}:
            print("replies are missing for some quick-look prompts", file=sys.stderr)
            return 2
        grades = grade_all(prompts_by_id, replies, args.grades, args.workers, args.timeout)
        summary = analyze_quicklook(prompts_by_id, replies, grades, args)
        verdict = "; ".join(summary["problems"]) or "within the limits"
        print(
            f"\n{summary['replies']} replies, {summary['stopped']} stopped, median length "
            f"{summary['median_length']:.0f} tokens, correct per task {summary['correct']}: {verdict}"
        )
        if args.json:
            write_json_atomic(args.json, summary)
        return 3 if summary["problems"] else 0
    if args.command == "runaway":
        replies = {key: r for key, r in replies.items() if key[0] in RUNAWAY_ARMS}  # phase 1's B to D left out
        prompts = read_jsonl(args.prompts)
        missing = [
            (a, p["idx"]) for a in sorted({a for a, _ in replies}) for p in prompts if (a, p["idx"]) not in replies
        ]
        if missing:
            print(f"{len(missing)} pairs are missing, first {missing[:5]}", file=sys.stderr)
            return 2
        tf = read_jsonl(args.teacher_forced) if args.teacher_forced else None
        summary = analyze_runaway(prompts, replies, tf)
        print_runaway(summary)
        decisions = runaway_decisions(summary)
    else:
        prompts_by_id = {p["idx"]: p for p in read_jsonl(args.prompts)}
        arms = sorted({a for a, _ in replies})
        missing = [(a, i) for a in arms for i in prompts_by_id if (a, i) not in replies]
        if missing:
            print(f"{len(missing)} pairs are missing, first {missing[:5]}", file=sys.stderr)
            return 2
        grades = grade_all(prompts_by_id, replies, args.grades, args.workers, args.timeout)
        summary = analyze_capability(prompts_by_id, grades, read_json(args.runaway), args.level, args.draws, args.seed)
        print_capability(summary)
        decisions = capability_decisions(summary)
    if args.json:
        write_json_atomic(args.json, summary)
    if args.decisions:
        write_decisions(args.decisions, decisions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
