"""Grade the probe's sampled replies with Marin's closure grader and print the table of README.md.

Every prompt record holds the stored prefix (`grading.prefix`), the prefix as its prompt shows it
(`grading.prefix_seen`: the stored prefix for arm U and for the control rows, the prefix without its trailing
newline for arm T), the expected text and the closure format. A reply is graded twice with
`verifyit.candidate.grade_candidate` on a `StructureClosureSpec`:

  now   the reply joined to the stored prefix, which is what training does;
  seen  the reply joined to the prefix as the prompt shows it.

A blank reply scores 0, as a failed submission does. Per stratum the script prints the share of replies with full
credit for arm T (the prompt as main renders it) and arm U (the trailing newline kept), their paired difference
with a 95% percentile interval from 10,000 bootstrap replicates over rows, and the exact McNemar test, by the
rules of PREREG.md section 4. A reply that reached the token cap before its closure region completed is
"unresolved"; a decision that changes when such replies are set to their extreme values is reported as
inconclusive.

Usage: python regrade.py --marin MARIN_CHECKOUT [--prompts prompts.jsonl] [--replies replies.jsonl]
       [--grades grades.jsonl]
MARIN_CHECKOUT is a checkout of marin-community/marin (the probe used 80004cc06); only `lib/verifyit/src` is
imported. With --grades the per-reply grades are written, or compared when the file exists (exit code 1 if they
differ).
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# (stratum, role): the three strata whose prefix ends in a newline, then the control stratum, whose prefix ends in
# no whitespace, so that both prompts are the same. The position in this list seeds the stratum's bootstrap.
STRATA = (
    ("code_fence/code_fence/close_required", "primary"),
    ("tool/tool_delimiter/close_required", "tool"),
    ("tool/repeated_opener_retention/opener_retention", "tool"),
    ("code_fence/code_fence/continue_required", "control"),
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261113
EFFECT = 0.10
ALPHA = 0.05
FULL_CREDIT = 1 - 1e-9
RENDERER_LABELS = ("trimming costs reward", "trimming raises reward")
GRADER_LABELS = ("grading against the seen prefix raises reward", "grading against the seen prefix lowers reward")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_grader(marin: Path):
    """Marin's closure grader and closure-state reader, imported from the checkout."""
    source = marin / "lib" / "verifyit" / "src"
    if not (source / "verifyit").is_dir():
        raise SystemExit(f"{source} holds no verifyit package; --marin must name a checkout of marin")
    sys.path.insert(0, str(source))
    from verifyit.candidate import grade_candidate  # noqa: PLC0415
    from verifyit.closure_scoring import ClosureFormat, closure_state  # noqa: PLC0415
    from verifyit.spec import StructureClosureSpec  # noqa: PLC0415

    def reward(text: str, prefix: str, expected: str, closure_format: str) -> float:
        if not text.strip():
            return 0.0
        spec = StructureClosureSpec(prefix=prefix, expected=expected, format=ClosureFormat(closure_format))
        verdict = grade_candidate(spec, text, {})
        if verdict.status.value != "scored":
            raise ValueError(f"the grader returned status {verdict.status.value}")
        return float(verdict.reward)

    def resolved(text: str, prefix: str, closure_format: str) -> bool:
        """Whether more tokens could no longer change the reward of a reply that reached the cap."""
        state = closure_state(prefix + text, ClosureFormat(closure_format))
        return state.completion_offset is not None and not state.completion_guard

    return reward, resolved


def grade_replies(prompts: dict[int, dict], replies: list[dict], reward, resolved) -> list[dict]:
    grades = []
    for reply in replies:
        prompt = prompts[reply["prompt_id"]]
        grading = prompt["grading"]
        text, capped = reply["text"], reply["stop_reason"] == "cap"
        record = {"idx": prompt["idx"], "row": prompt["row"], "probe_arm": prompt["probe_arm"]}
        record |= {"stratum": prompt["stratum"], "n_new": reply["n_new"], "stop_reason": reply["stop_reason"]}
        for name, prefix in (("now", grading["prefix"]), ("seen", grading["prefix_seen"])):
            record[f"reward_{name}"] = reward(text, prefix, grading["expected"], grading["format"])
            unresolved = capped and not resolved(text, prefix, grading["format"])
            record[f"unresolved_{name}"] = unresolved
        record["empty"] = not text.strip()
        record["equals_golden"] = text == grading["golden"]
        grades.append(record)
    return grades


def mcnemar(new: np.ndarray, old: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided test on full credit: b rows where only `new` has it, c rows where only `old` has it."""
    b = int(np.sum((new >= FULL_CREDIT) & (old < FULL_CREDIT)))
    c = int(np.sum((new < FULL_CREDIT) & (old >= FULL_CREDIT)))
    if b + c == 0:
        return b, c, 1.0
    tail = sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)
    return b, c, min(1.0, 2 * tail)


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def decide(estimate: float, low: float, high: float, p: float, labels: tuple[str, str]) -> str:
    if estimate >= EFFECT and low > 0 and p < ALPHA:
        return labels[0]
    if estimate <= -EFFECT and high < 0 and p < ALPHA:
        return labels[1]
    if low > -EFFECT and high < EFFECT:
        return "no effect"
    return "inconclusive"


def contrast(new: np.ndarray, old: np.ndarray, draws: np.ndarray, labels: tuple[str, str]) -> dict:
    difference = new - old
    low, high = interval(difference[draws].mean(axis=1))
    b, c, p = mcnemar(new, old)
    estimate = float(difference.mean())
    return {
        "estimate": estimate,
        "interval": (low, high),
        "b": b,
        "c": c,
        "p": p,
        "label": decide(estimate, low, high, p, labels),
    }


def robust_contrast(new, old, new_open, old_open, draws, labels) -> dict:
    """The contrast as graded, and its decision when unresolved replies take their extreme values."""
    main = contrast(new, old, draws, labels)
    low = contrast(np.where(new_open, 0.0, new), np.where(old_open, 1.0, old), draws, labels)
    high = contrast(np.where(new_open, 1.0, new), np.where(old_open, 0.0, old), draws, labels)
    same = low["label"] == main["label"] == high["label"]
    return main | {"decision": main["label"] if same else "inconclusive (capped replies)"}


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def points(result: dict) -> str:
    low, high = result["interval"]
    return (
        f"{100 * result['estimate']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}], McNemar "
        f"{result['b']}/{result['c']}, p {result['p']:.2g}: {result['decision']}"
    )


def report(grades: list[dict]) -> None:
    by_stratum: dict[str, dict[str, dict[str, dict]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
    for grade in grades:
        by_stratum[grade["stratum"]][grade["row"]][grade["probe_arm"]] = grade
    for position, (stratum, role) in enumerate(STRATA):
        rows = [arms for _row, arms in sorted(by_stratum[stratum].items())]
        count = len(rows)
        generator = np.random.default_rng([BOOTSTRAP_SEED, position])
        draws = generator.integers(0, count, size=(BOOTSTRAP_REPLICATES, count))
        t_now = np.array([arms["T"]["reward_now"] for arms in rows])
        print(f"\n{stratum} ({role}), {count} rows")
        if role == "control":
            low, high = interval(t_now[draws].mean(axis=1))
            full = int(np.sum(t_now >= FULL_CREDIT))
            print(
                f"  full credit {full} of {count} = {percent(t_now.mean())} [{percent(low)}, {percent(high)}]; "
                f"{sum(arms['T']['unresolved_now'] for arms in rows)} capped replies unresolved"
            )
            continue
        t_seen = np.array([arms["T"]["reward_seen"] for arms in rows])
        u_now = np.array([arms["U"]["reward_now"] for arms in rows])
        opened = {
            "t_now": np.array([arms["T"]["unresolved_now"] for arms in rows]),
            "t_seen": np.array([arms["T"]["unresolved_seen"] for arms in rows]),
            "u_now": np.array([arms["U"]["unresolved_now"] for arms in rows]),
        }
        for name, values in (("T now ", t_now), ("T seen", t_seen), ("U now ", u_now)):
            low, high = interval(values[draws].mean(axis=1))
            print(f"  {name} {percent(values.mean())} [{percent(low)}, {percent(high)}]")
        renderer = robust_contrast(u_now, t_now, opened["u_now"], opened["t_now"], draws, RENDERER_LABELS)
        grader = robust_contrast(t_seen, t_now, opened["t_seen"], opened["t_now"], draws, GRADER_LABELS)
        print(f"  U now - T now  (keep the newline in the prompt): {points(renderer)}")
        print(f"  T seen - T now (grade against the seen prefix):  {points(grader)}")
        for arm in ("T", "U"):
            mine = [arms[arm] for arms in rows]
            print(
                f"  arm {arm}: {sum(g['empty'] for g in mine)} empty replies, "
                f"{sum(g['equals_golden'] for g in mine)} equal to the golden continuation, "
                f"{sum(g['stop_reason'] == 'cap' for g in mine)} at the cap"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marin", type=Path, required=True, help="a checkout of marin-community/marin")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    parser.add_argument("--replies", type=Path, default=HERE / "replies.jsonl")
    parser.add_argument("--grades", type=Path, help="write the per-reply grades, or compare when the file exists")
    args = parser.parse_args()
    prompts = {prompt["idx"]: prompt for prompt in read_jsonl(args.prompts)}
    replies = read_jsonl(args.replies)
    missing = set(prompts) - {reply["prompt_id"] for reply in replies}
    if missing or len(replies) != len(prompts):
        raise SystemExit(f"{len(replies)} replies for {len(prompts)} prompts; {len(missing)} prompts have none")
    reward, resolved = load_grader(args.marin)
    grades = sorted(grade_replies(prompts, replies, reward, resolved), key=lambda grade: grade["idx"])
    print(f"{len(grades)} replies graded")
    report(grades)
    if args.grades is None:
        return 0
    if args.grades.exists():
        same = read_jsonl(args.grades) == grades
        print(f"\ngrades {'equal' if same else 'DIFFER from'} {args.grades.name}")
        return 0 if same else 1
    with open(args.grades, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(grade) + "\n" for grade in grades)
    print(f"\nwrote {args.grades.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
