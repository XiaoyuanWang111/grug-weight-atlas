"""Build the prompt file of phase 2's capability tasks: CRUXEval input and output prediction, and MMLU-Pro.

Selection is seeded, like the GSM8K draw: CRUXEval problems are drawn by shuffling the row numbers 0 .. 799 with
`random.Random(seed)` and taking `count` of them from `offset` on, and the same problems are used for both
directions; MMLU-Pro documents are drawn the same way from the rows of the test split, or, with --mmlu-pro-ids,
from the rows whose question_id the file lists (#9984's 1,606-document subset is not public). A later extension
takes the next `count` numbers of the same shuffles (--offset), so it never repeats a prompt.

Records (one JSON line per prompt, sorted by prompt length so that a wave of the run holds prompts of similar
length and little padding): idx, unique across the tasks and GSM8K (100000 + N for input prediction of CRUXEval
problem sample_N, 200000 + N for its output prediction, 300000 + question_id for MMLU-Pro); task (cruxeval_input,
cruxeval_output or mmlu_pro); group, the unit that the paired bootstrap resamples (the CRUXEval problem, shared by
its two directions, or the MMLU-Pro question text); source_id; user_text, prompt_text and prompt_ids (rendered
and checked as the GSM8K prompts are); gold; and the task's fields (code, input, output; or category, shots, pool).

Usage: python -m swaptest.taskprompts --cruxeval test.jsonl --mmlu-pro-test test.parquet
       --mmlu-pro-validation validation.parquet --checkpoint DIR --out prompts.jsonl [--count 270] [--offset 0]
       [--seed 9984] [--mmlu-pro-ids ids.txt]
       python -m swaptest.taskprompts --check-data --cruxeval ... --mmlu-pro-test ... --mmlu-pro-validation ...
--check-data only checks the three files' row counts (800, 12,032, 70) and fields; exit code 1 if they differ.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from swaptest import cruxeval, mmlu_pro
from swaptest.chat import ChatTokenizer, load_tokenizer, render_user_turn
from swaptest.prompts import check_prompt, select_indices
from swaptest.records import write_jsonl_atomic

IDX_BASE = {"cruxeval_input": 100000, "cruxeval_output": 200000, "mmlu_pro": 300000}
TASKS = tuple(IDX_BASE)


def task_of(idx: int) -> str:
    """The task of a prompt id (ids below 100000 are GSM8K prompts)."""
    for task, base in sorted(IDX_BASE.items(), key=lambda kv: -kv[1]):
        if idx >= base:
            return task
    raise ValueError(f"prompt id {idx} is not a capability prompt")


def render(tokenizer: ChatTokenizer, user_text: str) -> dict[str, Any]:
    text, ids = render_user_turn(tokenizer, user_text)
    check_prompt(text, ids, tokenizer.bos_token_id)
    return {"user_text": user_text, "prompt_text": text, "prompt_ids": ids}


def cruxeval_prompts(rows: list[dict[str, str]], indices: list[int], tokenizer: ChatTokenizer) -> list[dict[str, Any]]:
    prompts = []
    for i in indices:
        row = rows[i]
        n = cruxeval.problem_number(row)
        for direction in cruxeval.DIRECTIONS:
            task = f"cruxeval_{direction}"
            prompts.append(
                {
                    "idx": IDX_BASE[task] + n,
                    "task": task,
                    "group": f"cruxeval:{row['id']}",
                    "source_id": row["id"],
                    **render(tokenizer, cruxeval.build_prompt(row, direction)),
                    "gold": row[direction],
                    "code": row["code"],
                    "input": row["input"],
                    "output": row["output"],
                }
            )
    return prompts


def mmlu_pro_prompts(
    rows: list[dict[str, Any]],
    indices: list[int],
    support: list[dict[str, Any]],
    tokenizer: ChatTokenizer,
    pool: str,
) -> list[dict[str, Any]]:
    prompts = []
    for i in indices:
        row = rows[i]
        user_text, shots = mmlu_pro.fit_prompt(support, row, tokenizer)
        prompts.append(
            {
                "idx": IDX_BASE["mmlu_pro"] + int(row["question_id"]),
                "task": "mmlu_pro",
                "group": "mmlu_pro:" + hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:16],
                "source_id": int(row["question_id"]),
                **render(tokenizer, user_text),
                "gold": row["answer"],
                "category": row["category"],
                "shots": shots,
                "pool": pool,
            }
        )
    return prompts


def check_data(crux: list, test: list, validation: list, counts: bool) -> list[str]:
    """Problems of the data files: repeated ids and, with `counts`, row counts other than the published ones."""
    problems = []
    for name, rows, expected in (
        ("CRUXEval test", crux, cruxeval.ROWS),
        ("MMLU-Pro test", test, mmlu_pro.TEST_ROWS),
        ("MMLU-Pro validation", validation, mmlu_pro.VALIDATION_ROWS),
    ):
        if counts and len(rows) != expected:
            problems.append(f"{name}: {len(rows)} rows, expected {expected}")
    if len({cruxeval.problem_number(r) for r in crux}) != len(crux):
        problems.append("CRUXEval test: problem ids repeat")
    if len({int(r["question_id"]) for r in test}) != len(test):
        problems.append("MMLU-Pro test: question ids repeat")
    return problems


def main(argv: list[str] | None = None, tokenizer: ChatTokenizer | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cruxeval", type=Path, required=True, help="CRUXEval test.jsonl")
    parser.add_argument("--mmlu-pro-test", type=Path, required=True)
    parser.add_argument("--mmlu-pro-validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="directory with the tokenizer and chat template")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--count", type=int, default=270, help="prompts per task")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9984)
    parser.add_argument("--mmlu-pro-ids", type=Path, help="question_id per line: draw MMLU-Pro only from these")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args(argv)
    crux = cruxeval.read_rows(args.cruxeval)
    test = mmlu_pro.read_rows(args.mmlu_pro_test)
    validation = mmlu_pro.read_rows(args.mmlu_pro_validation)
    problems = check_data(crux, test, validation, counts=args.check_data)
    for problem in problems:
        print(problem)
    if args.check_data:
        print(
            f"CRUXEval {len(crux)} rows, MMLU-Pro test {len(test)}, validation {len(validation)}: "
            f"{'as expected' if not problems else 'NOT as expected'}"
        )
        return 1 if problems else 0
    if problems:
        return 1
    if args.checkpoint is None or args.out is None:
        parser.error("--checkpoint and --out are required unless --check-data is given")
    tokenizer = tokenizer or load_tokenizer(args.checkpoint)
    pool, candidates = "all", test
    if args.mmlu_pro_ids is not None:
        wanted = {int(line) for line in args.mmlu_pro_ids.read_text(encoding="utf-8").split()}
        candidates = [r for r in test if int(r["question_id"]) in wanted]
        if len(candidates) != len(wanted):
            print(f"{len(wanted) - len(candidates)} listed question ids are not in the test split", file=sys.stderr)
            return 1
        pool = "listed"
    crux_indices = select_indices(len(crux), args.count, args.offset, args.seed)
    mmlu_indices = select_indices(len(candidates), args.count, args.offset, args.seed)
    prompts = cruxeval_prompts(crux, crux_indices, tokenizer)
    prompts += mmlu_pro_prompts(candidates, mmlu_indices, validation, tokenizer, pool)
    prompts.sort(key=lambda p: (len(p["prompt_ids"]), p["idx"]))
    write_jsonl_atomic(args.out, prompts)
    for task in TASKS:
        lengths = [len(p["prompt_ids"]) for p in prompts if p["task"] == task]
        print(
            f"{task}: {len(lengths)} prompts, prompt tokens min {min(lengths)}, max {max(lengths)}, "
            f"mean {sum(lengths) / len(lengths):.0f}"
        )
    shots = [p["shots"] for p in prompts if p["task"] == "mmlu_pro"]
    print(
        f"MMLU-Pro pool {pool}; worked examples per prompt: "
        + ", ".join(f"{k}: {shots.count(k)}" for k in range(mmlu_pro.MAX_SHOTS, -1, -1))
    )
    print(f"wrote {len(prompts)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
