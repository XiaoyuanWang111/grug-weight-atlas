"""Build the GSM8K prompt file of the swap test from a local copy of the GSM8K test split.

Input: the `openai/gsm8k` `main` test split, as JSONL (one {"question": ..., "answer": ...} object per line) or
as parquet (columns `question` and `answer`). The selection shuffles the row indices 0 .. n-1 with
`random.Random(seed)` and takes `count` of them from `offset` on, so an extension drawn later (offset 48) never
repeats a prompt. Each prompt is one user message "Question: {question}\\nAnswer:" rendered with the checkpoint's
chat template, thinking left at the template default, and encoded without added special tokens.

Usage: python -m swaptest.prompts --gsm8k TEST_FILE --checkpoint DIR --out prompts.jsonl
       [--count 48] [--offset 0] [--seed 9984]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from swaptest.chat import ChatTokenizer, load_tokenizer, render_user_turn
from swaptest.gsm8k import reference_answer
from swaptest.records import write_jsonl_atomic

QUESTION_TEMPLATE = "Question: {question}\nAnswer:"
THINK_LINE = "Reasoning: /think"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"


def read_gsm8k(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"row {i} of {path} lacks a question or answer string")
    return rows


def select_indices(n: int, count: int, offset: int, seed: int) -> list[int]:
    """`count` row indices from a seeded shuffle of 0 .. n-1, starting at `offset`, in ascending order."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    if offset + count > n:
        raise ValueError(f"cannot take {count} prompts from offset {offset} of {n} rows")
    return sorted(order[offset : offset + count])


def check_prompt(text: str, ids: list[int], bos_id: int) -> None:
    """Exactly one start token (first), the thinking line, and the assistant header at the end."""
    if ids.count(bos_id) != 1 or ids[0] != bos_id:
        raise ValueError(f"prompt must hold exactly one start token, at position 0; it holds {ids.count(bos_id)}")
    if THINK_LINE not in text:
        raise ValueError(f"rendered prompt lacks {THINK_LINE!r}")
    if not text.endswith(ASSISTANT_HEADER):
        raise ValueError("rendered prompt does not end with the assistant header")


def build_prompts(rows: list[dict[str, str]], indices: list[int], tokenizer: ChatTokenizer) -> list[dict[str, Any]]:
    prompts = []
    for idx in indices:
        user_text = QUESTION_TEMPLATE.format(question=rows[idx]["question"])
        text, ids = render_user_turn(tokenizer, user_text)
        check_prompt(text, ids, tokenizer.bos_token_id)
        prompts.append(
            {
                "idx": idx,
                "question": rows[idx]["question"],
                "answer": rows[idx]["answer"],
                "gold": reference_answer(rows[idx]["answer"]),
                "user_text": user_text,
                "prompt_text": text,
                "prompt_ids": ids,
            }
        )
    return prompts


def main(argv: list[str] | None = None, tokenizer: ChatTokenizer | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gsm8k", type=Path, required=True, help="GSM8K test split, .jsonl or .parquet")
    parser.add_argument("--checkpoint", type=Path, required=True, help="directory with the tokenizer and template")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9984)
    args = parser.parse_args(argv)
    tokenizer = tokenizer or load_tokenizer(args.checkpoint)
    rows = read_gsm8k(args.gsm8k)
    prompts = build_prompts(rows, select_indices(len(rows), args.count, args.offset, args.seed), tokenizer)
    write_jsonl_atomic(args.out, prompts)
    lengths = [len(p["prompt_ids"]) for p in prompts]
    print(f"wrote {len(prompts)} prompts to {args.out}: indices {[p['idx'] for p in prompts]}")
    print(f"prompt tokens: min {min(lengths)}, max {max(lengths)}, mean {sum(lengths) / len(lengths):.1f}")


if __name__ == "__main__":
    main()
