"""MMLU-Pro with chain-of-thought few-shot prompts: prompt building and answer extraction.

Data: `TIGER-Lab/MMLU-Pro` at revision b189ec765aa7ed75c8acfea42df31fdae71f97be (the revision #9984 re-joined its
labels to): `data/test-00000-of-00001.parquet` (12,032 rows) and `data/validation-00000-of-00001.parquet` (70
rows, the pool of worked examples), with columns question_id, question, options, answer, answer_index,
cot_content, category and src.

Prompt, as Marin's canonical MMLU-Pro task at 80004cc06 builds it (`taskcompendium/importers/evalchemy/
reasoning.py`, a port of Marin's Evalchemy): an instruction naming the category; up to five worked examples of the
same category from the validation split, in file order; then the question, its options lettered A, B, ... (options
"N/A" are left out) and "Answer: Let's think step by step." The number of examples is the largest of 5, 4, ..., 1
whose prompt text has fewer than 32,768 - 30,720 = 2,048 tokens (the task's fit-the-context rule with #9984's
context and generation cap), else 0. Each prompt is one user message; thinking stays on, as in #9984.

Answer, with the same task's extractor (#9984 describes its evaluator's extraction the same way): in the text after
the last end-of-reasoning marker, the first match of `answer is \\(?([A-J])\\)?`, else the first match of
`Answer:\\s*([A-J])`, both ignoring case and upper-cased; else the last standalone capital letter A-J, matched with
case. The graded text is the final content (`swaptest.answers`). Two quirks of this extractor are kept: the
case-insensitive patterns read "the answer is a ..." as A, and the fallback reads a final pronoun "I" as the
letter I.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swaptest.answers import final_text
from swaptest.chat import ChatTokenizer

INITIAL_PROMPT = (
    "The following are multiple choice questions (with answers) about {$}. Think step by step "
    'and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n'
)
MAX_SHOTS = 5
FIT_LIMIT = 32768 - 30720  # #9984's context window minus its generation cap
FIRST_PATTERNS = (re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE), re.compile(r"Answer:\s*([A-J])", re.IGNORECASE))
LAST_LETTER = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)
COLUMNS = ("question_id", "question", "options", "answer", "answer_index", "cot_content", "category", "src")
TEST_ROWS, VALIDATION_ROWS = 12032, 70


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a split, from parquet or from JSONL with the same columns."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path, columns=list(COLUMNS)).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not isinstance(row.get("options"), list):
            raise ValueError(f"row {i} of {path} lacks a question string or an options list")
        if not isinstance(row.get("answer"), str) or not re.fullmatch(r"[A-J]", row["answer"]):
            raise ValueError(f"row {i} of {path} has answer {row.get('answer')!r}, expected a letter A-J")
    return rows


def example_text(example: dict[str, Any], including_answer: bool) -> str:
    options = [option for option in example["options"] if option != "N/A"]
    text = "Question:\n" + example["question"] + "\nOptions:\n"
    text += "".join(f"{chr(65 + index)}. {option}\n" for index, option in enumerate(options))
    if including_answer:
        cot = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        return text + cot + "\n\n"
    return text + "Answer: Let's think step by step."


def build_prompt(support: Sequence[dict[str, Any]], row: dict[str, Any], shots: int) -> str:
    category = row["category"]
    selected = [example for example in support if example["category"] == category][:shots]
    return (
        INITIAL_PROMPT.replace("{$}", category)
        + "\n"
        + "".join(example_text(example, including_answer=True) for example in selected)
        + example_text(row, including_answer=False)
    )


def fit_prompt(support: Sequence[dict[str, Any]], row: dict[str, Any], tokenizer: ChatTokenizer) -> tuple[str, int]:
    """The prompt text with the most worked examples that fits, and that number (0 always counts as fitting)."""
    for shots in range(MAX_SHOTS, 0, -1):
        text = build_prompt(support, row, shots)
        if len(tokenizer.encode(text, add_special_tokens=False)) < FIT_LIMIT:
            return text, shots
    return build_prompt(support, row, 0), 0


def extract_answer(text: str) -> str | None:
    text = final_text(text)
    for pattern in FIRST_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    match = LAST_LETTER.search(text)
    return match.group(0) if match else None


def grade(row: dict[str, Any], graded_text: str) -> dict[str, Any]:
    answer = extract_answer(graded_text) if graded_text else None
    return {"correct": answer == row["gold"], "answer": answer}
