"""GSM8K answer extraction and scoring in the manner of Evalchemy's "flexible" GSM8K filter.

The scored text is cut at the first end-of-turn pattern, then the answer is the last `\\boxed{number}`, else the
last "answer is / answer: number", else the last number. Evalchemy compares the result with the reference using
Minerva equivalence; here the two are compared as numbers after removing `$` and thousands separators, which
agrees with it on GSM8K's numeric references.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
BOXED = re.compile(rf"\\boxed\{{\s*({NUMBER})\s*\}}")
FINAL = re.compile(rf"(?i)\b(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({NUMBER})")
NUMERIC = re.compile(NUMBER)
STOP_PATTERNS = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
    "\nYou are an AI assistant",
    "Question:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)


def truncate_at_stop(text: str) -> str:
    cuts = [i for pattern in STOP_PATTERNS if (i := text.find(pattern)) >= 0]
    return text[: min(cuts)] if cuts else text


def extract_answer(text: str) -> str | None:
    text = truncate_at_stop(text)
    for pattern in (BOXED, FINAL, NUMERIC):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    return None


def to_number(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace("$", "").replace(",", "").strip())
    except InvalidOperation:
        return None


def reference_answer(solution: str) -> str:
    """The number after the final `####` of a GSM8K solution."""
    return solution.rsplit("####", 1)[-1].strip()


def is_correct(text: str, gold: str) -> bool:
    answer = extract_answer(text)
    if answer is None:
        return False
    predicted, reference = to_number(answer), to_number(gold)
    return predicted is not None and reference is not None and predicted == reference
