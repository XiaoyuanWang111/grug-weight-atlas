"""Which part of a reply the capability graders read: the final content after the reasoning.

The model writes its reasoning after `<|start_think|>`, closes it with `<|end_think|>` and writes its final answer
after that. #9984's evaluator (Evalchemy ae362881) read MMLU-Pro answers from the final content when there was
one, fell back to the reasoning when a reply ended normally without final content, and gave no answer when a reply
was cut off inside its reasoning. Marin's canonical task code at 80004cc06 implements the same rule
(`verifyit.adapters.completion.answer_text` and `final_text`: the text after the last end-of-reasoning marker).
The rule is used here for both capability tasks; #9984's CRUXEval extraction is not available.
"""

from __future__ import annotations

REASONING_END_MARKERS = ("</think>", "<|end_think|>")
START_THINK = "<|start_think|>"


def last_end_marker(text: str) -> tuple[int, int] | None:
    """Start and end offsets of the last end-of-reasoning marker, or None."""
    found = [(i, i + len(m)) for m in REASONING_END_MARKERS if (i := text.rfind(m)) >= 0]
    return max(found, key=lambda span: span[1]) if found else None


def final_text(text: str) -> str:
    """The text after the last end-of-reasoning marker (all of it when there is none)."""
    span = last_end_marker(text)
    return text if span is None else text[span[1] :]


def answer_text(text: str, stopped: bool) -> str:
    """The graded text of a reply ('' when there is none).

    `text` is the decoded reply with special tokens kept and the final stop token left out; `stopped` is whether
    the reply ended with a stop token rather than at the token cap.
    """
    span = last_end_marker(text)
    if span is not None:
        reasoning, content = text[: span[0]], text[span[1] :]
    elif text.lstrip().startswith(START_THINK):
        reasoning, content = text, ""
    else:
        reasoning, content = "", text
    if content:
        return content
    if not stopped:
        return ""
    return final_text(reasoning.replace(START_THINK, "", 1))
