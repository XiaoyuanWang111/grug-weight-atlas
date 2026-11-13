"""CRUXEval input and output prediction: prompts, answer reading and grading by execution.

Data: `cruxeval-org/cruxeval` at revision b96af0450242eb4da433032b90998f25588a5d0f, file `test.jsonl`: 800 rows
with fields `id` ("sample_N"), `code` (Python source of a function `f`), `input` (the arguments of one call, as
source text) and `output` (the call's value, as a literal).

Prompts: the two templates of Marin's canonical CRUXEval task at 80004cc06 (`crux_input.txt`, `crux_output.txt`
in `lib/taskcompendium/.../importers/evalchemy/assets/`, copied byte for byte to `swaptest/data/`; that task's
NOTICE derives them from Marin's Evalchemy, lm-evaluation-harness (MIT) and CRUXEval), filled as that task fills
them: `{input}` and `{output}` first, `{code}` last. Each prompt is one user message; thinking stays on, as in
#9984.

Answer reading, the CRUXEval harness rule: the graded text is the final content (`swaptest.answers`); when it
holds `[ANSWER]`, the text after the first `[ANSWER]` up to the next `[/ANSWER]` (where the harness stops
generating); that text is split at the first `==`. Input prediction takes the call before it, from the first `f(`
that does not continue a name or attribute (so `assert f(16)` gives `f(16)`); output prediction takes all of the
text after it, so a second `==` inside a string literal stays in the answer. Without `==` the whole text is the
answer. As in the harness, an input answer without a call of `f` and an output answer that contains the call
`f(<input>)` itself are wrong.

Grading: the program `{code}\\nassert {output} == {answer}` runs in a fresh Python process: the standard library
only (no site-packages), an environment holding only PATH and a fixed hash seed (so set iteration order is reproducible), a temporary
working directory, a 10-second limit (Marin's canonical CRUXEval limit) and, on Linux, limits on memory, CPU time
and file size. It passes when the process exits with status 0 and its last output is a random receipt
that the program prints after the assertion, so an answer that ends the process early (`exit(0)`) fails.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

DIRECTIONS = ("input", "output")
ANSWER_OPEN, ANSWER_CLOSE = "[ANSWER]", "[/ANSWER]"
CALL = re.compile(r"(?<![\w.])f\(")
TIMEOUT_S = 10.0
MEMORY_LIMIT = 2 << 30
FILE_LIMIT = 1 << 20
ROWS = 800


def template(direction: str) -> str:
    return resources.files("swaptest").joinpath(f"data/cruxeval_{direction}.txt").read_text("utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if not all(isinstance(row.get(k), str) for k in ("id", "code", "input", "output")):
            raise ValueError(f"row {i} of {path} lacks one of the string fields id, code, input, output")
        if not re.fullmatch(r"sample_\d+", row["id"]):
            raise ValueError(f"row {i} of {path} has id {row['id']!r}, expected sample_N")
    return rows


def problem_number(row: dict[str, str]) -> int:
    return int(row["id"].removeprefix("sample_"))


def build_prompt(row: dict[str, str], direction: str) -> str:
    text = template(direction).replace("{input}", row["input"]).replace("{output}", row["output"])
    return text.replace("{code}", row["code"])


def answer_block(text: str) -> str:
    """The text after the first [ANSWER] up to the next [/ANSWER]; all of the text when there is no [ANSWER]."""
    if ANSWER_OPEN not in text:
        return text
    return text.split(ANSWER_OPEN, 1)[1].split(ANSWER_CLOSE, 1)[0]


def extract_answer(text: str, direction: str, call_input: str) -> str | None:
    """The answer expression read from the graded text, or None when the rule rejects it."""
    block = answer_block(text)
    if direction == "input":
        before = block.split("==", 1)[0]
        call = CALL.search(before)
        return before[call.start() :].strip() if call else None
    answer = block.split("==", 1)[1].strip() if "==" in block else block.strip()
    return None if f"f({call_input})" in answer else answer


def limit_resources() -> None:
    """Runs in the child before the program (Linux): memory, CPU seconds and file size."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIMEOUT_S) + 1, int(TIMEOUT_S) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT, FILE_LIMIT))


def passes(program: str, timeout: float = TIMEOUT_S) -> bool:
    """Run `program` in a fresh isolated Python process; True when it exits 0 after printing its receipt."""
    receipt = secrets.token_hex(16)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.py"
        path.write_text(f"{program}\nprint({receipt!r})\n", encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            done = subprocess.run(
                [sys.executable, "-S", "-s", str(path)],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                preexec_fn=limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False
    output = done.stdout.decode("utf-8", errors="replace").split()
    return done.returncode == 0 and bool(output) and output[-1] == receipt


def grade(row: dict[str, Any], direction: str, graded_text: str, timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """Correctness and the answer read from the graded text of one reply."""
    answer = extract_answer(graded_text, direction, row["input"]) if graded_text else None
    if not answer:
        return {"correct": False, "answer": answer}
    program = f"{row['code']}\nassert {row['output']} == {answer}"
    return {"correct": passes(program, timeout), "answer": answer}
