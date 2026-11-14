"""Check that two prompt files hold the same records for some tasks.

Phase 2c analyses phase 2's CRUXEval replies together with new MMLU-Pro replies, so the CRUXEval prompts of its
prompt file must be phase 2's prompts, record for record; and the prompts it runs must be the MMLU-Pro prompts of
that file.

Usage: python -m swaptest.sameprompts --reference FILE --file FILE --tasks cruxeval_input,cruxeval_output
Exit code 0 when both files hold at least one prompt of every listed task and the same set of records for them;
1 otherwise, with a line that says what differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from swaptest.records import read_jsonl


def task_records(path: Path, tasks: set[str]) -> dict[int, str]:
    """The records of the listed tasks, as canonical JSON by prompt id."""
    records: dict[int, str] = {}
    for prompt in read_jsonl(path):
        if prompt["task"] in tasks:
            records[prompt["idx"]] = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return records


def compare(reference: Path, other: Path, tasks: set[str]) -> dict[str, Any]:
    a, b = task_records(reference, tasks), task_records(other, tasks)
    present = {t: (0, 0) for t in tasks}
    for name, records in (("reference", a), ("file", b)):
        for text in records.values():
            task = json.loads(text)["task"]
            n_ref, n_file = present[task]
            present[task] = (n_ref + (name == "reference"), n_file + (name == "file"))
    return {
        "only_reference": len(set(a) - set(b)),
        "only_file": len(set(b) - set(a)),
        "different": sum(a[i] != b[i] for i in set(a) & set(b)),
        "equal": sum(a[i] == b[i] for i in set(a) & set(b)),
        "empty_tasks": sorted(t for t, (n_ref, n_file) in present.items() if n_ref == 0 or n_file == 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--tasks", required=True, help="comma-separated task names")
    args = parser.parse_args(argv)
    tasks = set(args.tasks.split(","))
    result = compare(args.reference, args.file, tasks)
    same = not (result["only_reference"] or result["only_file"] or result["different"] or result["empty_tasks"])
    print(
        f"tasks {','.join(sorted(tasks))}: {result['equal']} records equal, {result['different']} differ, "
        f"{result['only_reference']} only in {args.reference.name}, {result['only_file']} only in {args.file.name}"
        + (f", no prompt of {', '.join(result['empty_tasks'])} in one of the files" if result["empty_tasks"] else "")
        + (": same prompts" if same else ": NOT the same prompts")
    )
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
