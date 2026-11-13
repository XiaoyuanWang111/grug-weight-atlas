"""Print the two SHA-256 values the pre-registration records: the prompts file and the harness tree.

The harness tree hash covers every file under the harness directory except `PREREG.md` (which receives the
hashes) and caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.egg-info`, `build`, `dist`, `.git`). It is
the SHA-256 of the lines "<file SHA-256>  <relative path>\\n", sorted by path, which is what this shell command
prints from inside the harness directory:

  find . -type f ! -path './PREREG.md' ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' \\
      ! -path '*/.ruff_cache/*' ! -path '*.egg-info/*' ! -path './build/*' ! -path './dist/*' ! -path './.git/*' \\
      | sed 's|^\\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum

Usage: python -m swaptest.hashes --prompts prompts.jsonl [--harness DIR]
Output: "prompts_sha256 <hex>" and "harness_tree_sha256 <hex>", one per line.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from swaptest.records import sha256_file

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
EXCLUDED_TOP = {"PREREG.md", "build", "dist"}


def tree_files(root: Path) -> list[str]:
    """Relative POSIX paths of the files the tree hash covers, sorted bytewise."""
    paths = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if not path.is_file() or rel.parts[0] in EXCLUDED_TOP:
            continue
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
            continue
        paths.append(rel.as_posix())
    return sorted(paths, key=lambda p: p.encode("utf-8"))


def tree_sha256(root: Path) -> tuple[str, int]:
    files = tree_files(root)
    lines = "".join(f"{sha256_file(root / rel)}  {rel}\n" for rel in files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest(), len(files)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--harness", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    digest, count = tree_sha256(args.harness)
    print(f"prompts_sha256 {sha256_file(args.prompts)}")
    print(f"harness_tree_sha256 {digest}")
    print(f"# {count} files under {args.harness.resolve()}")


if __name__ == "__main__":
    main()
