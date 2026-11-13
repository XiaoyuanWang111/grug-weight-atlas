"""Print one field of a JSON file, for shell scripts.

Usage: python -m swaptest.jsonfield FILE KEY[.KEY ...]
Each KEY selects a field of an object, or an element of a list by its number. Strings print as they are, other
values as JSON (true, false, null, numbers, objects). Exit code 1 if the field does not exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def field(data: Any, path: str) -> Any:
    for key in path.split("."):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        value = field(json.loads(Path(args[0]).read_text(encoding="utf-8")), args[1])
    except (KeyError, IndexError, ValueError, TypeError) as error:
        print(f"no field {args[1]} in {args[0]}: {error!r}", file=sys.stderr)
        return 1
    print(value if isinstance(value, str) else json.dumps(value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
