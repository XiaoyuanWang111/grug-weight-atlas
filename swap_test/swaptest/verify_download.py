"""Compare a downloaded Hugging Face snapshot with the repository's file listing at a pinned revision.

Every listed file must exist with the listed size; with --sha256, every LFS file must also have the listed
SHA-256. Two listings are used when available, and both must match the directory:
  - the listing saved with this package (`swaptest/data/listings/`) for the two model repositories of the
    experiment, read from the Hub API with file metadata before the experiment was planned;
  - the listing the Hub API returns now for the pinned revision (skipped with --no-hub).
When both exist they must also agree with each other. --include restricts the comparison to file names that
match one of the given shell patterns and --exclude drops names that match one (for a repository of which only
some files were downloaded).

Usage: python -m swaptest.verify_download --repo ORG/NAME --revision COMMIT --dir DIR [--repo-type model|dataset]
       [--include PATTERN ...] [--exclude PATTERN ...] [--sha256] [--no-hub] [--workers 8]
Exit code 1 if anything differs or no listing is available.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

from swaptest.records import sha256_file

Listing = dict[str, tuple[int, str | None]]  # file name -> (size, LFS SHA-256 or None)


def remote_listing(repo: str, revision: str, repo_type: str = "model") -> Listing:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo, repo_type=repo_type, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"the Hub resolved {repo}@{revision} to {info.sha}")
    return {s.rfilename: (s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings}


def listing_file_name(repo: str, revision: str) -> str:
    """Repository name without its owner and the first 12 characters of the commit; the file's content names the
    full repository and commit, which must match."""
    return f"{repo.rsplit('/', 1)[-1]}@{revision[:12]}.json"


def saved_listing(repo: str, revision: str) -> Listing | None:
    """The listing saved with this package for repo@revision, or None."""
    entry = resources.files("swaptest").joinpath("data/listings", listing_file_name(repo, revision))
    if not entry.is_file():
        return None
    data = json.loads(entry.read_text("utf-8"))
    if data["repo"] != repo or data["revision"] != revision:
        raise ValueError(f"saved listing {entry} is for {data['repo']}@{data['revision']}")
    return {name: (size, sha) for name, (size, sha) in data["files"].items()}


def restrict(listing: Listing, include: list[str] | None, exclude: list[str] | None = None) -> Listing:
    """Entries whose names match one of `include` (all when None) and none of `exclude`."""

    def matches(name: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(name, p) for p in patterns)

    return {
        n: v
        for n, v in listing.items()
        if (not include or matches(n, include)) and not (exclude and matches(n, exclude))
    }


def compare_listing(listing: Listing, directory: Path, check_sha256: bool, workers: int) -> list[str]:
    problems = []
    for name, (size, _) in sorted(listing.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif path.stat().st_size != size:
            problems.append(f"size of {name}: {path.stat().st_size}, listed {size}")
    if check_sha256:
        hashed = [(n, sha) for n, (_, sha) in sorted(listing.items()) if sha and (directory / n).is_file()]
        with ThreadPoolExecutor(workers) as pool:
            digests = pool.map(lambda item: sha256_file(directory / item[0]), hashed)
            problems += [f"SHA-256 of {n} differs" for (n, sha), d in zip(hashed, digests, strict=True) if d != sha]
    return problems


def compare_listings(saved: Listing, hub: Listing) -> list[str]:
    """Differences between the saved listing and the Hub's (size and LFS SHA-256 per file)."""
    problems = [f"only in the saved listing: {n}" for n in sorted(set(saved) - set(hub))]
    problems += [f"only in the Hub listing: {n}" for n in sorted(set(hub) - set(saved))]
    problems += [
        f"saved and Hub listings differ for {n}: {saved[n]} vs {hub[n]}"
        for n in sorted(set(saved) & set(hub))
        if saved[n] != hub[n]
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset"))
    parser.add_argument("--include", nargs="+", help="compare only files matching these shell patterns")
    parser.add_argument("--exclude", nargs="+", help="leave out files matching these shell patterns")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--no-hub", action="store_true", help="use only the saved listing")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    listings: dict[str, Listing] = {}
    saved = saved_listing(args.repo, args.revision)
    if saved is not None:
        listings["saved"] = restrict(saved, args.include, args.exclude)
    if not args.no_hub:
        hub = remote_listing(args.repo, args.revision, args.repo_type)
        listings["Hub"] = restrict(hub, args.include, args.exclude)
    if not listings or not any(listings.values()):
        print(f"no listing with matching files for {args.repo}@{args.revision}")
        return 1

    problems = compare_listings(listings["saved"], listings["Hub"]) if len(listings) == 2 else []
    for source, listing in listings.items():
        found = compare_listing(listing, args.dir, args.sha256 and source == next(iter(listings)), args.workers)
        problems += [f"{source} listing: {p}" for p in found]
    for problem in problems:
        print(problem)
    listing = next(iter(listings.values()))
    shards = sum(name.endswith(".safetensors") for name in listing)
    total = sum(size for size, _ in listing.values())
    print(
        f"{args.repo}@{args.revision} against the {' and '.join(listings)} listing(s): {len(listing)} files "
        f"({shards} safetensors shards, {total:,} bytes){', SHA-256 checked' if args.sha256 else ''}; "
        f"{'all match' if not problems else f'{len(problems)} problems'}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
