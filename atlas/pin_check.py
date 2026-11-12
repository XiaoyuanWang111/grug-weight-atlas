"""Do the weight files at the commits that Marin's issues pin equal those at the commits read here?

marin#9984 and the recipes of the merging campaigns pin earlier commits of several repositories than the ones the
atlas runs read. For each repository this script asks the Hugging Face API for the file list at both commits, with
blob hashes, and compares the safetensors files by LFS SHA-256 and size. It also lists the other files that differ.

Usage: python atlas/pin_check.py > results/pin_check.txt

Network: two API requests per repository; no file content is read.
"""

import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
ORG = "open-athena"
# repository: (commit pinned by marin#9984 or the campaign recipes, commit read by the atlas runs)
PAIRS = {
    "Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92": (
        "fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab",
        "80975627aebeaec3ef946ad609d9c1d95458ebd8",
    ),
    "Snowball-67B-A2B-CruxEval-Expert-E-Step272": (
        "fb4f8051a344fec59f902bdd2ca876ac1b2f3296",
        "2eb988bb5df70378950d983ac8e70dae550cf6d5",
    ),
    "Snowball-67B-A2B-MMLUPro-Expert-E2-Step254": (
        "acb97183af633533df4f4f63d83adc1e0a7a9748",
        "0f22c03aff076f46c292196cf0d03a9bc4ab81c5",
    ),
    "Snowball-67B-A2B-Merge-TA0.5": (
        "07001c3333337bdea61647c74bcb227f60310fa6",
        "cd0605de0ab76473b2262a9545596436e726b27d",
    ),
    "Snowball-67B-A2B-Merge-TIES-d0.5": (
        "ffb5b68590fdf7c5fe01351a332e7915d574bc61",
        "a8ade36ffb263fee9e49c8a508b82845464e8804",
    ),
    "Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02": (
        "50fe2f488aee7730bbccc721024b35229075602e",
        "059f83373e03d146506a4be37513d45e6ec6b126",
    ),
    "Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04": (
        "fd07ef0f2b78a83099a0042e59c00de986e7b647",
        "1515d17b22346106f9d9e549aaca65411fd70860",
    ),
}


def files(repo: str, commit: str) -> dict[str, tuple[str | None, int | None]]:
    """Map each file of the repository at this commit to its content hash and size."""
    url = f"https://huggingface.co/api/models/{ORG}/{repo}/revision/{commit}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "status-check"})
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.load(response)
    out = {}
    for sibling in info["siblings"]:
        lfs = sibling.get("lfs") or {}
        out[sibling["rfilename"]] = (lfs.get("sha256") or sibling.get("blobId"), sibling.get("size"))
    return out


def weights(listing: dict) -> dict:
    return {name: value for name, value in listing.items() if name.endswith(".safetensors")}


def main() -> None:
    for repo, (pinned, read) in PAIRS.items():
        at_pinned = files(repo, pinned)
        time.sleep(0.5)
        at_read = files(repo, read)
        time.sleep(0.5)
        same = weights(at_pinned) == weights(at_read)
        other = sorted(
            name
            for name in set(at_pinned) | set(at_read)
            if not name.endswith(".safetensors") and at_pinned.get(name) != at_read.get(name)
        )
        print(
            f"{repo}: pinned {pinned[:8]} vs read {read[:8]} | safetensors files {len(weights(at_pinned))} / "
            f"{len(weights(at_read))} | weight files identical: {same} | other files that differ: {other}"
        )


if __name__ == "__main__":
    main()
