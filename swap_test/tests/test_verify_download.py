"""Download verification against file listings (sizes and SHA-256), including the listings saved with the package."""

import hashlib

from swaptest import verify_download
from swaptest.verify_download import compare_listing, compare_listings, restrict, saved_listing

MERGE = ("open-athena/Snowball-67B-A2B-Merge-TA0.5", "cd0605de0ab76473b2262a9545596436e726b27d")
SMALL = ("open-athena/Snowflake-d512-Datakit-SFT-262K-2026.10.22", "cbc90512edcb5e38bad11e82d46bb69c3ef98afa")


def test_compare_listing(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "config.json").write_bytes(b"{}")
    good = hashlib.sha256(b"x" * 10).hexdigest()
    listing = {"a.safetensors": (10, good), "config.json": (2, None)}
    assert compare_listing(listing, tmp_path, check_sha256=True, workers=2) == []
    assert compare_listing({"a.safetensors": (10, "0" * 64)}, tmp_path, True, 2) == ["SHA-256 of a.safetensors differs"]
    problems = compare_listing({"a.safetensors": (11, good), "b.safetensors": (1, None)}, tmp_path, False, 2)
    assert problems == ["size of a.safetensors: 10, listed 11", "missing: b.safetensors"]


def test_saved_listings_match_the_planned_checkpoints():
    merge = saved_listing(*MERGE)
    shards = {n: v for n, v in merge.items() if n.endswith(".safetensors")}
    assert len(shards) == 502 and sum(size for size, _ in shards.values()) == 134_157_855_008
    assert all(sha is not None and len(sha) == 64 for _, sha in shards.values())
    assert merge["tokenizer.json"][0] == 17_209_892 and merge["chat_template.jinja"] == (5706, None)
    small = saved_listing(*SMALL)
    assert "vllm_validation.json" in small and sum(n.endswith(".safetensors") for n in small) == 2
    assert saved_listing(MERGE[0], "0" * 40) is None


def test_saved_listings_of_phase_2_sources():
    prefix = "open-athena/Snowball-67B-A2B-"
    sources = {
        ("10T-Mixed-RLVR-Sync-Step92", "80975627aebeaec3ef946ad609d9c1d95458ebd8"): (79, 134_157_838_912),
        ("CruxEval-Expert-E-Step272", "2eb988bb5df70378950d983ac8e70dae550cf6d5"): (39, 134_157_827_064),
        ("MMLUPro-Expert-E2-Step254", "0f22c03aff076f46c292196cf0d03a9bc4ab81c5"): (39, 134_157_827_064),
    }
    for (name, revision), (count, total) in sources.items():
        listing = saved_listing(prefix + name, revision)
        shards = {n: v for n, v in listing.items() if n.endswith(".safetensors")}
        assert len(shards) == count and sum(size for size, _ in shards.values()) == total
        assert all(sha is not None and len(sha) == 64 for _, sha in shards.values())
        assert listing["model.safetensors.index.json"][1] is None


def test_restrict_and_compare_listings():
    listing = {"main/test-00000-of-00001.parquet": (10, "a" * 64), "main/train-00000-of-00001.parquet": (20, None)}
    assert list(restrict(listing, ["main/test-*"])) == ["main/test-00000-of-00001.parquet"]
    assert restrict(listing, None) == listing
    assert list(restrict(listing, None, ["*test*"])) == ["main/train-00000-of-00001.parquet"]
    assert restrict(listing, ["main/*"], ["*.parquet"]) == {}
    assert compare_listings(listing, listing) == []
    other = {"main/test-00000-of-00001.parquet": (11, "a" * 64), "README.md": (1, None)}
    assert compare_listings(listing, other) == [
        "only in the saved listing: main/train-00000-of-00001.parquet",
        "only in the Hub listing: README.md",
        "saved and Hub listings differ for main/test-00000-of-00001.parquet: (10, 'aaaa"
        + "a" * 60
        + "') vs (11, 'aaaa"
        + "a" * 60
        + "')",
    ]


def test_offline_check_against_the_saved_listing(tmp_path, capsys):
    assert (
        verify_download.main(["--repo", "someone/unknown", "--revision", "0" * 40, "--dir", str(tmp_path), "--no-hub"])
        == 1
    )
    assert (
        verify_download.main(
            ["--repo", SMALL[0], "--revision", SMALL[1], "--dir", str(tmp_path), "--no-hub", "--include", "*.json"]
        )
        == 1
    )
    assert "saved listing: missing: vllm_validation.json" in capsys.readouterr().out
