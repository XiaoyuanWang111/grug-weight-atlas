"""The pre-registration hashes: prompts file and harness tree (PREREG.md and caches excluded)."""

import hashlib

from swaptest import hashes


def make_tree(root):
    (root / "swaptest" / "__pycache__").mkdir(parents=True)
    (root / "swaptest" / "a.py").write_bytes(b"print(1)\n")
    (root / "swaptest" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (root / "swaptest.egg-info").mkdir()
    (root / "swaptest.egg-info" / "PKG-INFO").write_bytes(b"x")
    (root / "README.md").write_bytes(b"readme\n")
    (root / "PREREG.md").write_bytes(b"blank\n")


def test_tree_hash_is_the_hash_of_sorted_sha256sum_lines(tmp_path):
    make_tree(tmp_path)
    assert hashes.tree_files(tmp_path) == ["README.md", "swaptest/a.py"]
    lines = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in (("README.md", b"readme\n"), ("swaptest/a.py", b"print(1)\n"))
    )
    assert hashes.tree_sha256(tmp_path) == (hashlib.sha256(lines.encode()).hexdigest(), 2)


def test_filling_in_prereg_or_caches_leaves_the_hash_unchanged(tmp_path, capsys):
    make_tree(tmp_path)
    before = hashes.tree_sha256(tmp_path)
    (tmp_path / "PREREG.md").write_bytes(b"filled in\n")
    (tmp_path / "swaptest" / "__pycache__" / "b.pyc").write_bytes(b"\1")
    assert hashes.tree_sha256(tmp_path) == before
    (tmp_path / "swaptest" / "a.py").write_bytes(b"print(2)\n")
    assert hashes.tree_sha256(tmp_path) != before
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_bytes(b'{"idx": 1}\n')
    hashes.main(["--prompts", str(prompts), "--harness", str(tmp_path / "swaptest")])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"prompts_sha256 {hashlib.sha256(prompts.read_bytes()).hexdigest()}"
    assert out[1].startswith("harness_tree_sha256 ") and len(out[1].split()[1]) == 64
