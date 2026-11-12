"""The Gram-matrix distances of pairwise.py on synthetic checkpoints whose distances are known."""

import json

import numpy as np

import pairwise


def unit_row(component, parent, children):
    """One units.jsonl row for a unit with the given parent and child vectors, as atlas2.py writes it."""
    deltas = [c - parent for c in children]
    n = len(children)
    return {
        "unit": component,
        "tensor": component,
        "component": component,
        "layer": 0,
        "size": parent.size,
        "parent_norm": float(np.linalg.norm(parent)),
        "children": [{"delta_norm": float(np.linalg.norm(d)), "changed": float((d != 0).mean())} for d in deltas],
        "gram": [[float(deltas[i] @ deltas[j]) for j in range(n)] for i in range(n)],
        "pairs": [[i, j, 0, 0, int((children[i] == children[j]).sum())] for i in range(n) for j in range(i + 1, n)],
    }


# parent P with |P| = 10; |A - P| = 0.5, |B - P| = 0.5, |A - B| = 0.6
P = np.array([6.0, 8.0, 0.0, 0.0])
A = P + np.array([0.0, 0.0, 0.3, 0.4])
B = P + np.array([0.0, 0.0, -0.3, 0.4])


def test_three_checkpoints_one_group():
    rows = [unit_row("model.self_attn.q_proj.weight", P, [A, B])]
    acc = pairwise.accumulate(rows, 2, 32.0)
    dist, diff = pairwise.matrices(acc["attention q k v o"], 2)
    np.testing.assert_allclose(dist, [[0, 5, 5], [5, 0, 6], [5, 6, 0]], atol=1e-12)
    # A and B each change 2 of 4 stored values; they are equal in 3 of 4
    np.testing.assert_allclose(diff, [[0, 50, 50], [50, 0, 25], [50, 25, 0]], atol=1e-12)
    for g in pairwise.GROUPS:
        if g != "attention q k v o":
            assert acc[g]["size"] == 0


def test_expert_weight_counts_the_sample_at_its_true_size():
    # an expert unit sampled once stands for `weight` experts; the "all" distances must equal those of the vectors
    # with the expert part written out `weight` times
    weight = 3
    pe, ae, be = np.array([6.0, 8.0]), np.array([6.3, 8.4]), np.array([6.0, 8.0])
    pn, an, bn = np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.array([1.0, 10.0])
    rows = [
        unit_row("model.mlp.experts.gate_proj.weight", pe, [ae, be]),
        unit_row("model.self_attn.q_proj.weight", pn, [an, bn]),
    ]
    dist, _ = pairwise.matrices(pairwise.accumulate(rows, 2, float(weight))["all"], 2)
    full = {
        name: np.concatenate([np.tile(e, weight), n]) for name, e, n in (("P", pe, pn), ("A", ae, an), ("B", be, bn))
    }
    norm = np.linalg.norm(full["P"])
    expected = [
        [0, np.linalg.norm(full["A"] - full["P"]), np.linalg.norm(full["B"] - full["P"])],
        [np.linalg.norm(full["A"] - full["P"]), 0, np.linalg.norm(full["A"] - full["B"])],
        [np.linalg.norm(full["B"] - full["P"]), np.linalg.norm(full["A"] - full["B"]), 0],
    ]
    np.testing.assert_allclose(dist, 100 * np.array(expected) / norm, rtol=1e-12)


def test_main_prints_the_matrices(tmp_path, monkeypatch, capsys):
    meta = {"parent": "open-athena/Snowball-67B-A2B-Parent", "children": ["org/Grug-67B-A2B-A", "org/Grug-67B-A2B-B"]}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    row = unit_row("model.self_attn.q_proj.weight", P, [A, B])
    (tmp_path / "units.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["pairwise.py", str(tmp_path)])
    pairwise.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "units 1 | layers [0] | parent (marked *) open-athena/Snowball-67B-A2B-Parent"
    assert "== attention q k v o (0.0 M scalars)" in lines
    assert lines[lines.index("  relative distance, percent of the parent's norm") + 3].split() == [
        "A",
        "5.0000",
        "0.0000",
        "6.0000",
    ]
