"""The merge builder against a direct formula, the shard reader, spec files, --check-against and merged arms."""

import json

import pytest
import torch
from helpers import random_weights, tiny_config, write_checkpoint
from safetensors.torch import load_file

from swaptest import merge
from swaptest.config import ModelConfig
from swaptest.loading import load_merged_model
from swaptest.merge import Checkpoint, MergeSpec, Source, build_merge, check_against, merge_values
from swaptest.patch import GROUP_PATTERNS, arm_index

CFG = ModelConfig.from_dict(tiny_config())
BIAS = r"model\.layers\.\d+\.mlp\.router\.bias"


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def marin_formula(anchor, donors, coefficients):
    """Task arithmetic transcribed from marin.merging.arithmetic.merge_tensor (scale 1.0), whole tensors."""
    base = anchor.float()
    result = torch.zeros_like(base)
    for c, d in zip(coefficients, donors, strict=True):
        result.add_((d.float() - base).mul_(c))
    return result.mul_(1.0).add_(base).to(anchor.dtype)


def make_sources(tmp_path, bias_exact=True):
    """B (bfloat16, float32 router bias) and two fine-tunes E, E2 that store the router bias in bfloat16.

    With bias_exact the fine-tunes' bias equals B's after widening (B's float32 values are bfloat16 numbers, as in
    the 67B checkpoints); otherwise B's bias holds float32 values that bfloat16 rounds."""
    base = random_weights(CFG, seed=3, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(4)
    experts = []
    for k in range(2):
        tuned = {}
        for name, t in base.items():
            if name.endswith(".mlp.router.bias"):
                tuned[name] = t.to(torch.bfloat16)
            else:
                tuned[name] = (t.float() + 0.05 * (k + 1) * torch.randn(t.shape, generator=g)).to(torch.bfloat16)
        experts.append(tuned)
    if bias_exact:
        for name in base:
            if name.endswith(".mlp.router.bias"):
                base[name] = base[name].to(torch.bfloat16).float()
    else:
        for name in base:
            if name.endswith(".mlp.router.bias"):
                base[name] = base[name] + 1e-4
    dirs = []
    for label, weights, shards in (("b", base, 4), ("e", experts[0], 2), ("e2", experts[1], 3)):
        write_checkpoint(tmp_path / label, tiny_config(), weights, shards=shards)
        dirs.append(tmp_path / label)
    return dirs, base, experts


def spec_for(dirs, coefficients, **kwargs):
    return MergeSpec(
        Source(dirs[0]), tuple(Source(d, {BIAS: "bfloat16"}) for d in dirs[1:]), tuple(coefficients), **kwargs
    )


def direct(base, experts, coefficients):
    return {n: marin_formula(base[n], [e[n].to(base[n].dtype) for e in experts], coefficients) for n in base}


def test_merge_values_is_marin_arithmetic_and_order_matters():
    g = torch.Generator().manual_seed(0)
    anchor = torch.randn(4000, generator=g).to(torch.bfloat16)
    donors = [(anchor.float() + 0.01 * torch.randn(4000, generator=g)).to(torch.bfloat16) for _ in range(3)]
    for coefficients in ((0.5, 0.5, 0.0), (1.0, 1.0, 1.0), (0.3, 0.7, 0.1)):
        ours = merge_values(anchor, donors, coefficients, "t")
        assert torch.equal(bits(ours), bits(marin_formula(anchor, donors, coefficients)))
    # Adding the anchor first rounds differently: the order of the operations is part of the result.
    anchor_first = anchor.float().clone()
    for c, d in zip((0.3, 0.7, 0.1), donors, strict=True):
        anchor_first.add_((d.float() - anchor.float()).mul_(c))
    assert not torch.equal(
        bits(anchor_first.to(torch.bfloat16)), bits(merge_values(anchor, donors, (0.3, 0.7, 0.1), "t"))
    )


def test_merge_values_rejects_bad_input():
    a = torch.ones(3)
    with pytest.raises(ValueError, match="incompatible"):
        merge_values(a, [torch.ones(3, dtype=torch.bfloat16)], (1.0,), "t")
    with pytest.raises(ValueError, match="nonfinite"):
        merge_values(a, [torch.tensor([1.0, float("inf"), 0.0])], (1.0,), "t")
    with pytest.raises(ValueError, match="overflow"):
        merge_values(torch.full((2,), 3e38), [torch.full((2,), -3e38)], (2.0,), "t")


def test_reader_matches_safetensors(tmp_path):
    dirs, _, _ = make_sources(tmp_path)
    checkpoint = Checkpoint(dirs[0])
    loaded = {}
    for path in sorted(dirs[0].glob("*.safetensors")):
        loaded |= load_file(str(path))
    assert sorted(checkpoint.tensors) == sorted(loaded)
    for name, tensor in loaded.items():
        assert checkpoint.read(name).dtype == tensor.dtype and torch.equal(checkpoint.read(name), tensor)
    name = "model.layers.2.mlp.experts.up_proj.weight"
    assert torch.equal(checkpoint.read(name, 3, 6), loaded[name][3:6])


@pytest.mark.parametrize("slice_values", [7, 1 << 25])
def test_build_merge_equals_the_direct_formula(tmp_path, monkeypatch, slice_values):
    monkeypatch.setattr(merge, "SLICE_VALUES", slice_values)
    dirs, base, experts = make_sources(tmp_path)
    built = build_merge(spec_for(dirs, (1.0, 1.0)), lambda _: None)
    expected = direct(base, experts, (1.0, 1.0))
    assert sorted(built) == sorted(expected)
    for name in expected:
        assert built[name].dtype == base[name].dtype and torch.equal(bits(built[name]), bits(expected[name])), name
    bias = "model.layers.0.mlp.router.bias"
    assert built[bias].dtype == torch.float32 and torch.equal(built[bias], base[bias])  # frozen bias stays B's


def test_check_against_finds_equality_and_a_planted_one_value_difference(tmp_path, capsys):
    dirs, base, experts = make_sources(tmp_path)
    half = direct(base, experts, (0.5, 0.5))
    reference = tmp_path / "ta05"
    write_checkpoint(reference, tiny_config(), half, shards=len(half))  # one tensor per shard, as Marin writes
    spec = tmp_path / "spec.json"
    widen = f"{BIAS}=bfloat16"
    args = ["--base", str(dirs[0]), "--donor", str(dirs[1]), "--donor", str(dirs[2]), "--coefficients", "1,1"]
    arms = ["--arm", "P", "--override", "N", "attention_gates", str(dirs[0])]
    assert merge.main([*args, "--widen", widen, *arms, "--write-spec", str(spec)]) == 0
    assert MergeSpec.load(spec).arms == {"P": None, "N": merge.ArmSource("attention_gates", dirs[0])}
    summary = tmp_path / "check.json"
    check = [
        "--spec",
        str(spec),
        "--coefficients",
        "0.5,0.5",
        "--check-against",
        str(reference),
        "--json",
        str(summary),
    ]
    code = merge.main(check)
    assert code == 0 and json.loads(summary.read_text())["equal"]
    assert "EQUAL in every stored value" in capsys.readouterr().out

    name = "model.layers.3.self_attn.attn_gate.weight"
    planted = half | {name: half[name].clone()}
    planted[name].view(torch.int16)[2, 5] ^= 1  # one bit of one stored value
    write_checkpoint(reference, tiny_config(), planted, shards=len(planted))
    code = merge.main(check)
    result = json.loads(summary.read_text())
    assert code == 1 and not result["equal"]
    assert result["differing"] == [
        {"name": name, "shard": result["first_difference"]["shard"], "values": 1, "reference_equals_base": False}
    ]
    first = result["first_difference"]
    assert first["tensor"] == name and first["index"] == [2, 5] and first["flat_index"] == 2 * CFG.hidden_size + 5
    assert first["merge"]["bits"] != first["reference"]["bits"]


def test_router_bias_kept_at_the_anchor_has_its_own_exit_code(tmp_path):
    dirs, base, experts = make_sources(tmp_path, bias_exact=False)
    half = direct(base, experts, (0.5, 0.5))
    for name in half:
        if name.endswith(".mlp.router.bias"):
            assert not torch.equal(half[name], base[name])
            half[name] = base[name]  # the published merge kept the anchor's bias
    write_checkpoint(tmp_path / "ref", tiny_config(), half)
    summary = check_against(spec_for(dirs, (0.5, 0.5)), tmp_path / "ref", lambda _: None)
    assert summary["router_bias_kept_at_anchor"] and len(summary["differing"]) == CFG.num_layers
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(spec_for(dirs, (0.5, 0.5)).to_json()))
    assert merge.main(["--spec", str(spec), "--check-against", str(tmp_path / "ref")]) == 3
    kept = spec_for(dirs, (0.5, 0.5), tensor_coefficients={BIAS: (0.0, 0.0)})
    assert check_against(kept, tmp_path / "ref", lambda _: None)["equal"]


def test_first_only_stops_at_the_first_differing_tensor(tmp_path):
    dirs, base, experts = make_sources(tmp_path)
    write_checkpoint(tmp_path / "ref", tiny_config(), direct(base, experts, (1.0, 1.0)))
    summary = check_against(spec_for(dirs, (0.5, 0.5)), tmp_path / "ref", lambda _: None, first_only=True)
    assert len(summary["differing"]) == 1 and summary["first_difference"]["tensor"] == summary["differing"][0]["name"]


def test_spec_validation(tmp_path):
    dirs, base, _ = make_sources(tmp_path)
    with pytest.raises(ValueError, match="one coefficient per donor"):
        spec_for(dirs, (1.0,))
    with pytest.raises(ValueError, match="nonnegative"):
        spec_for(dirs, (1.0, -0.5))
    with pytest.raises(ValueError, match="matches no tensor"):
        merge.MergePlan(spec_for(dirs, (1.0, 1.0), tensor_coefficients={"nothing": (0.0, 0.0)}))
    with pytest.raises(ValueError, match="incompatible"):  # without widening the router bias dtypes differ
        merge.MergePlan(MergeSpec(Source(dirs[0]), (Source(dirs[1]), Source(dirs[2])), (1.0, 1.0)))
    smaller = {n: t for n, t in base.items() if n != "lm_head.weight"}
    write_checkpoint(tmp_path / "short", tiny_config(), smaller)
    with pytest.raises(ValueError, match="names"):
        merge.MergePlan(spec_for([dirs[0], dirs[1], tmp_path / "short"], (1.0, 1.0)))
    assert (
        merge.main(
            [
                "--base",
                str(dirs[0]),
                "--donor",
                str(dirs[1]),
                "--coefficients",
                "1",
                "--write-spec",
                str(tmp_path / "s.json"),
            ]
        )
        == 2
    )  # bias dtypes differ without --widen
    spec = spec_for(dirs, (1.0, 1.0), tensor_coefficients={BIAS: (0.0, 0.0)})
    assert MergeSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec


def test_merged_arms_take_their_gates_from_other_checkpoints(tmp_path):
    dirs, base, experts = make_sources(tmp_path)
    half = direct(base, experts, (0.5, 0.5))
    write_checkpoint(tmp_path / "ta05", tiny_config(), half)
    spec = spec_for(
        dirs,
        (1.0, 1.0),
        arms={
            "P": None,
            "N": merge.ArmSource("attention_gates", tmp_path / "ta05"),
            "G": merge.ArmSource("attention_gates", dirs[0]),
        },
    )
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_json()))
    loaded = load_merged_model(tmp_path / "ta05", path, "PNG", lambda _: None)
    gates = [n for n in loaded.weights if GROUP_PATTERNS["attention_gates"].fullmatch(n)]
    assert len(gates) == 3 * CFG.num_layers
    variants = loaded.model.variants
    for name in gates:
        (n_arms, n_value), (g_arms, g_value) = variants[name]
        assert n_arms == (arm_index("N"),) and torch.equal(n_value, half[name])
        assert g_arms == (arm_index("G"),) and torch.equal(g_value, base[name])
    ids = torch.tensor([[256, 5, 6, 7]] * 3)
    cache = loaded.model.new_cache(3, 4)
    hidden = loaded.model.forward(ids, cache, 0, torch.zeros(3, dtype=torch.long), torch.tensor([4, 5, 6]))
    logits = loaded.model.logits(hidden[:, -1]).float()
    assert (logits[0] - logits[1]).abs().max() > 1e-3 and (logits[0] - logits[2]).abs().max() > 1e-3

    same = spec_for(dirs, (1.0, 1.0), arms={"N": merge.ArmSource("attention_gates", tmp_path / "full")})
    write_checkpoint(tmp_path / "full", tiny_config(), direct(base, experts, (1.0, 1.0)))
    path.write_text(json.dumps(same.to_json()))
    with pytest.raises(ValueError, match="equals the merged tensor"):
        load_merged_model(tmp_path / "ta05", path, "N", lambda _: None)
    with pytest.raises(ValueError, match="not defined"):
        load_merged_model(tmp_path / "ta05", path, "G", lambda _: None)
