"""Patch name rules on the published manifest and the 67B tensor names, and the checks at load time."""

import pytest
import torch
from helpers import perturbed, random_weights, tiny_config
from safetensors.torch import save_file

from swaptest.checkpoint import expected_shapes
from swaptest.config import ModelConfig
from swaptest.patch import check_manifest_groups, group_names, load_manifest, load_patch

MERGE_CONFIG = {  # config.json of the 67B checkpoint (shapes only)
    "model_type": "grug_moe",
    "architectures": ["GrugMoeForCausalLM"],
    "grugmoe_artifact_schema_version": 1,
    "vocab_size": 128256,
    "hidden_size": 2560,
    "num_hidden_layers": 26,
    "num_attention_heads": 20,
    "num_key_value_heads": 5,
    "head_dim": 128,
    "num_experts": 256,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 1280,
    "shared_expert_intermediate_size": 2560,
    "sliding_window": 2048,
    "qk_mult": 1.75,
    "qk_mult_long_scale": 1.0,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "disable_pko": True,
    "disable_long_rope": True,
}
CFG = ModelConfig.from_dict(tiny_config())


def test_rules_give_78_and_26_on_the_published_manifest():
    manifest = load_manifest()
    names = manifest["groups"]["attention_gates"] + manifest["groups"]["router_weights"]
    assert len(group_names(names, "attention_gates", 26)) == 78
    assert len(group_names(names, "router_weights", 26)) == 26
    check_manifest_groups(manifest, names, 26)
    assert manifest["tensors"] == 104


def test_rules_select_exactly_the_manifest_groups_among_all_67b_tensors():
    names = list(expected_shapes(ModelConfig.from_dict(MERGE_CONFIG)))
    assert len(names) == 502
    manifest = load_manifest()
    assert group_names(names, "attention_gates", 26) == sorted(manifest["groups"]["attention_gates"])
    assert group_names(names, "router_weights", 26) == sorted(manifest["groups"]["router_weights"])


def test_a_missing_layer_is_rejected():
    names = load_manifest()["groups"]["router_weights"][1:]
    with pytest.raises(ValueError, match="found 25"):
        group_names(names, "router_weights", 26)


def save_patch(tmp_path, tensors):
    path = tmp_path / "patch.safetensors"
    save_file({n: t.contiguous() for n, t in tensors.items()}, str(path))
    return path


def test_load_patch_returns_both_groups_and_arm_tables(tmp_path):
    weights = random_weights(CFG, dtype=torch.bfloat16)
    patch = load_patch(save_patch(tmp_path, perturbed(weights, CFG)), weights, CFG.num_layers)
    assert {g: len(t) for g, t in patch.groups.items()} == {"attention_gates": 15, "router_weights": 5}
    tables = patch.arm_tables("ABCD")
    assert [len(tables[i]) for i in range(4)] == [0, 15, 5, 20]
    assert tables[3]["model.layers.2.mlp.router.weight"] is tables[2]["model.layers.2.mlp.router.weight"]


@pytest.mark.parametrize("fault", ["shape", "dtype", "identical", "extra"])
def test_load_patch_rejects_bad_tensors(tmp_path, fault):
    weights = random_weights(CFG, dtype=torch.bfloat16)
    tensors = perturbed(weights, CFG)
    name = "model.layers.1.self_attn.attn_gate.weight"
    if fault == "shape":
        tensors[name] = tensors[name][:, :-1]
    elif fault == "dtype":
        tensors[name] = tensors[name].float()
    elif fault == "identical":
        tensors[name] = weights[name].clone()
    else:
        tensors["model.layers.1.mlp.router.bias"] = weights["model.layers.1.mlp.router.bias"] + 1
    with pytest.raises(ValueError):
        load_patch(save_patch(tmp_path, tensors), weights, CFG.num_layers)
