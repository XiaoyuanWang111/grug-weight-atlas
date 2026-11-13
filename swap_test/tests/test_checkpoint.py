"""Configuration parsing and loading safetensors shards with their stored dtypes."""

import json

import pytest
import torch
from helpers import FIXTURES, STOP_IDS, random_weights, tiny_config, write_checkpoint

from swaptest.checkpoint import cast_weights, check_weights, dtype_counts, load_weights
from swaptest.config import ModelConfig, stop_token_ids

CFG = ModelConfig.from_dict(tiny_config())


def test_shards_load_with_stored_dtypes(tmp_path):
    weights = random_weights(CFG, dtype=torch.bfloat16)
    write_checkpoint(tmp_path, tiny_config(), weights, shards=3)
    loaded = load_weights(tmp_path)
    assert check_weights(loaded, CFG) == torch.bfloat16
    assert loaded["model.layers.0.mlp.router.bias"].dtype == torch.float32
    assert all(torch.equal(loaded[n], weights[n]) for n in weights)
    assert dtype_counts(loaded)["float32"] == CFG.num_layers * CFG.num_experts
    assert stop_token_ids(tmp_path) == STOP_IDS


def test_index_and_shards_must_agree(tmp_path):
    weights = random_weights(CFG)
    write_checkpoint(tmp_path, tiny_config(), weights)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.extra.weight"] = next(iter(index["weight_map"].values()))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index and shards differ"):
        load_weights(tmp_path)


def test_missing_tensor_wrong_shape_and_wrong_dtype_are_rejected():
    weights = random_weights(CFG)
    with pytest.raises(ValueError, match="keys differ"):
        check_weights({n: t for n, t in weights.items() if n != "lm_head.weight"}, CFG)
    with pytest.raises(ValueError, match="shape mismatch"):
        check_weights(weights | {"lm_head.weight": weights["lm_head.weight"][:-1]}, CFG)
    with pytest.raises(ValueError, match="dtype"):
        check_weights(weights | {"lm_head.weight": weights["lm_head.weight"].double()}, CFG)


def test_cast_keeps_router_bias_in_float32():
    cast = cast_weights(random_weights(CFG), torch.bfloat16)
    assert cast["model.layers.0.mlp.router.bias"].dtype == torch.float32
    assert cast["model.layers.0.mlp.router.weight"].dtype == torch.bfloat16


def test_config_reads_the_oracle_and_rejects_other_architectures():
    oracle = ModelConfig.from_dict(json.loads((FIXTURES / "levanter_oracle" / "config.json").read_text()))
    assert oracle.long_layers == [3, 4] and oracle.qk_mult_long_scale == 1.0
    assert CFG.long_layers == [3, 4] and CFG.kv_group == 2
    with pytest.raises(ValueError, match="disagree"):
        ModelConfig.from_dict(tiny_config(num_layers=4))
    with pytest.raises(ValueError, match="sconv"):
        ModelConfig.from_dict(tiny_config(sconv=True))
    with pytest.raises(ValueError, match="schema"):
        ModelConfig.from_dict(tiny_config(grugmoe_artifact_schema_version=2))
    with pytest.raises(ValueError, match="qk_mult"):
        ModelConfig.from_dict({k: v for k, v in tiny_config().items() if k != "qk_mult"})
