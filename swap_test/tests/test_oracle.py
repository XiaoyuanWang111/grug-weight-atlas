"""The model against Marin's committed Levanter oracle (5 layers, window 3, float32).

Fixture `fixtures/levanter_oracle/` is MarinSkyRL's `tests/fixtures/grug_training_oracle/` (Marin 80004cc06),
generated on CPU JAX by `tests/cluster/vllm/grug_training_oracle.py` at Marin 6195867835ef; its SHA-256 values
equal those listed in that fixture's README.
"""

import json

import numpy as np
import torch
from helpers import FIXTURES
from safetensors.torch import load_file

from swaptest.checks import cached_logits, full_logits
from swaptest.config import ModelConfig
from swaptest.model import GrugModel

ORACLE = FIXTURES / "levanter_oracle"


def oracle_model() -> tuple[GrugModel, dict]:
    cfg = ModelConfig.from_dict(json.loads((ORACLE / "config.json").read_text()))
    weights = load_file(str(ORACLE / "model.safetensors"))
    return GrugModel(cfg, weights), json.loads((ORACLE / "config.json").read_text())


def test_hidden_states_and_logits_match_oracle():
    model, _ = oracle_model()
    obs = dict(np.load(ORACLE / "observations.npz"))
    ids = torch.from_numpy(obs["input_ids"]).long()
    trace: list[torch.Tensor] = []
    cache = model.new_cache(ids.shape[0], ids.shape[1])
    logits = model.logits(model.forward(ids, cache, 0, torch.zeros(ids.shape[0], dtype=torch.long), trace=trace))
    names = ["hidden.embed"] + [f"hidden.layer.{i}" for i in range(model.cfg.num_layers)] + ["hidden.final"]
    for name, got in zip(names, trace, strict=True):
        np.testing.assert_allclose(got.numpy(), obs[name], atol=2e-5, rtol=2e-5, err_msg=name)
    np.testing.assert_allclose(logits.numpy(), obs["logits"], atol=2e-5, rtol=2e-5)
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    assert abs(float(loss) - float(obs["loss"])) < 1e-5


def test_cached_decode_matches_full_forward_across_the_window():
    model, raw = oracle_model()
    ids = torch.randint(0, raw["vocab_size"], (1, 16), generator=torch.Generator().manual_seed(0))
    full = full_logits(model, ids)
    for prefill_len, chunk in ((5, 5), (8, 3), (1, 1)):
        assert float((cached_logits(model, ids, prefill_len, chunk) - full).abs().max()) < 1e-6
