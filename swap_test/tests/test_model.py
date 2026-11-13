"""Model and cache on tiny random configurations: reference parity, cache, arms, ragged batches, row dropping."""

import pytest
import torch
from helpers import perturbed, random_weights, tiny_config
from reference_eager import reference_logits

from swaptest.checks import cached_logits, compare_logits, full_logits
from swaptest.config import ModelConfig
from swaptest.engine import left_pad, prefill
from swaptest.kvcache import KVCache, entry_slots
from swaptest.model import GrugModel, Overrides
from swaptest.patch import ARM_GROUPS, GROUP_PATTERNS, arm_index

CFG = ModelConfig.from_dict(tiny_config())
GEN = torch.Generator().manual_seed(7)
SEQ = torch.randint(0, CFG.vocab_size, (2, 13), generator=GEN)


def arm_weights(weights, patch, arm):
    """The full weight dict of one arm, with its patch groups applied directly."""
    groups = ARM_GROUPS[arm]
    return {n: patch[n] if any(GROUP_PATTERNS[g].fullmatch(n) for g in groups) else t for n, t in weights.items()}


def arm_tables(patch, arms):
    by_group = {g: {n: t for n, t in patch.items() if GROUP_PATTERNS[g].fullmatch(n)} for g in GROUP_PATTERNS}
    return {arm_index(a): {n: t for g in ARM_GROUPS[a] for n, t in by_group[g].items()} for a in arms}


def test_full_forward_matches_eager_reference_with_grouped_query_heads():
    weights = random_weights(CFG)
    ours = full_logits(GrugModel(CFG, weights), SEQ)
    assert float((ours - reference_logits(CFG, weights, SEQ)).abs().max()) < 1e-5


@pytest.mark.parametrize(("prefill_len", "chunk"), [(13, 13), (6, 6), (9, 3), (7, 5), (1, 1)])
def test_cached_decode_matches_full_forward(prefill_len, chunk):
    model = GrugModel(CFG, random_weights(CFG))
    diff = compare_logits(cached_logits(model, SEQ, prefill_len, chunk), full_logits(model, SEQ))
    assert diff["max_abs_diff"] < 1e-5 and diff["top1_agreement"] == 1.0


def test_bfloat16_cached_decode_stays_close_to_full_forward():
    model = GrugModel(CFG, random_weights(CFG, dtype=torch.bfloat16))
    diff = compare_logits(cached_logits(model, SEQ, 5, 3), full_logits(model, SEQ))
    assert not diff["nan"] and diff["top1_agreement"] >= 0.9


def test_arms_in_one_batch_equal_separate_runs():
    weights = random_weights(CFG)
    patch = perturbed(weights, CFG)
    model = GrugModel(CFG, weights, Overrides.from_arm_tables(arm_tables(patch, "ABCD")))
    ids = SEQ[:1].repeat(4, 1)
    arm = torch.tensor([arm_index(a) for a in "ABCD"])
    mixed_full = full_logits(model, ids, arm)
    mixed_cached = cached_logits(model, ids, 5, 3, arm)
    separate = [full_logits(GrugModel(CFG, arm_weights(weights, patch, a)), SEQ[:1])[0] for a in "ABCD"]
    for i in range(4):
        assert float((mixed_full[i] - separate[i]).abs().max()) < 1e-5
        assert float((mixed_cached[i] - separate[i]).abs().max()) < 1e-5
    assert float((separate[0] - separate[1]).abs().max()) > 1e-3  # the overrides change the output


def test_overrides_shared_by_two_arms_form_one_variant():
    weights = random_weights(CFG)
    overrides = Overrides.from_arm_tables(arm_tables(perturbed(weights, CFG), "ABCD"))
    gate = overrides.variants["model.layers.0.self_attn.attn_gate.weight"]
    router = overrides.variants["model.layers.0.mlp.router.weight"]
    assert [arms for arms, _ in gate] == [(1, 3)] and [arms for arms, _ in router] == [(2, 3)]


def test_override_with_wrong_shape_is_rejected():
    weights = random_weights(CFG)
    bad = {1: {"model.layers.0.self_attn.attn_gate.weight": torch.zeros(3, 3)}}
    with pytest.raises(ValueError, match="does not match"):
        GrugModel(CFG, weights, Overrides.from_arm_tables(bad))


def test_left_padded_ragged_batch_matches_single_rows():
    model = GrugModel(CFG, random_weights(CFG))
    prompts = [SEQ[0, :3].tolist(), SEQ[0, 2:10].tolist(), SEQ[1, :6].tolist()]
    cont = SEQ[1, 6:13]
    ids, pad = left_pad(prompts)
    arm = torch.zeros(3, dtype=torch.long)
    cache = model.new_cache(3, ids.shape[1] + len(cont))
    steps = [model.logits(prefill(model, cache, ids, pad, arm, chunk=3, row_block=2)).float()]
    for s, token in enumerate(cont.tolist()):
        step_ids = torch.full((3, 1), token)
        steps.append(model.logits(model.forward(step_ids, cache, ids.shape[1] + s, pad, arm)[:, -1]).float())
    batched = torch.stack(steps, dim=1)
    for r, prompt in enumerate(prompts):
        single = full_logits(model, torch.tensor([prompt + cont.tolist()]))[0, len(prompt) - 1 :]
        assert float((batched[r] - single).abs().max()) < 1e-5


def test_dropped_rows_leave_the_others_unchanged():
    model = GrugModel(CFG, random_weights(CFG))
    ids = torch.randint(0, CFG.vocab_size, (4, 14), generator=torch.Generator().manual_seed(3))
    pad, prompt_len = torch.zeros(4, dtype=torch.long), 4
    reference = cached_logits(model, ids, prompt_len, prompt_len)
    cache = model.new_cache(4, ids.shape[1])
    model.forward(ids[:, :prompt_len], cache, 0, pad)
    alive, rows = list(range(4)), {r: [] for r in range(4)}
    for s in range(prompt_len, ids.shape[1]):
        if s == 7:  # drop rows 0 and 2: row 3 moves into slot 0
            order = cache.drop([alive.index(0), alive.index(2)])
            alive = [alive[j] for j in order]
        logits = model.logits(model.forward(ids[alive, s : s + 1], cache, s, pad[: len(alive)]))
        for i, r in enumerate(alive):
            rows[r].append(logits[i, 0])
    assert alive == [3, 1] and cache.rows == 2
    for r in (1, 3):
        assert float((torch.stack(rows[r]) - reference[r, prompt_len:]).abs().max()) < 1e-5


def test_ring_entries_hold_the_last_window_slots():
    assert entry_slots(4, 3).tolist() == [0, 1, 2]
    assert entry_slots(4, 6).tolist() == [4, 5, 2, 3]
    cache = KVCache(CFG, rows=1, max_len=12, dtype=torch.float32)
    short, long = 0, CFG.long_layers[0]
    assert cache.sizes[short] == CFG.sliding_window and cache.sizes[long] == 12
    values = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1).expand(1, 10, CFG.num_kv_heads, CFG.head_dim)
    cache.write(short, slice(0, 1), 0, values[:, :7], values[:, :7])
    k, _ = cache.entries(short, slice(0, 1), 7)
    assert k[0, 0, :, 0].tolist() == [4.0, 5.0, 6.0, 3.0]  # slots 4, 5, 6 overwrote 0, 1, 2
    cache.write(long, slice(0, 1), 0, values, values)
    assert cache.entries(long, slice(0, 1), 10)[0][0, 0, :, 0].tolist() == list(map(float, range(10)))
