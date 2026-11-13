"""Batched generation: mixed arms and ragged prompts in one wave equal single-row runs; stops, cap, resume."""

import torch
from helpers import STOP_IDS, perturbed, random_weights, tiny_config

from swaptest.config import ModelConfig
from swaptest.engine import Finished, Row, generate
from swaptest.model import GrugModel, Overrides
from swaptest.patch import ARM_GROUPS, GROUP_PATTERNS, arm_index
from swaptest.sampling import Sampler

CFG = ModelConfig.from_dict(tiny_config())
WEIGHTS = random_weights(CFG, seed=11)
PATCH = perturbed(WEIGHTS, CFG, scale=1.0)
TABLES = {
    arm_index(a): {n: t for n, t in PATCH.items() if any(GROUP_PATTERNS[g].fullmatch(n) for g in ARM_GROUPS[a])}
    for a in "ABCD"
}
MODEL = GrugModel(CFG, WEIGHTS, Overrides.from_arm_tables(TABLES))
G = torch.Generator().manual_seed(2)
PROMPTS = {i: torch.randint(0, 256, (n,), generator=G).tolist() for i, n in enumerate([5, 9, 3, 7])}


def run(rows, cap=24, temperature=1.0, chunk=4, row_block=3):
    done: dict[tuple[str, int], Finished] = {}
    generate(
        MODEL,
        rows,
        Sampler(9984, temperature),
        cap,
        STOP_IDS,
        lambda f: done.__setitem__(f.row.key, f),
        chunk=chunk,
        row_block=row_block,
    )
    return done


def test_one_wave_of_mixed_arms_equals_single_rows():
    rows = [Row(a, i, p) for i, p in PROMPTS.items() for a in "ABCD"]
    batched = run(rows)
    assert len(batched) == 16
    for key, f in batched.items():
        alone = run([Row(key[0], key[1], PROMPTS[key[1]])])[key]
        assert f.row.generated == alone.row.generated and f.stop_reason == alone.stop_reason


def test_stops_and_cap():
    done = run([Row(a, i, p) for i, p in PROMPTS.items() for a in "AB"], cap=40)
    reasons = [f.stop_reason for f in done.values()]
    assert set(reasons) <= {"cap", *STOP_IDS}
    for f in done.values():
        if f.stop_reason == "cap":
            assert len(f.row.generated) == 40 and not set(f.row.generated) & set(STOP_IDS)
        else:
            assert f.row.generated[-1] == f.stop_reason and not set(f.row.generated[:-1]) & set(STOP_IDS)
    assert "cap" in reasons and any(r != "cap" for r in reasons), reasons  # both kinds occur in this setup


def test_arms_share_tokens_until_their_distributions_differ():
    done = run([Row(a, 1, PROMPTS[1]) for a in "ABCD"], cap=30)
    first = done[("A", 1)].row.generated
    for arm in "BCD":
        other = done[(arm, 1)].row.generated
        assert other != first  # the overrides matter here
    # common random numbers: identical logits give identical tokens, so row A run twice agrees with itself
    assert run([Row("A", 1, PROMPTS[1])], cap=30)[("A", 1)].row.generated == first


def test_continuing_from_a_saved_prefix_reproduces_the_rest():
    full = run([Row("B", 0, PROMPTS[0])], cap=30)[("B", 0)]
    k = 7
    resumed = run([Row("B", 0, PROMPTS[0], generated=full.row.generated[:k])], cap=30)[("B", 0)]
    assert resumed.row.generated == full.row.generated and resumed.stop_reason == full.stop_reason


def test_greedy_decoding_is_deterministic():
    rows = lambda: [Row("A", i, p) for i, p in PROMPTS.items()]  # noqa: E731
    first, second = run(rows(), temperature=0.0), run(rows(), temperature=0.0)
    assert all(first[k].row.generated == second[k].row.generated for k in first)
