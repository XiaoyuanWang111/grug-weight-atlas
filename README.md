# Grug weight atlas

This repository measures how public checkpoints of the Snowball / Grug 67B-A2B family (Hugging Face organisation
`open-athena`) differ from each other, tensor family by tensor family. The scripts read single tensors, single routed
experts or single expert rows of the safetensors files over HTTP range requests, so no checkpoint is downloaded. The
tables in `results/` are the scripts' outputs at the commits listed below, and the atlas tables can be recomputed
offline from the per-unit records stored next to them.

Contents: [layout](#layout), [checkpoints](#checkpoints), [atlas runs](#atlas-runs),
[1. the experts and merges of marin#9984](#1-what-moved-in-the-experts-and-merges-of-marin9984-and-a-2-x-2-swap-test),
[2. the two training lines](#2-distances-between-the-two-training-lines),
[3. `dare-ties-02`](#3-what-campaign-2s-dare-ties-02-stores),
[4. BF16 rounding](#4-the-bf16-rounding-rule-and-its-check-against-marin10700), [reproduce](#reproduce).

## Layout

| path | content |
|---|---|
| `atlas/st_index.py` | shared helpers: tensor index of a checkpoint, throttled range reads, bfloat16 encode and decode, loader of an atlas run |
| `atlas/atlas2.py` | records an atlas run: per unit, the parent's norm and the Gram matrix of the children's deltas |
| `atlas/pairwise.py`, `atlas/ta_full_strength.py`, `atlas/gate_layers.py`, `atlas/expert_units.py` | tables computed from an atlas run, offline |
| `atlas/family_probe.py` | displacement by tensor family between a parent and its children |
| `atlas/gate_patch.py`, `atlas/make_blocks.py` | the 2 x 2 test of marin#9984 (section 1) |
| `atlas/dare_probe_wide.py`, `atlas/v01_probe.py` | what the Campaign 2 merge `dare-ties-02` stores, and whether the released models still hold it (section 3) |
| `atlas/grid_probe.py`, `atlas/grid_fit.py`, `atlas/ladder_check.py` | the BF16 rounding rule (section 4) |
| `atlas/pin_check.py`, `results/pin_check.txt` | the weight files at the commits Marin's issues pin against those read here |
| `results/atlas_step92/`, `atlas_lineage/`, `atlas_base12/`, `atlas_lce/` | the four atlas runs (`meta.json`, `log.txt`, `units.jsonl`) and the tables computed from them |
| `results/family_probe/` | outputs of `family_probe.py` |
| `results/dare_probe/` | outputs of `dare_probe_wide.py` and `v01_probe.py`, and the recipe parameters with the 27 rows read |
| `results/rounding/ladder_check.txt` | output of `ladder_check.py` |
| `patch_2x2/` | `manifest.json` and `build_log.txt` of the full-strength patch; the patch file is attached to the release |
| `recipes_2x2/` | `tensor_coefficients` blocks that build the two missing cells of the 2 x 2 as new merges |
| `tests/` | offline unit tests |

## Checkpoints

All repositories are in the `open-athena` organisation. `atlas2.py`, `family_probe.py`, `grid_probe.py` and
`grid_fit.py` read the `main` revision and record the commit that `main` pointed to when the tensor index was taken;
`gate_patch.py` and `dare_probe_wide.py` read at that commit.

| short name | repository | commit |
|---|---|---|
| Step12 (Base of the merging campaigns) | `Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02` | `059f83373e03d146506a4be37513d45e6ec6b126` |
| Step20 (BFCL donor of the campaigns) | `Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04` | `1515d17b22346106f9d9e549aaca65411fd70860` |
| Datakit SFT 09.21 | `Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` | `fe666478454e79fe6027a0bab0388d9abd68f51a` |
| Step92 (NUPA donor of the campaigns; B of #9984) | `Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92` | `80975627aebeaec3ef946ad609d9c1d95458ebd8` |
| RLVR2-Sync-Step116 | `Snowball-67B-A2B-10T-Mixed-RLVR2-Sync-Step116` | `bb4c56ba83c821478aece4979eaa12400404402c` |
| RLVR-Async-Step92 | `Snowball-67B-A2B-10T-Mixed-RLVR-Async-Step92` | `929fc490219ed1f75883863518220270b40d0f2a` |
| skew8 (long-context extension) | `snowball-67b-a2b-base-262k-qk175-skew8` | `3bbf729f0555965c8667600ee91c3c109ac21b19` |
| skew4 (long-context extension) | `snowball-67b-a2b-base-262k-qk175-skew4` | `58926559098171aa2e2001296921ff5ac4caac94` |
| E | `Snowball-67B-A2B-CruxEval-Expert-E-Step272` | `2eb988bb5df70378950d983ac8e70dae550cf6d5` |
| E2 | `Snowball-67B-A2B-MMLUPro-Expert-E2-Step254` | `0f22c03aff076f46c292196cf0d03a9bc4ab81c5` |
| Merge-TA0.5 | `Snowball-67B-A2B-Merge-TA0.5` | `cd0605de0ab76473b2262a9545596436e726b27d` |
| Merge-TIES-d0.5 | `Snowball-67B-A2B-Merge-TIES-d0.5` | `a8ade36ffb263fee9e49c8a508b82845464e8804` |
| MOPD-Step100 | `Snowball-67B-A2B-MOPD-Step100` | `f9c9bd62b68051442b34d8abebd2adde84873d1c` |
| CruxEval-OPD-Step75 | `Snowball-67B-A2B-CruxEval-OPD-Step75` | `a99b40686bd3f5219c05542d0cbfa7086cc5c87a` |
| CruxEval-RS-SFT-Step282 | `Snowball-67B-A2B-CruxEval-RS-SFT-Step282` | `96a38958dde26d0fa57b9d16ede2f9ab53f029e3` |
| CruxEval-Plain-SFT-Step399 | `Snowball-67B-A2B-CruxEval-Plain-SFT-Step399` | `0eaaf87090e6e1191c8db3d004fb84c4f9ba774e` |
| MixedRS-SFT-Step727 | `Snowball-67B-A2B-MixedRS-SFT-Step727` | `ec071ccd7a2abee598de45b8423291269669eb0c` |
| MixedPlain-SFT-Step803 | `Snowball-67B-A2B-MixedPlain-SFT-Step803` | `a9e46a07bab138b14f966d39b0d649ebfbb03136` |
| Campaign2-DARE-TIES-02 | `Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10` | `59931ee6b0222ff73cbd917c5a5c5ef626773aea` |
| Campaign4-Serial-Transfer-06 | `Grug-67B-A2B-Merge-Campaign4-Serial-Transfer-06-2026.10.14` | `c1475d8d7a5df97afb437f8a040fba9ef57d0583` |
| Campaign4-Adaptive-Delta-08 | `Grug-67B-A2B-Merge-Campaign4-Adaptive-Delta-08-2026.10.14` | `bbb16f06b1a53006e45da8b1ce1cfbb9328f7779` |
| Merge-NUPA-BFCL-Shared100 | `Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09` | `eabbade635ebb40871b135f87318008d0b3383c7` |
| BFCL-Recovery | `Grug-67B-A2B-BFCL-Recovery` | `4d9dd199cc9467458bd7913804876f7ddeb7f8f7` |
| v0.1 | `Marin-Snowball-67B-A2B-v0.1` | `adc0078a19a4383bc998fcdb43867b3360baf5f3` |
| v0.1.1 | `Marin-Snowball-67B-A2B-v0.1.1` | `59852d6f07261722bbec1edb6ab148317156b58e` |
| GLM53-RLVR-SFT | `Grug-67B-A2B-GLM53-RLVR-SFT-2026.09.23` | `f1aebb65b4d0f0437a6bfd3d7bea9357641b201c` |
| Datakit SFT 09.21 dr-doom | `Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` | `fd4a272b78d4c224851aaeae7f51987c5d19649c` |
| Snowflake d512 Base | `Snowflake-d512-Base-262K-2026.10.19` | `557d177660e2d700ab069c6f5713d066a645f59d` |
| Snowflake d512 SFT | `Snowflake-d512-Datakit-SFT-262K-2026.10.22` | `cbc90512edcb5e38bad11e82d46bb69c3ef98afa` |
| Snowflake d768 Base | `Snowflake-d768-Base-262K-2026.10.19` | `3b68348ab43aa87ea3ffa538ae1caa9bfd5ace38` |
| Snowflake d768 SFT | `Snowflake-d768-Datakit-SFT-262K-2026.10.22` | `7aea4a95619ac418e54ab1795e614dbb5dc54525` |

The commits of the first 23 rows are those in the `meta.json` files of the atlas runs and in `patch_2x2/manifest.json`;
those of v0.1 and v0.1.1 are printed in `results/dare_probe/v01_probe.txt`. The last six checkpoints appear only in
`results/family_probe/`, whose outputs do not record commits; their commits are those of the tensor indexes taken for
those runs.

marin#9984 and the recipes of the merging campaigns pin earlier commits of seven of these repositories (Step92 at
`fdc3c3c5`, E at `fb4f8051`, E2 at `acb97183`, Merge-TA0.5 at `07001c33`, Merge-TIES-d0.5 at `ffb5b685`, Step12 at
`50fe2f48`, Step20 at `fd07ef0f`). `pin_check.py` compares the file lists at both commits: in all seven the
safetensors files have the same SHA-256 and size, and only `README.md`, `LICENSE` and, for Step92,
`chat_template.jinja` differ (`results/pin_check.txt`).

## Atlas runs

`atlas2.py` cuts a checkpoint into units: every tensor outside the routed experts, blocks of 4,096 rows of
`embed_tokens` and `lm_head`, and one routed expert's slice of each expert tensor for a sample of experts (8 of 256 per
layer: experts 0, 32, ..., 224). For each unit it reads the parent and all children and writes one line to
`units.jsonl` with the parent's norm, each child's delta norm and share of changed stored values, the Gram matrix of
the children's deltas (inner products accumulated in float64) and, per pair of children, the count of equal stored
values. Distances between any two checkpoints of a run follow from the Gram matrix:
|a - b|^2 = |a - p|^2 + |b - p|^2 - 2 <a - p, b - p> for the parent p.

| run | parent | children | layers | units | embedding and head rows |
|---|---|---|---|---:|---|
| `atlas_step92` | Step92 | MOPD-Step100, CruxEval-OPD-Step75, CruxEval-RS-SFT-Step282, CruxEval-Plain-SFT-Step399, E, MixedRS-SFT-Step727, MixedPlain-SFT-Step803, E2, Merge-TIES-d0.5, Merge-TA0.5 | all 26 | 1,110 | all |
| `atlas_lineage` | Step12 | Step92, Step20, Datakit SFT 09.21, RLVR2-Sync-Step116, RLVR-Async-Step92 | 12, 0, 25 | 128 | last 1,280 |
| `atlas_base12` | Step12 | Step92, Step20, Merge-NUPA-BFCL-Shared100, Campaign2-DARE-TIES-02, Campaign4-Serial-Transfer-06, Campaign4-Adaptive-Delta-08, BFCL-Recovery | 0, 12, 25 | 128 | last 1,280 |
| `atlas_lce` | Step12 | skew8, skew4, Datakit SFT 09.21, Step92 | 0, 12, 25 | 128 | last 1,280 |

`pairwise.py` prints, per group of tensors, the distance matrix in percent of the parent's norm in that group and the
share of stored scalars that differ. It weights each sampled routed expert by 32 so that the routed experts count at
their true size in the group "all".

`atlas_step92` was recorded by an earlier version of `atlas2.py` that wrote the same fields except the step histogram
and the count of equal stored values per pair of children. In `results/atlas_step92/pairwise.txt` the child-to-child
entries of the "stored scalars that differ" matrices therefore read 100.00 and carry no information; the
parent-to-child entries are measured. In `results/atlas_lce/pairwise.txt` the columns of skew8 and skew4 are both
labelled `open-athena/snow`, because their repository names do not match the name-shortening rule of `pairwise.py`;
they appear in the order of `meta.json`, skew8 first.

## 1. What moved in the experts and merges of marin#9984, and a 2 x 2 swap test

**Question.** marin#9984 merged B + a (E - B) + b (E2 - B) with B = Step92 and two expert fine-tunes of it, E and E2
(full SFT with AdamH at learning rate 1e-5). It published Merge-TA0.5 (a = b = 0.5) and Merge-TIES-d0.5 (density 0.5,
a = b = 1), and withheld the merge at a = b = 1.0, which scored GSM8K 50.0, GPQA-D 30.0 and IFEval 73.3 with unusually
long outputs (median 85k to 95k characters). Which tensor families did E and E2 move, where do the three merges sit,
and how can the moved families be swapped between the merges?

**Method.** `atlas_step92` holds every tensor outside the routed experts in full and 8 of 256 routed experts per
layer, for Step92 and ten children. `ta_full_strength.py` sums the units per tensor family and gives |x - B| / |B| for
each checkpoint; for the withheld merge it computes |dE + dE2|^2 = |dE|^2 + |dE2|^2 + 2 <dE, dE2> from the two expert
deltas, before rounding. `gate_layers.py` gives the same per layer for the attention GatedNorm and gate.

Displacement |x - B| / |B| in percent of Step92's norm in each family (selected rows of
`results/atlas_step92/ta_full_strength.txt`):

| tensor family | scalars | E | E2 | Merge-TA0.5 | Merge-TIES-d0.5 | a = b = 1.0 (computed) | MOPD-Step100 | MixedRS-SFT-Step727 | cosine dE, dE2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_gated_norm.up_proj.weight` | 8,519,680 | 3.449 | 3.434 | 2.871 | 3.922 | 5.729 | 0.140 | 0.391 | 0.386 |
| `attn_gated_norm.down_proj.weight` | 8,519,680 | 2.874 | 2.785 | 2.315 | 3.264 | 4.614 | 0.264 | 0.466 | 0.329 |
| `self_attn.attn_gate.weight` | 1,331,200 | 1.327 | 1.332 | 1.122 | 1.513 | 2.218 | 0.085 | 0.188 | 0.391 |
| `mlp.router.weight` | 17,039,360 | 0.166 | 0.153 | 0.132 | 0.191 | 0.239 | 0.008 | 0.031 | 0.118 |
| `lm_head.weight` | 328,335,360 | 0.128 | 0.104 | 0.108 | 0.138 | 0.194 | 0.084 | 0.490 | 0.379 |
| `mlp_gated_norm.down_proj.weight` | 8,519,680 | 0.072 | 0.070 | 0.067 | 0.086 | 0.116 | 0.146 | 0.428 | 0.337 |
| `mlp_gated_norm.up_proj.weight` | 8,519,680 | 0.070 | 0.067 | 0.064 | 0.084 | 0.107 | 0.122 | 0.398 | 0.233 |
| `self_attn.o_proj.weight` | 170,393,600 | 0.072 | 0.071 | 0.069 | 0.087 | 0.119 | 0.126 | 0.369 | 0.382 |
| `self_attn.q_proj.weight` | 170,393,600 | 0.060 | 0.058 | 0.055 | 0.073 | 0.092 | 0.127 | 0.400 | 0.207 |
| `shared_expert.up_proj.weight` | 170,393,600 | 0.069 | 0.066 | 0.063 | 0.083 | 0.107 | 0.123 | 0.405 | 0.259 |
| `mlp.experts.gate_proj.weight` (8 of 256 experts) | 681,574,400 | 0.050 | 0.044 | 0.043 | 0.060 | 0.068 | 0.098 | 0.407 | 0.055 |

- The attention GatedNorm projections and the attention gate (78 tensors, 18,370,560 scalars, 0.027% of the model)
  moved 1.33% to 3.45% in E and E2. The attention, shared-expert and routed-expert matrices and the MLP GatedNorm
  moved 0.04% to 0.07%.
- The RMS of the stored change per scalar is about the same in the attention GatedNorm (2.8e-4, 3.4e-4), the attention
  gate (3.2e-4), the router weight (2.1e-4) and `embed_norm` (2.5e-4), and 5e-6 to 7e-6 in the weight matrices.
- In Marin's main branch at `a5e98ff2b`, `GrugMoeAdamHConfig.create_mask` (`experiments/grug/moe/optimizer.py:198`)
  puts a parameter into the plain-Adam group when `"attn_gate" in path_lower`. The attention GatedNorm parameters
  (`attn_gated_norm.w_*`) contain that substring and the MLP GatedNorm parameters do not (the substring issue of
  marin#5639), so the attention GatedNorm takes Adam's absolute steps and the MLP GatedNorm takes AdamH's relative
  ones. The training configuration of E and E2 was not read; #9984 names AdamH, and the equal absolute change per
  scalar across the plain-Adam tensors is what this mask would produce.
- Checkpoints trained another way do not show the difference: MOPD-Step100 (MarinSkyRL) moved the attention GatedNorm
  0.140% / 0.264% (up / down projection) and the MLP GatedNorm 0.122% / 0.146%; MixedRS-SFT-Step727 (AdamW at 1e-6)
  0.391% / 0.466% against 0.398% / 0.428%.
- The two expert deltas point partly the same way in the three families (cosine 0.33 to 0.39; 0.06 in the routed
  experts). Task arithmetic at a = b = 1.0 holds the shared part twice and moves the three families 5.73% / 4.61% /
  2.22%, up to 8.016% in `attn_gated_norm.up_proj` of layer 21 (`gate_layers.txt`). TIES averages the sign-consistent
  updates by their coefficient mass, so where both experts agree it holds the mean of the two deltas, and where one
  survives it holds that delta in full: 3.92% / 3.26% / 1.51%. In the attention and expert matrices a = b = 1.0 sits
  0.07% to 0.12% from B, no farther than MOPD-Step100 (0.10% to 0.13%).
- Not established: that the displacement of these three families causes the long outputs of the withheld merge.

**The 2 x 2 test.** Two swaps between the merges of #9984 separate the three families from the rest:

| | rest of the model at 0.5 | rest of the model at 1.0 |
|---|---|---|
| three families at 0.5 | Merge-TA0.5 (public) | cell C: the withheld merge with the 78 tensors of Merge-TA0.5 |
| three families at 1.0 | cell D: Merge-TA0.5 with the 78 tensors of the patch | the withheld merge (`ta-a1.0-b1.0`) |

`gate_patch.py` reads the 78 tensors of the three families and the 26 router weights (another 2-D family of the
plain-Adam group) from B, E, E2, Merge-TA0.5 and Merge-TIES-d0.5. It recomputes the half-strength merge with the
arithmetic of `marin.merging.arithmetic.merge_tensor` (float32, each update scaled, then summed, base added last, cast
to bfloat16): in all 35,409,920 stored values of the 104 tensors it equals the public Merge-TA0.5. The recomputed TIES
merge differs from the public Merge-TIES-d0.5 in some values; that is expected, because the top-k cut has ties among
bfloat16 differences and `torch.topk` breaks them in its own order. It then writes the 104 tensors at a = b = 1.0.
Its output is `patch_2x2/build_log.txt`; the displacement columns are for the stored, rounded values:

```
family                                  scalars  TA0.5 differ  TIES differ |      E     E2  TA0.5   TIES   full   worst layer
attn_gated_norm.down_proj.weight      8,519,680             0      236,537 |   2.87   2.79   2.32   3.26   4.61     6.61 L21
attn_gated_norm.up_proj.weight        8,519,680             0      232,205 |   3.45   3.43   2.87   3.92   5.73     8.02 L21
self_attn.attn_gate.weight            1,331,200             0       66,162 |   1.33   1.33   1.12   1.51   2.22     5.92 L24
mlp.router.weight                    17,039,360             0    1,173,940 |   0.17   0.15   0.13   0.19   0.24     1.99 L25
```

The patch file `gates_full_strength.safetensors` is attached to the release of this repository and is not in git:
104 bfloat16 tensors, 70,831,936 bytes, SHA-256
`e8c7387ac84f01f3f1b1569397db07f99b9a4221ec78d411e86d84aa022b5946`. Each tensor is
bfloat16(float32(B) + 1.0 (float32(E) - float32(B)) + 1.0 (float32(E2) - float32(B))). `patch_2x2/manifest.json`
lists the source commits, the tensor names in two groups (`attention_gates`, 78 tensors, for cell D;
`router_weights`, 26 tensors, to swap the router weights as well) and the checks above. Cell D is Merge-TA0.5 at
commit `cd0605de0ab76473b2262a9545596436e726b27d` with every tensor of the `attention_gates` group replaced by the
patch's tensor of the same name. To rebuild the file:

```
python atlas/gate_patch.py > patch_2x2/build_log.txt
sha256sum patch_2x2/gates_full_strength.safetensors
```

`recipes_2x2/gate_half.json` (cell C) and `recipes_2x2/gate_full.json` (cell D) build the two cells as new merges
instead: each is the value of the recipe key `tensor_coefficients` of the #9974 tools (coefficients over the donors
E, E2 for the 78 tensors), to be added to the recipe of the two existing merges with default coefficients [1.0, 1.0]
(cell C) or [0.5, 0.5] (cell D). `python atlas/make_blocks.py` writes them.

**Commands.**

```
python atlas/atlas2.py --out results/atlas_step92 --parent open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 \
  --children open-athena/Snowball-67B-A2B-MOPD-Step100 open-athena/Snowball-67B-A2B-CruxEval-OPD-Step75 \
  open-athena/Snowball-67B-A2B-CruxEval-RS-SFT-Step282 open-athena/Snowball-67B-A2B-CruxEval-Plain-SFT-Step399 \
  open-athena/Snowball-67B-A2B-CruxEval-Expert-E-Step272 open-athena/Snowball-67B-A2B-MixedRS-SFT-Step727 \
  open-athena/Snowball-67B-A2B-MixedPlain-SFT-Step803 open-athena/Snowball-67B-A2B-MMLUPro-Expert-E2-Step254 \
  open-athena/Snowball-67B-A2B-Merge-TIES-d0.5 open-athena/Snowball-67B-A2B-Merge-TA0.5
python atlas/ta_full_strength.py results/atlas_step92 > results/atlas_step92/ta_full_strength.txt
python atlas/gate_layers.py results/atlas_step92 > results/atlas_step92/gate_layers.txt
python atlas/pairwise.py results/atlas_step92 > results/atlas_step92/pairwise.txt
```

**Limits.** Routed experts are a sample of 8 of 256 per layer; every other tensor is read in full. The a = b = 1.0
column of the first table is computed from the expert deltas before rounding; the patch holds the stored, rounded
values. The patch holds the values of the withheld merge in these tensors only if that merge was built with the same
arithmetic, which #9984 states ("#9974 tools, B anchor"); its weights are not public. Neither swap of the 2 x 2 has been
run, so the link between the three families and the long outputs is untested.

## 2. Distances between the two training lines

**Question.** How far apart are Step12 (Base of the merging campaigns) and Step92 (their NUPA donor), and how much of
that distance was already there before post-training? The family tree on Marin's main branch
(`infra/marina/apps/snowball_family_tree/graph-data.json` at `a5e98ff2b`) puts Step12 under Datakit SFT 09.21 and the
skew8 long-context extension, and Step92 under Datakit SFT 09.11 and the skew4 long-context extension. The two
extensions are sibling runs of 1,000 updates each from the final cooldown of the shared pretraining run (step 156,000
to 157,000), and the tree names the public skew8 and skew4 repositories as their models. Step20 starts from the
public Step12, and Step92, RLVR-Async-Step92 and RLVR2-Sync-Step116 all start from Datakit SFT 09.11.

**Method.** Two atlas runs with parent Step12 on the same units (layers 0, 12 and 25; every tensor of those layers
outside the routed experts, 8 of 256 routed experts per layer, the last 1,280 rows of `embed_tokens` and `lm_head`),
and `pairwise.py` on each. Distances are in percent of Step12's norm in each group; the last column is the share of
stored routed-expert scalars that differ.

| pair | routed experts | attention q, k, v, o | shared expert | routed scalars that differ | source |
|---|---:|---:|---:|---:|---|
| skew8, skew4 | 3.01 | 3.06 | 2.74 | 92.95% | `atlas_lce` |
| skew8, Step12 | 1.63 | 1.50 | 1.40 | 71.61% | `atlas_lce` |
| skew4, Step92 | 2.31 | 1.89 | 1.71 | 80.11% | `atlas_lce` |
| skew8, Step92 | 3.78 | 3.55 | 3.19 | 93.67% | `atlas_lce` |
| skew4, Step12 | 3.41 | 3.38 | 3.05 | 93.52% | `atlas_lce` |
| Step12, Step92 | 4.01 | 3.75 | 3.39 | 93.88% | both |
| Step12, Datakit SFT 09.21 | 0.07 | 0.16 | 0.16 | 20.07% | both |
| Step12, Step20 | 0.15 | 0.20 | 0.20 | 33.18% | `atlas_lineage` |
| Step92, RLVR2-Sync-Step116 | 0.21 | 0.27 | 0.25 | 34.41% | `atlas_lineage` |
| Step92, RLVR-Async-Step92 | 0.60 | 0.64 | 0.64 | 70.96% | `atlas_lineage` |

- The two lines were already 3% apart after the long-context extension, and each line's post-training added 1.4% to
  2.3%. In the routed experts, 3.01^2 + 1.63^2 + 2.31^2 = 17.1 against 4.01^2 = 16.1: the gap between the two
  extensions is a bit over half of the squared
  distance between Step12 and Step92, and the two post-training legs the rest (slightly correlated).
- The two RLVR arms that started from Datakit SFT 09.11 are 0.60% apart in the routed experts. If they moved
  independently and equally far, each moved about 0.42% (0.598 / sqrt 2), and Step92's own RLVR training is about
  (0.42 / 4.01)^2 = 1% of the energy of Step92 - Step12. Without that assumption, at least one arm moved 0.30% or more.
- `family_probe.py` reads the GatedNorms, the attention gate and the router weight in all 26 layers, and q_proj,
  o_proj, `shared_expert.up_proj` and routed expert 0 of `gate_proj` in layers 0, 13 and 25
  (`results/family_probe/probe_family_skew8_all.txt`). skew8 against skew4: q_proj 3.19%, o_proj 2.93%,
  `shared_expert.up_proj` 2.74%, routed expert `gate_proj` 2.49%, with 92% to 94% of stored values different.
  skew8 against Datakit SFT 09.21: 1.4% to 1.5% for these matrices (routed 0.52%); skew8 against Step92: 3.2% to 3.7%
  (routed 2.62%); skew4 against Step92: 1.7% to 1.9% (routed 0.87%; `probe_family_skew4_step92.txt`).
- Per sampled expert (`results/atlas_base12/expert_units_Sync-Step92.txt`, same units), Step92 is 2.01% to 10.97%
  from Step12 over the 72 units (pooled 4.009%, mean 3.672%); expert 160 of layer 12 is at 10.15% to 10.97%.

**Commands.**

```
python atlas/atlas2.py --out results/atlas_lineage --parent open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  --children open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 \
  open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04 open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 \
  open-athena/Snowball-67B-A2B-10T-Mixed-RLVR2-Sync-Step116 open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Async-Step92 \
  --layers 12,0,25 --embed-blocks last
python atlas/atlas2.py --out results/atlas_lce --parent open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  --children open-athena/snowball-67b-a2b-base-262k-qk175-skew8 open-athena/snowball-67b-a2b-base-262k-qk175-skew4 \
  open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 \
  --layers 0,12,25 --embed-blocks last
python atlas/pairwise.py results/atlas_lineage > results/atlas_lineage/pairwise.txt
python atlas/pairwise.py results/atlas_lce > results/atlas_lce/pairwise.txt
python atlas/expert_units.py results/atlas_base12 Sync-Step92 > results/atlas_base12/expert_units_Sync-Step92.txt
python atlas/family_probe.py open-athena/snowball-67b-a2b-base-262k-qk175-skew8 \
  open-athena/snowball-67b-a2b-base-262k-qk175-skew4 open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 \
  open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  > results/family_probe/probe_family_skew8_all.txt
```

The other files in `results/family_probe/` come from the same script with these arguments (parent first):

| file | parent | children |
|---|---|---|
| `probe_family_skew4_step92.txt` | skew4 | Step92 |
| `probe_family_skew8_sft0921.txt` | skew8 | Datakit SFT 09.21 |
| `probe_family_sft0921_children.txt` | Datakit SFT 09.21 | GLM53-RLVR-SFT, Datakit SFT 09.21 dr-doom, Step12 |
| `probe_family_snowflake512.txt` | Snowflake d512 Base | Snowflake d512 SFT |
| `probe_family_snowflake768.txt` | Snowflake d768 Base | Snowflake d768 SFT |

**Limits.** The atlas rows rest on three layers and on 8 of 256 routed experts per layer; the family probe on routed
expert 0 of three layers. The 1% figure assumes that the two RLVR arms moved independently and equally far. Datakit
SFT 09.11, the parent of Step92, is not public, so Step92's own task vector, Step92 - Datakit SFT 09.11, is not
measured here.

## 3. What Campaign 2's `dare-ties-02` stores

**Question.** Campaign 2 merged the two donors Step92 and Step20 into Base (Step12) under budgets on the number of
changed stored values. The recipe `dare-ties-02` has method `dare_ties`, coefficients [0.5, 0.5], density 0.2,
scale 1, seed 0 and ties_density 0.2, and its support is 1,705 routed experts (5,115 expert rows over 78 tensors).
Campaign 2 reports for this merge 6.04% of Base scalars changed, NUPA200 0.00 and BFCL-Parity 45.53. What do the
changed values hold?

**Method.** `dare_probe_wide.py` takes, for each of the three routed-expert projections in layers 0, 12 and 25, the
first two expert rows of the recipe's support and the first row outside it (27 rows of 3,276,800 scalars). It reads
each row from Base, both donors and the merge at the recorded commits, and tests every changed value for bit-exact
equality with bfloat16(Base + k (donor - Base)) for k = 1 / density = 5 and either donor, then with Base + 5 x the mean
of both deltas, then with other multiples of one delta (2.5, 1, 0.5, 10).

Changed values in the 18 supported rows (`results/dare_probe/dare_probe_wide.txt`):

| stored value | count | share |
|---|---:|---:|
| Base + 5 x (Step92 - Base) | 10,745,502 | 77.01% |
| Base + 5 x (Step20 - Base) | 2,869,608 | 20.57% |
| Base + 5 x the mean of both deltas | 338,630 | 2.43% |
| other multiples of one delta | 0 | 0.00% |
| unexplained | 0 | 0.00% |
| all changed values | 13,953,740 of 58,982,400 | 23.66% of the scalars |

- The nine rows outside the support are identical to Base.
- The 0.5 coefficients do not scale the stored update. DARE divides the kept 20% by the density, and the TIES step
  divides the sign-consistent sum by the coefficient mass, which cancels the 0.5 wherever one donor survives.
- In each supported row 22.42% to 24.55% of the values changed, and the row sits 3.63% to 10.51% from Base. The same
  rows of Step92 are 1.63% to 4.74% from Base, those of Step20 0.07% to 0.16%.
- In `atlas_base12` (24 sampled experts in layers 0, 12 and 25), Campaign2-DARE-TIES-02 changed 5.10% of the sampled
  routed-expert scalars and sits 2.97% from Base there; the cosine of its delta is 0.15 with Step92 - Step12 and 0.006
  with Step20 - Step12. Campaign4-Adaptive-Delta-08 changed 5.60% and sits 0.18% from Base; Step20 itself changed
  33.18% and sits 0.15% from Base.
- The released models still hold the written values (`results/dare_probe/v01_probe.txt`; six rows: `gate_proj` of
  layers 0, 12 and 25, the first two support rows each). At the positions where the merge holds
  Base + 5 x (Step92 - Base), 17.51% to 18.89% of each row, v0.1 (bfloat16) equals the merge in 71.23% to 96.94% of
  the values; |v0.1 - merge| / |merge - Base| there is 0.0004 to 0.0057, and the regression coefficient of
  (v0.1 - Base) on (merge - Base) is 1.0000. For v0.1.1 (float32) the ratio is 0.0042 to 0.0202 and the coefficient
  0.9999 to 1.0000. On the positions of the same rows that the merge left at Base, v0.1 equals Base in 73.40% to
  97.44% of the values and moved 0.0044% to 0.0916% of their norm; v0.1.1 moved 0.0997% to 0.1828%. So v0.1
  descends from the merge, and the written 5 x deltas are still in both releases.

`results/dare_probe/dare-ties-02_probed_rows.json` records the recipe's `parameters`, the 27 (tensor, row) pairs read
and the source of the recipe. The recipe is `campaign2-recipes/dare-ties-02.json` of the dataset
`open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09` at commit
`1ef199578aeec9f08e142a8dd760d44658264be4` (31,562 bytes, SHA-256
`76604818afce91d78bdd1e1767a71f7663a3adeb96f3b8678f1ea735e72276b5`). It is not copied into this repository; to
fetch that one file and run the probe:

```
curl -L -o dare-ties-02.json https://huggingface.co/datasets/open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09/resolve/1ef199578aeec9f08e142a8dd760d44658264be4/campaign2-recipes/dare-ties-02.json
sha256sum dare-ties-02.json
python atlas/dare_probe_wide.py dare-ties-02.json > results/dare_probe/dare_probe_wide.txt
python atlas/v01_probe.py dare-ties-02.json open-athena/Marin-Snowball-67B-A2B-v0.1 \
  open-athena/Marin-Snowball-67B-A2B-v0.1.1 > results/dare_probe/v01_probe.txt
```

`dare_probe_wide.py` also writes `results/dare_probe/dare_probe_wide.json` (per row: counts by category, changed
values, displacements). The `atlas_base12` run:

```
python atlas/atlas2.py --out results/atlas_base12 --parent open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  --children open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 \
  open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04 \
  open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09 open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10 \
  open-athena/Grug-67B-A2B-Merge-Campaign4-Serial-Transfer-06-2026.10.14 \
  open-athena/Grug-67B-A2B-Merge-Campaign4-Adaptive-Delta-08-2026.10.14 open-athena/Grug-67B-A2B-BFCL-Recovery \
  --layers 0,12,25 --embed-blocks last
python atlas/pairwise.py results/atlas_base12 > results/atlas_base12/pairwise.txt
```

**Limits.** 18 of the 5,115 supported expert rows were read, in three layers, and 6 of them in the released models.
The classification tests a fixed list of multiples; every changed value matched one of them. The scores and the
6.04% are Campaign 2's own numbers, not measured here. Campaign 4 reports 9.47% changed for Adaptive-Delta-08; the
sample of 24 experts is too small for per-recipe shares, so the 5.60% above is not an estimate of it.

## 4. The BF16 rounding rule and its check against marin#10700

**Question.** How much of a fine-tune's update survives in a BF16 checkpoint?

**Rule.** The stored child is round_to_BF16(stored parent + u), with u drawn from N(0, sigma^2) for every scalar, one
sigma per checkpoint, the same for small and large weights. A stored value changes only when |u| exceeds half a BF16 step
at that weight; in the binade [2^e, 2^(e+1)) the step is 2^(e-7).

**Method.** `grid_fit.py` fits sigma by maximum likelihood to the share of changed values in each binade of
`model.layers.12.self_attn.q_proj.weight` (6,553,600 values, 14 binades), parent Step92, and reports under the fitted
rule the share of updates rounded away (erased), the regression coefficient of the stored difference on the update
(kept) and their cosine. `grid_probe.py` prints the observed changes per binade (share, grid steps, direction).
`ladder_check.py` fits sigma to one number of #10700, "69.6% erased at alpha = 1", and predicts the other points of
its ladder.

| child of Step92 | sigma | kept | cosine | erased |
|---|---:|---:|---:|---:|
| MOPD-Step100 | 1.29e-5 | 0.66 | 0.72 | 69% |
| CruxEval-OPD-Step75 | 1.25e-5 | 0.65 | 0.71 | 70% |
| CruxEval-RS-SFT-Step282 | 2.54e-5 | 0.90 | 0.87 | 50% |
| MixedPlain-SFT-Step803 | 4.16e-5 | 0.98 | 0.94 | 36% |
| E | 7.76e-6 | 0.46 | 0.60 | 80% |
| E2 | 7.63e-6 | 0.46 | 0.59 | 80% |

- For the six children the largest gap between the observed and the predicted share of changed values in a binade is
  0.11 to 0.59 points.
- The alternative "both checkpoints are rounded copies of float32 masters" predicts 17% changed in the binade 2^-7 for
  MOPD-Step100; the observed share is 2.07%. So each child started from its parent's stored BF16 values.
- Step12 against its donors, same tensor: Step20 sigma 1.99e-5, 43.1% changed, largest gap 0.72 points; Step92 sigma
  3.48e-4, 95.1% changed. Routed expert 0 of `model.layers.12.mlp.experts.gate_proj.weight` (`--slice 0:3276800`):
  Step20 sigma 1.38e-5, 30.9% changed, kept 0.67; Step92 sigma 3.22e-4, 94.6% changed.
- In `atlas_step92` the eight fine-tunes of Step92 differ from it in 15.06% (E2) to 61.18% (MixedPlain-SFT-Step803)
  of all stored scalars (`results/atlas_step92/pairwise.txt`, group "all").

`results/rounding/ladder_check.txt` against #10700 (65,695,399,644 of 67,078,882,816 entries with a nonzero delta):

| alpha | changed % of all entries | erased % of the real deltas | kept | effective coefficient (alpha x kept) | relative L2 error % |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 8.28 | 91.55 | 0.206 | 0.051 | 92.80 |
| 0.5 | 16.13 | 83.53 | 0.386 | 0.193 | 85.64 |
| 0.75 | 23.35 | 76.16 | 0.537 | 0.403 | 78.72 |
| 0.875 | 26.66 | 72.78 | 0.596 | 0.522 | 75.49 |
| 0.9375 | 28.25 | 71.16 | 0.624 | 0.585 | 73.93 |
| 1.0 | 29.77 | 69.60 | 0.650 | 0.650 | 72.41 |

The sigma fitted to "69.6% erased at alpha = 1" is 1.267e-5. #10700 reports at alpha = 0.5: 16.15% of all entries
changed, 83.51% erased and a relative L2 error of 84.46%.

**Commands.**

```
python atlas/grid_fit.py model.layers.12.self_attn.q_proj.weight open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 \
  open-athena/Snowball-67B-A2B-MOPD-Step100 open-athena/Snowball-67B-A2B-CruxEval-OPD-Step75 \
  open-athena/Snowball-67B-A2B-CruxEval-RS-SFT-Step282 open-athena/Snowball-67B-A2B-MixedPlain-SFT-Step803 \
  open-athena/Snowball-67B-A2B-CruxEval-Expert-E-Step272 open-athena/Snowball-67B-A2B-MMLUPro-Expert-E2-Step254
python atlas/grid_fit.py model.layers.12.self_attn.q_proj.weight open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04 open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92
python atlas/grid_fit.py --slice 0:3276800 model.layers.12.mlp.experts.gate_proj.weight \
  open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  open-athena/Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04 open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92
python atlas/grid_probe.py model.layers.12.self_attn.q_proj.weight open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92 \
  open-athena/Snowball-67B-A2B-MOPD-Step100
python atlas/ladder_check.py > results/rounding/ladder_check.txt
```

**Limits.** The fit uses one tensor, `q_proj` of layer 12 (and one routed expert for Step12's donors). The output of
`grid_fit.py` is not in `results/`; the values of its table were recorded when it ran and can be checked with the
commands above (13.1 MB read per checkpoint). `ladder_check.py` takes the share of weights per binade from that tensor
of Step92, not from the checkpoints of #10700, and fits sigma to a single reported number.

## Reproduce

Python 3.12 or later; the scripts need only `numpy`, and the tests need `pytest`. The offline tables were recomputed
with Python 3.12 and numpy 2.5. Run every command from the repository root. The scripts that read the network write
their tensor indexes to `index/` (not in git) and start at most one request every 0.2 s; the first index of a
repository costs one API request and two range requests per safetensors file.

| script | network read | reading time in `log.txt` |
|---|---|---|
| `atlas2.py`, `atlas_step92` (11 checkpoints, 1,110 units) | 81.2 GB | 79.1 GB in 127.5 min (about 10 MiB/s), the last 1.7 GB in 6.0 min after a restart; first indexing of the 11 repositories 25 min |
| `atlas2.py`, `atlas_lineage` (6 checkpoints, 128 units) | 4.3 GB | 13.9 min |
| `atlas2.py`, `atlas_base12` (8 checkpoints, 128 units) | 5.7 GB | 15.6 min |
| `atlas2.py`, `atlas_lce` (5 checkpoints, 128 units) | 3.6 GB | 8.1 min |
| `family_probe.py` | 122,572,800 values, about 245 MB, per Grug 67B checkpoint | not recorded |
| `gate_patch.py` | 520 requests, about 354 MB | not recorded |
| `dare_probe_wide.py` | 108 requests of 6,553,600 bytes, about 708 MB | not recorded |
| `v01_probe.py` | per later checkpoint 24 requests (6 rows of 4 repositories): about 157 MB for a bfloat16 checkpoint, 197 MB for a float32 one | not recorded |
| `grid_probe.py`, `grid_fit.py` | 13.1 MB per checkpoint for `q_proj` of layer 12 | not recorded |
| `pin_check.py` | 14 API requests, no file content | not recorded |
| `pairwise.py`, `ta_full_strength.py`, `gate_layers.py`, `expert_units.py`, `ladder_check.py`, `make_blocks.py` | none | |

`atlas2.py` appends to `units.jsonl` and skips the units already there, so a new run needs a new output directory.
`atlas_step92` was recorded by the earlier version described under [atlas runs](#atlas-runs);
`atlas2.py --experts-per-layer 8 --layers all --embed-blocks all` reads the same units.

Offline, from the committed atlas runs:

```
python atlas/pairwise.py results/atlas_step92 > results/atlas_step92/pairwise.txt
python atlas/pairwise.py results/atlas_lineage > results/atlas_lineage/pairwise.txt
python atlas/pairwise.py results/atlas_base12 > results/atlas_base12/pairwise.txt
python atlas/pairwise.py results/atlas_lce > results/atlas_lce/pairwise.txt
python atlas/ta_full_strength.py results/atlas_step92 > results/atlas_step92/ta_full_strength.txt
python atlas/gate_layers.py results/atlas_step92 > results/atlas_step92/gate_layers.txt
python atlas/expert_units.py results/atlas_base12 Sync-Step92 > results/atlas_base12/expert_units_Sync-Step92.txt
python atlas/ladder_check.py > results/rounding/ladder_check.txt
python -m pytest
```

The output of each command equals the committed file. The tests check the bfloat16 encoder against an exact
reference, the task arithmetic and TIES of `gate_patch.py` on hand-checked arrays, the safetensors writer, and the
distance arithmetic of `pairwise.py`. Lint: `ruff check` and `ruff format --check` with the settings in
`pyproject.toml`.

## License

Apache-2.0; see `LICENSE`.
