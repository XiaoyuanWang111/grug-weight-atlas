# Deviations from the plan

## 1. Thresholds of the stand-in validation (step 5), changed before any 67B result existed

The first run of step 5 on the machine (2026-11-13 16:13 UTC, d512 stand-in) stopped with S0. Observed:

- (b) port against the reference model in float32, 64 positions: top-1 agreement 1.0, largest logit difference
  1.04e-4. Threshold was 1e-4.
- (c) log-probability of the recorded next token: recorded by vLLM -2.5997, ours in bfloat16 -2.5492, ours in
  float32 -2.5663. Threshold was 0.05 on the bfloat16 value (observed 0.0505).
- (d) greedy replies equal to vLLM's over the first 32 tokens: 2 of 4. The other two diverge where the top two
  logits are within 0.016 and 0.084 of each other. Threshold was 3 of 4.
- (e) cached against full forward over 2,600 tokens: top-1 agreement 0.997 (passed).

Reading: the port matches the reference implementation in float32 at every position; the differences from the
vLLM record are of the size bfloat16 arithmetic produces between two implementations (our own bfloat16 and
float32 values differ by 0.017 on the same log-probability).

Change: (b) largest difference below 5e-4; (c) within 0.08; (d) at least 2 of 4 equal and no divergence at a
top-two margin of 0.1 or more. Checks (a) and (e) are unchanged. The change was made after seeing the stand-in
results and before the 67B checkpoint was loaded.

## 2. Thresholds of the 67B self-check (step 7), changed before any reply was sampled

The first run of step 7 (2026-11-13 16:24 UTC) stopped on two checks:

- cached: KV-cached decode against a full forward, 2,600 positions, top-1 agreement 0.987 (threshold 0.99).
- mixed: arm A decoded inside a batch with the other arms against arm A alone, 33 positions, top-1 agreement
  0.970, that is 32 of 33 (threshold 0.99).

The other checks passed: load and memory, the prompt template, top-1 equal to the reference model at 63 of 64
positions.

A diagnostic on the same 2,600 positions (16:28 UTC) located the 34 disagreements: 8, 4, 7, 8, 2 and 5 in the
position ranges 0-512, 512-1024, 1024-1536, 1536-2048, 2048-2300 and 2300-2600, so they do not cluster behind
the 2,048-token window or in the single-token steps. At the disagreeing positions the margin between the two
top tokens has median 0.105 (5 above 0.5, 3 above 1.0, largest 1.93). Over the five highest logits the two
computations differ by 0.17 on average (99th percentile 1.06). Two full forwards give identical logits.

Reading: the two computations are equivalent in exact arithmetic and differ in bfloat16 because chunking and
batch composition change the order of operations. This is noise of the arithmetic, not an error of the cache
or of the arm overrides (the unit tests show exact agreement in float32).

Change: cached top-1 agreement at least 0.98; mixed at least 0.95. Added secondary analysis: after the main
run, the same prompts, seed and noise are run once more at batch size 96 instead of 192. With common random
numbers the two runs can differ only through this arithmetic noise, so the replicate shows how far it moves
each arm's share of capped replies. The decision rule is applied to the main run; the replicate is reported
beside it.

## 3. Eligibility for capability scoring (phase 2), corrected before any capability reply was sampled

PREREG_phase2.md says an arm is eligible when V1 passed, its R is at most 0.15 and it shows no effect against A.
The code of `analyze_phase2.py` tested "outcome is not effect" instead, so in the run of 2026-11-13 it reported
arm N (3 of 48 capped, inconclusive against A) as eligible and as the candidate. By the pre-registered text N
is not eligible and the candidate is G (0 of 48 capped, no effect against A). The code now tests "no effect".
The runaway decisions (V2, the comparisons and the necessity outcomes) do not depend on this line. Found on
2026-11-13 22:15 UTC, before step 22.

## 4. Capability runs issued from the published tree by a driver (phase 2), fixed before any capability reply was sampled

`run_phase2.sh` stops when the harness tree differs from the one recorded in the filled pre-registration of
phase 2 (`09fbae2901d2da97e274226b9fef0be81393a3fa7dd47021fa059b6e2e4431a9`). That tree holds the eligibility
condition that deviation 3 corrects, and with it the scored arm would be N. Steps 22 to 24 therefore run from
the published tree (commit `12f2727ff362b0a19c26512940ad92f2b18bf81a`, tree
`f3f10e1aa86657f932b77de9fe2adbc8dd61caf25fa5fdd1679e93f60d3658a4`) through a driver, `run_phase2b.sh`, that
issues the commands of steps 22 to 24 of `run_phase2.sh` with that script's default settings and the scored arm
G, the candidate by the pre-registered text.

Before the first capability run the driver checks, and stops otherwise, that:

- the two trees differ in `swaptest/` only in the one eligibility line of `analyze_phase2.py`, so every module
  that generates or grades replies is byte-identical to the pre-registered tree;
- the task prompts and the merge spec have the SHA-256 recorded in `prereg_hashes.txt`, and the filled
  pre-registration has the SHA-256 recorded beside it;
- the runaway summary given to the capability analysis, `runaway_reanalysis.json` (the published code on the
  replies and teacher-forced rows of the runaway run), is byte-identical to the published file.

No other setting changes: 270 prompts per task, a cap of 4,096 new tokens, batch size 384, seed 9984,
10,000 bootstrap draws, level 0.95. Written on 2026-11-14 before step 22.

## 5. MMLU-Pro measured again with the zero-shot prompt (phase 2c), fixed before any phase-2c reply was sampled

Phase 2's capability runs ended with V3 failed: arm A answered 103 of 270 MMLU-Pro prompts correctly (0.381)
against a floor of 0.518, so the comparison of arm G with arm A was not interpreted
(`results/swap_test/phase2/capability.txt`). That result stands.

The cause is the MMLU-Pro prompt. Phase 2 put up to five worked examples and the question into one user message,
as Marin's canonical task does. Marin's Evalchemy config for `mmlu-pro`, the task that #9984's record ids name,
runs it with no examples (`num_fewshot: 0`). `PREREG_phase2c.md` gives the evidence and fixes a second
measurement of this one target with the zero-shot prompt. Arms, sampling, grading, statistics, V3 and the
decision rule are phase 2's, and phase 2's CRUXEval replies enter the analysis unchanged.

Code added for it:

- `taskprompts.py`: `--mmlu-pro-zero-shot` (prompt ids 400000 + question_id) and `--tasks`;
- `sameprompts.py`: checks that two prompt files hold the same records for some tasks;
- `analyze_phase2.py`: the capability analysis leaves out replies to prompts that its prompt file does not
  hold and takes `--arms`; the quick look judges only the tasks of its prompt file;
- `scripts/run_phase2c.sh`: the steps of phase 2c.

`run.py` and the modules it imports are unchanged, and so are the grading rules (`mmlu_pro.py` changes in its
docstring only). On phase 2's prompt file and reply files the changed capability analysis writes the same
`capability.json`, `capability_decisions.txt` and report as the run did.

Phase 2c reads phase 2's runaway summary from `runaway_reanalysis.json`, the output of the published code
(deviation 3), through the setting `RUNAWAY`. Written on 2026-11-14, after phase 2's capability analysis and
before phase 2c's quick look.
