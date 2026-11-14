# Pre-registration (draft): attention-gate test, phase 2c

Phase 2c measures the MMLU-Pro target of phase 2's capability test a second time, with the zero-shot prompt of
Marin's Evalchemy config for the task. This document fixes the design and the rules before any phase-2c reply is
sampled. It is a draft until the lines under "Record" are filled in: `run_phase2c.sh` step c2 writes the filled
copy `prereg_phase2c.md` next to the results and logs its SHA-256 before the first run. Everything this document
does not change is as in `PREREG_phase2.md`.

## Why a second measurement

Phase 2's capability runs (2026-11-14) ended with the validity gate V3 failed. Arm A, Merge-TA0.5 as published,
answered 103 of 270 MMLU-Pro prompts correctly (0.381); V3 asks for at least 0.518, ten points below #9984's
61.83. By the pre-registered rule the comparison of arm G with arm A was not interpreted. That result stands and
is reported as it is (`results/swap_test/phase2/capability.txt`).

The MMLU-Pro prompt of phase 2 differs from the prompt of Marin's Evalchemy runs:

- Phase 2 built the prompt as Marin's canonical MMLU-Pro task does (marin 80004cc06): the instruction, up to five
  worked examples and the question in one user message. 263 of the 270 prompts held five examples.
- #9984's scores come from Marin's Evalchemy runs, and Marin's Evalchemy config for the task `mmlu-pro` sets
  `num_fewshot: 0` (`experiments/evaluation/configs/evalchemy/mmlu-pro.yaml`; the same at marin a7eeacc84 of
  2026-10-15 and at 89a267e03 of 2026-11-11). #9984's record ids name that task (for example
  `...-step254-mmlu-pro-31ab`). #9984 does not state the number of examples itself, so this is an inference
  from the config.
- The one-message layout with five examples lowers the MMLU-Pro scores of Grug checkpoints. Our comment on
  marin#10828 of 2026-11-04 (https://github.com/marin-community/marin/issues/10828#issuecomment-6629293113)
  reports 28.8% and 41.3% for two checkpoints with that layout against 77.0% and 75.6% zero-shot.

The phase-2 design should have used the zero-shot prompt. `PREREG_phase2.md` says that #9984's evaluator code is
not available; it is public (marin-community/evalchemy at ae362881), and the three prompt texts were compared
with it for phase 2c: the two CRUXEval templates of phase 2 equal Evalchemy's character for character, and
`swaptest.mmlu_pro.build_prompt` builds Evalchemy's text for all 12,032 test documents, with no example and
with five.

Phase 2c was designed after phase 2's capability results were known, arm G's among them (arm G scored 0.567 on
the five-example prompt). The decision to measure again follows from V3 and from the config above, and neither
depends on arm G's replies. The prompt is the one of that config; no other prompt was tried. Every rule below is
phase 2's.

## What changes

Only the MMLU-Pro prompts.

- Documents: the 270 MMLU-Pro documents of phase 2 (draw numbers 0 to 269 of the shuffle with seed 9984).
- Prompt: the instruction with the document's category, the question, its options lettered from A ("N/A" options
  left out) and "Answer: Let's think step by step.", with no worked example. One user message, rendered with
  Merge-TA0.5's chat template with thinking on.
- Prompt ids: 400000 + question_id (phase 2 used 300000 + question_id). The sampling noise is keyed by the prompt
  id, so it is new for these prompts and, as before, shared by the two arms.

## What does not change

- Arms: A (Merge-TA0.5 as published) and G (the full-strength merge with the 78 attention-gate tensors at B's
  values), the candidate by the runaway test. The merge spec is phase 2's file; its SHA-256 is recorded below.
- Generation: temperature 1.0, seed 9984, the same stop tokens and numerics. The cap is 4,096 new tokens unless
  the quick look below raises it for both arms.
- CRUXEval: not sampled again. The 540 CRUXEval replies of each arm from phase 2 enter the analysis (arm A passed
  V3's CRUXEval floors there: 0.715 against 0.665 and 0.719 against 0.576). They are graded again by the same
  grader; the SHA-256 of both reply files is recorded below.
- Grading: the final content after the reasoning; MMLU-Pro answers by the first match of
  `answer is \(?([A-J])\)?`, else of `Answer:\s*([A-J])`, else the last standalone capital letter A-J.
- Statistics and rules: per-target differences, the composite D = 0.25 d(CRUXEval input) + 0.25 d(CRUXEval
  output) + 0.5 d(MMLU-Pro), the paired bootstrap (10,000 draws, seed 9984, level 0.95), McNemar p-values, V3
  (the three floors 0.665, 0.576 and 0.5183, and at most 10% of arm A's 810 task replies at the cap), the
  stopping criterion (upper end of the Newcombe 95% interval of the difference in capped shares below 0.05) and
  the decision rule (better / trade-off / worse / not better / not interpreted) are those of `PREREG_phase2.md`,
  applied to the 540 CRUXEval pairs of phase 2 and the 270 MMLU-Pro pairs of phase 2c.

## Steps and gates

| step | what | gate |
|---|---|---|
| c1 | prompt files: the task prompts (phase 2's 540 CRUXEval prompts and the 270 zero-shot MMLU-Pro prompts), the run file (those 270) and the quick-look file (48 zero-shot prompts, draw numbers 800 to 847, used by nothing else) | the CRUXEval rows must equal phase 2's rows |
| c2 | this document filled in; SHA-256 logged | |
| c3 | quick look: arm A alone on the 48 prompts, cap 4,096 | S7; the cap rule |
| c4 | arm A on the 270 prompts | |
| c5 | V3 on arm A: its 540 CRUXEval replies of phase 2 and its 270 new replies | if V3 fails, stop: arm G is not sampled |
| c6 | arm G on the same 270 prompts with the same cap | |
| c7 | capability analysis and decision | |

- S7: with fewer than 21 of the 48 quick-look replies correct, phase 2c stops before the main runs (exit code
  51). A model that scores #9984's 0.618 gives fewer than 21 with probability 0.004; one at the V3 floor with
  probability 0.10; one at phase 2's 0.381 with probability 0.75.
- Cap rule: if 5 or more of the 48 quick-look replies reach 4,096 new tokens, both arms' main runs use a cap of
  8,192; otherwise 4,096. #9984 allowed 30,720 tokens, and V3 allows 10% of arm A's replies at the cap. The
  chosen cap is written to `cap.txt` before step c4 and is not changed afterwards.
- The quick look reads only arm A, the reference arm, so it cannot favour the candidate.

## What the result can say

The decision for G is one of phase 2's five outcomes. If V3 fails again in step c5, the capability question stays
open and no further measurement is planned here. Phase 2's comparison on the five-example prompt is not used as
evidence for or against G.

## Differences from #9984

Those of `PREREG_phase2.md` for the cap (4,096 or 8,192 instead of 30,720), the 270 documents drawn from all
12,032, the seeded noise and the CPU numerics. The MMLU-Pro prompt is now the zero-shot prompt of Marin's
`mmlu-pro` config.

## Reported whatever the outcome

The quick-look review; every reply and every grade; the V3 analysis of step c5; the capability report and JSON;
the cap chosen; this document as filled in and its SHA-256; every deviation.

## Record

Filled in on the machine before the quick look:

- Task prompts file SHA-256: aaabb862e1409621266ff1a1ec56b355d7e73bd40f5c49473d465b312cf955cc (task_prompts_sha256)
- MMLU-Pro run file SHA-256: 26d4a56a8444e38f650989e48a25eed009bd7b3393ca5adfd5dcdfb59701cc5d (run_prompts_sha256)
- Quick-look prompts file SHA-256: 7199f1d7563eafe08be6706a8f422a748ab34fda6ec3a249a89b2cfea0b92429 (quick_prompts_sha256)
- Merge spec SHA-256: 2950042a5085ce68285ddaa2a5edc92c6caeaf5122c761301c1b85335aa04cf0 (merge_spec_sha256)
- Harness tree SHA-256: 25db15cc128b97b987aa1122e5380163efe77bfd5ced36486553ce533e85c6fc (harness_tree_sha256)
- Phase 2 replies of arm A SHA-256: 34a6861d8b2bfcc4329f10bc35f56aae96fa5cf60b7245f54e1d770851ebf8a2 (phase2_replies_a_sha256)
- Phase 2 replies of arm G SHA-256: 696687e83be842302ecc3b37ef40e5dc6965f53e9357b465f0bf8b05a3acfd96 (phase2_replies_g_sha256)
- Runaway summary SHA-256: dd2475fc97876e1fa0489a8fd35ad039b77f97343525bbfb65cfd2fa08036570 (runaway_summary_sha256)
- MMLU-Pro revision: b189ec765aa7ed75c8acfea42df31fdae71f97be (mmlu_pro_revision)
