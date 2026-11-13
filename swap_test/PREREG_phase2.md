# Pre-registration (draft): attention-gate test, phase 2

This document fixes the design, the statistics and the decision rules of phase 2 before its runs. It is a draft
until the lines under "Record" are filled in: `run_phase2.sh` step 18 writes the filled copy `prereg_phase2.md`
next to the results and logs its SHA-256 before the runaway run starts. Any change to the design, the code, the
prompt files or the merge spec after that point is reported as a deviation. Phase 1 (`PREREG.md`) must have
finished with its validity gate V1 passed.

## Questions and hypotheses

marin#9984 merged the student B with two expert fine-tunes E (CRUXEval) and E2 (MMLU-Pro) by task arithmetic. The
half-strength merge B + 0.5 (E - B) + 0.5 (E2 - B), published as Merge-TA0.5, behaves normally; the full-strength
merge (coefficients 1.0 and 1.0, withheld) produced replies that do not stop. Phase 2 rebuilds the full-strength
merge from the public checkpoints and asks two questions.

- H_nec (necessity): in the full-strength merge, setting the 78 attention-gate tensors (per layer
  `attn_gated_norm.down_proj`, `attn_gated_norm.up_proj` and `self_attn.attn_gate`, 26 layers) back to their
  half-strength values, or to B's values, removes the failure to stop.
- H_con (a better merge): a full-strength merge with the attention gates held back that stops normally carries
  more of the two experts' skills than Merge-TA0.5, on the targets #9984 used: CRUXEval input and output
  prediction, and MMLU-Pro.

## Materials

| item | value |
|---|---|
| B | `open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92`, commit `80975627aebeaec3ef946ad609d9c1d95458ebd8` (79 shards, 134,157,838,912 bytes; the router biases are float32) |
| E | `open-athena/Snowball-67B-A2B-CruxEval-Expert-E-Step272`, commit `2eb988bb5df70378950d983ac8e70dae550cf6d5` (39 shards; all bfloat16) |
| E2 | `open-athena/Snowball-67B-A2B-MMLUPro-Expert-E2-Step254`, commit `0f22c03aff076f46c292196cf0d03a9bc4ab81c5` (39 shards; all bfloat16) |
| Merge-TA0.5 | `open-athena/Snowball-67B-A2B-Merge-TA0.5`, commit `cd0605de0ab76473b2262a9545596436e726b27d`; its `config.json`, `tokenizer.json`, `chat_template.jinja` and `generation_config.json` serve every arm |
| CRUXEval | `cruxeval-org/cruxeval`, file `test.jsonl` (800 problems), at the revision recorded below |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro`, `data/test-00000-of-00001.parquet` (12,032 documents) and `data/validation-00000-of-00001.parquet` (70, the worked examples), at the revision recorded below |
| GSM8K prompts | phase 1's prompt file (48 prompts) and arm A's replies to them from phase 1's main run |

The safetensors files of B, E and E2 at these commits have the same SHA-256 as at the commits #9984 pins
(`fdc3c3c5`, `fb4f8051`, `acb97183`).

## The merge

P = B + 1.0 (E - B) + 1.0 (E2 - B), built in memory with the arithmetic of Marin's merge tool
(`marin.merging.arithmetic.merge_tensor`, task arithmetic): per tensor, B and each expert read as float32, each
update (expert - B) multiplied by its coefficient, the updates added to zeros in donor order, B added last, the
result cast to B's stored dtype. The experts' bfloat16 router biases are widened to float32 on reading, as
Marin's tool does for allow-listed tensors. Nothing is written to disk.

Check S3, before anything else: the same code with coefficients 0.5 and 0.5 must reproduce the downloaded
Merge-TA0.5 in every stored value of every tensor (a bitwise comparison). If the only differences are router
biases and the downloaded values there equal B's (the Merge-TA0.5 card says the router bias was kept in B's
float32), the merge is rebuilt with the router-bias coefficients set to 0 and the check is repeated; it must then
pass, and the same spec builds P. Any other difference stops phase 2: P would not be the withheld merge.

## Arms

| arm | weights |
|---|---|
| A | Merge-TA0.5 as published (phase 1's reference arm) |
| P | the rebuilt full-strength merge |
| N | P with the 78 attention-gate tensors replaced by Merge-TA0.5's tensors of the same names (the half-strength values) |
| G | P with the 78 attention-gate tensors replaced by B's (the request's arm G0) |

P, N and G share every other weight, in particular every routed expert, and run in one mixed batch.

## Runaway test (necessity)

Design: phase 1's 48 GSM8K prompts, rendering, sampling and stopping, unchanged (temperature 1.0 with Gumbel noise
seeded by (9984, prompt row number, reply position) and shared by all arms and both phases; stop tokens 128009
and 128001; cap 4,096 new tokens; bfloat16 with float32 where the reference model uses it). Arms P, N and G run in
one batch of 144 rows. Arm A's replies come from phase 1.

Primary statistic: R, the share of an arm's replies that reach the cap without a stop token.

Rules, all with phase 1's comparison rule (`swaptest/stats.py`: "effect" if the difference is at least 0.25 and
the one-sided Fisher p is below 0.01; "no effect" if the upper end of the Newcombe 95% interval is below 0.15;
otherwise "inconclusive"):

- V1: phase 1's gate (R_A <= 0.15 and arm A's GSM8K accuracy >= 0.80), recomputed from arm A's replies.
- V2: R_P >= 0.50 passes (the consequence of #9984's median output of 85,000 to 95,000 characters); R_P < 0.30
  fails (the harness does not reproduce the failure, and no null result about the gates counts); in between it
  is "partial".
- Each of P, N, G against A; and P against N and P against G, where "effect" means P runs away more often.
- Necessity, for N and for G, judged only when V1 and V2 pass and P shows an effect against A:
  - **necessary**: the arm shows no effect against A and P shows an effect against the arm;
  - **not necessary**: the arm shows an effect against A and P no effect against the arm;
  - **partial**: P shows an effect against the arm, but the arm is not "no effect" against A;
  - **inconclusive**: otherwise. An inconclusive arm may be extended once by phase 1's 48 extension prompts
    (draw numbers 48 to 95), for A (if phase 1 did not extend it), P and that arm; the rule is then applied to all
    96 prompts and is final.

Secondary (reported, not used for decisions): phase 1's secondary statistics for every arm, and the
teacher-forced readout of arm A's replies under P, N and G (mean NLL; log-probability of the stop token at A's
stop position).

## Which arms get capability scores

An arm is eligible when V1 passed, its R is at most 0.15 and it shows no effect against A. The candidate is N if
eligible, else G if eligible, else none; P is scored as well when it is eligible (descriptive only: #9984 saw P's
counterpart fail in its serving stack). If no arm is eligible, the runaway result is final and no capability run
is made. Only the candidate's comparison is a test of H_con; there is one such test, so no multiplicity
correction applies.

## Capability test (a better merge)

Prompts: 270 per target, drawn once and used by every arm.
- CRUXEval: the row numbers 0 to 799 shuffled with `random.Random(9984)`; the first 270 problems; each in both
  directions (540 prompts). The prompt is the template of Marin's canonical CRUXEval task (marin 80004cc06) for
  that direction, filled with the problem's code and its input or output.
- MMLU-Pro: the test split's row numbers shuffled with `random.Random(9984)`; the first 270 documents. #9984's
  1,606-document training subset is not public, so the draw covers all 12,032 documents (87% of them were held
  out from E2's training), and the reference value is #9984's "all documents" score. The prompt is Marin's
  canonical chain-of-thought prompt: instruction with the category, the category's worked examples from the
  validation split (up to five, as many as keep the prompt text under 2,048 tokens: #9984's context of 32,768
  minus its cap of 30,720), the question, its options and "Answer: Let's think step by step."
- Each prompt is one user message, rendered with Merge-TA0.5's chat template with thinking on, as #9984 ran.

Generation: as in the runaway test (temperature 1.0, the same seeded Gumbel noise keyed by each prompt's id, the
same stop tokens), with a cap of 4,096 new tokens. Arm A runs alone from Merge-TA0.5's checkpoint; the scored
arms run together from the merge. #9984 allowed 30,720 tokens; replies at the cap are graded on their text.

Grading:
- Graded text: the final content after the reasoning (after the last `<|end_think|>`); a reply that stopped
  without final content is graded on its reasoning; a reply cut off inside its reasoning has no answer. This is
  #9984's documented MMLU-Pro rule, used for both tasks.
- CRUXEval: if the text holds `[ANSWER]`, the text after the first `[ANSWER]` up to the next `[/ANSWER]`; split
  at the first `==`. Input prediction takes the call of `f` before it, output prediction all text after it; an
  input answer without a call of `f`, or an output answer containing the call `f(<input>)`, is wrong. The program
  `<code>` + `assert <reference output> == <answer>` must run without error within 10 seconds in a fresh Python
  process and print a receipt after the assertion.
- MMLU-Pro: the first match of `answer is \(?([A-J])\)?`, else of `Answer:\s*([A-J])` (both ignoring case), else
  the last standalone capital letter A-J; correct when it equals the reference letter.

Statistics, for the candidate X against A on the same prompts:
- per target t (CRUXEval input, CRUXEval output, MMLU-Pro): d_t = accuracy of X minus accuracy of A;
- composite D = 0.25 d(CRUXEval input) + 0.25 d(CRUXEval output) + 0.5 d(MMLU-Pro): each expert's domain weighs
  one half;
- intervals: percentile intervals from 10,000 paired bootstrap draws (seed 9984) that resample prompt groups with
  replacement within each domain: CRUXEval problems (both directions of a problem together) and MMLU-Pro question
  texts. The composite interval has level 0.95; each target's interval has level 0.95. Exact two-sided McNemar
  p-values per target are reported.

Gates:
- V3 (on arm A, before any capability claim): on each target an accuracy no more than 10 points below #9984's
  Merge-TA0.5 score (CRUXEval input 76.5, output 67.6, MMLU-Pro all documents 61.83; so at least 66.5, 57.6 and
  51.83), and at most 10% of A's task replies at the cap.
- Stopping normally (for X): eligible as above, and the upper end of the Newcombe 95% interval for X's capped
  share minus A's capped share, over all task prompts, below 0.05.

Decision for X:
- **better merge than Merge-TA0.5** if V3 passes, X stops normally, the lower end of D's 95% interval is above 0,
  and no target's 95% interval lies wholly below 0;
- **trade-off** if D's lower end is above 0 but some target's interval lies wholly below 0;
- **worse** if D's upper end is below 0;
- **not better** otherwise (a null: at this size the design detects a composite gain of about 5 points with 80%
  power; a null does not exclude gains up to the upper end of D's interval);
- **not interpreted** if V3 fails or X does not stop normally.

The same statistics are reported for P when it is scored, as a description.

## Stop conditions

| stop | condition | exit code of `run_phase2.sh` |
|---|---|---:|
| S3 | the rebuilt Merge-TA0.5 differs from the download (after the router-bias rule above) | 30 |
| load | the merge, the arms' overrides or the memory check fails; or the planned caches do not fit | 31 |
| checks | the template check, cached decode against a full forward (top-1 >= 99%) or arm P alone against arm P in the mixed batch (>= 99%) fails | 32 |
| S4 | the projected runaway run (P running away) is longer than 8 hours | 33 |
| S5 | in the quick look (arm A, 6 prompts per task from draw numbers 794 to 799), more than 2 replies reach the cap, the median length exceeds 2,000 tokens, or a task has fewer than 2 correct replies | 34 |
| review | the quick-look replies have not been read and judged sound | 35 (pause) |
| S6 | the projected capability runs exceed 20 hours | 40 |
| approval | the capability runs have not been approved after the runaway result | 44 (pause) |

V3 failing ends the script with exit code 43 after the analysis is written; no eligible arm ends it with exit
code 39 after the runaway result is written.

## Differences from #9984

- Generation cap 4,096 new tokens instead of 30,720 (CPU cost); the runaway statistic is defined at this cap, and
  capability replies at the cap are graded on their text.
- MMLU-Pro is sampled from all 12,032 test documents instead of scored on the 1,606-document training subset,
  which is not public.
- #9984's evaluator code is not available; prompts and answer extraction follow Marin's canonical task code at
  marin 80004cc06 (derived from Marin's evaluation fork), except CRUXEval answer reading, which follows the
  CRUXEval harness rule above.
- Sampling uses the harness's seeded Gumbel noise instead of a GPU server's unseeded sampler; numerics are
  bfloat16 on CPU.

## Reported whatever the outcome

The merge check summary; the runaway report and JSON (every arm's statistics, V1, V2, every comparison, the
necessity labels, eligibility); the teacher-forced rows; the quick-look review; the capability report and JSON
(per-arm scores, V3, per-target differences with intervals and McNemar p, the composite with its interval, the
capped shares, the decision); every reply and every grade; the self-check summary; all logs; this document as
filled in and its SHA-256; every deviation.

## Record

Filled in on the machine before the runaway run:

- Task prompts file SHA-256: ________________________________________________________________ (task_prompts_sha256)
- Merge spec SHA-256: ________________________________________________________________ (merge_spec_sha256)
- Harness tree SHA-256: ________________________________________________________________ (harness_tree_sha256)
- GSM8K prompts file SHA-256: ________________________________________________________________ (gsm8k_prompts_sha256)
- CRUXEval revision: ________________________________________ (cruxeval_revision)
- MMLU-Pro revision: ________________________________________ (mmlu_pro_revision)
- MMLU-Pro pool: ________________________________________ (mmlu_pro_pool)
- Merge check: ________________________________________ (merge_check)

The harness tree hash is printed by `python -m swaptest.hashes --prompts PROMPTS_FILE`; it covers every file of
the harness except the two pre-registration documents and caches (`swaptest/hashes.py` defines it exactly).
