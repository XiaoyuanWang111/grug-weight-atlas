# Closure probe pre-registration

Written 2026-11-13 (US Pacific; 2026-11-14 UTC) before any reply exists. Frozen with the files listed in
`prereg_hashes.txt` (SHA-256); `analyze_probe.py` refuses to run when any of them differs, and checks that the
worktree `<marin worktree>` is at 80004cc06 with no local change. A change after the freeze goes
to `DEVIATIONS.md` with its reason. Design and evidence: PLAN.md.

## 1. Sample and sampling

- Model: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` at revision fe666478454e79fe6027a0bab0388d9abd68f51a,
  the checkpoint snowball-100k training starts from (`experiments/post_training/inventory/train_quick_loop.py`
  L33-L34); its tokenizer and `chat_template.jinja` from the same revision (stage 3's copies).
- Frame: stage 2's 12,288 decoded closure rows (`stage2/rows/closure_rows.jsonl`). Strata (format/family/control) and
  rows drawn with `random.Random(f"closure_probe:20261113:{stratum}").sample(sorted(row_index), k)`:
  `code_fence/code_fence/close_required` 200 of 284 (primary), `tool/tool_delimiter/close_required` 100 of 615,
  `tool/repeated_opener_retention/opener_retention` 100 of 618 (secondary), `code_fence/code_fence/continue_required`
  100 of 285 (negative control).
- Prompt file `prompts_probe.jsonl` (`build_probe.py`): 900 prompts. For each newline row an arm-T and an arm-U
  prompt; for each control row one arm-T prompt (`u_identical_to_t`). idx = 700,000 + 2 × row number, + 1 for arm U.
  - Arm T: training's rendering, `apply_chat_template([user instruction, assistant prefix], tokenize=True,
    add_generation_prompt=False, continue_final_message=True)` (stage 3's `build_prompts.Renderer.render`).
  - Arm U: arm T's text followed by the characters the renderer removed (the prefix's trailing whitespace: one `\n`),
    tokenized as a whole by `tokenizer.encode(text, add_special_tokens=False)`; equal to the rendering of a template
    without the assistant `| trim`.
- Sampling on the CPU machine: `swaptest.run`, harness arm A, **CAP 512 new tokens**, temperature 1.0, seed 9984,
  batch size 384, one reply per prompt.

## 2. Processing of one reply (`analyze_probe.evaluate`)

The continuation is the record's `text` (decoded with special tokens kept, final stop token excluded, as
`swaptest.run` writes it). For each reply, with `prefix` the stored prefix and `prefix_seen` the prefix as the model
saw it (arm T: `prefix.rstrip()`; arm U and control: `prefix`):

1. **Reward graded now** (measure 1): stage 2's `graders.closure(text, prefix, expected, format)` (verifyit
   `grade_candidate` on a `StructureClosureSpec`; a blank continuation scores 0). This is what training gives.
2. **Reward against the seen prefix** (measure 2): the same call with `prefix_seen` in place of `prefix`, same
   expected text and format.
3. **Category**, per join: `stop` when the reply ended with a stop token; for a reply with `stop_reason == "cap"`,
   `cap_resolved` when `closure_state(join + text)` has a completion offset and an empty completion guard (its reward
   cannot change with more tokens), else `cap_unresolved`. Unresolved replies keep the reward the grader gives the
   capped text.
4. **Form class** (measure 3): stage 3's `s3grade.closure_form(text, golden)`: exact, leading_newline, leading_space,
   trailing_newline, trailing_space, trailing_text, reasoning_markup, other.
5. **Content** (secondary): stage 3's whitespace-free closure rule (`s3grade.content_correct`): the continuation
   without closed reasoning blocks, all whitespace removed, starts with the whitespace-free golden.
6. **Masked**: a grader that raises; the row is left out of the stratum's analysis set, and counted.

Full credit means reward ≥ 1 − 1e-9; the closure reward is 0 or 1, so a mean reward is a full-credit share
(replies with any other value are reported).

## 3. Analysis set

Per newline stratum, the rows with an arm-T and an arm-U reply and no masked measure; per control row, its arm-T
reply (it stands for arm U too). Rows are ordered by their `row` field (string sort).

## 4. Statistics and decisions (`analyze_probe.py`)

- Per newline stratum: means of **T now** (arm T, measure 1), **T seen** (arm T, measure 2), **U now** (arm U,
  measure 1).
- Contrasts, paired on rows: **renderer fix** = U now − T now (decided; primary on `code_fence/close_required`);
  **grader fix** = T seen − T now (decided, secondary); **fixes compared** = U now − T seen (descriptive).
- Intervals: 95% percentile intervals (numpy linear) from 10,000 paired bootstrap replicates over rows:
  `numpy.random.default_rng([20261113, k]).integers(0, n, size=(10000, n))`, k = the stratum's position in
  `probe_common.STRATA` (0 to 3); the same replicates for every mean and contrast of the stratum.
- Exact McNemar test on full credit: b = rows with full credit only in the new measure, c = only in the old;
  p = min(1, 2 · P(X ≤ min(b, c))), X ~ Binomial(b + c, 1/2); p = 1 when b + c = 0.
- **Decision rule** for a contrast with estimate D, interval [lo, hi] and McNemar p:
  - D ≥ +10 points, lo > 0 and p < 0.05: an effect in the positive direction (renderer fix: **"trimming costs
    reward"**; grader fix: "grading against the seen prefix raises reward");
  - D ≤ −10 points, hi < 0 and p < 0.05: an effect in the negative direction ("trimming raises reward"; "grading
    against the seen prefix lowers reward");
  - −10 < lo and hi < +10 points: **"no effect"**;
  - otherwise **"inconclusive"**.
  Robustness to capped replies: the label is computed three times, with `cap_unresolved` replies as graded, and with
  them set to the two extreme assignments (in the new measure 0 and in the old 1; in the new 1 and in the old 0);
  unless all three labels agree, the decision is **"inconclusive (capped replies)"**.
- The primary question is answered by the renderer-fix decision on `code_fence/close_required`. The tool strata are
  reported with the same rule; their grader accepts both whitespace forms (FRAME_TABLE.md), so a difference there
  would come from other behaviour. No correction for multiple contrasts.
- Control: full-credit share of its arm-T replies with its bootstrap interval (k = 3); no decision. It shows whether
  the checkpoint closes code fences where trimming removes nothing.
- Secondary, reported whatever the outcome: categories by stratum and arm; form classes with full credit now and
  seen and the content verdict; replies with right content and reward 0; rows of the 11,634 quota rows per pass whose
  full credit the trimming changes (quota rows per stratum from FRAME_TABLE.md, held fixed, × the stratum's renderer
  fix; the total's interval sums the strata's replicates).

## 5. Validity checks

- `run_config.json` beside `replies.jsonl` must name this prompt file (prompts_sha256) and arm A, else the analysis
  stops; a cap, temperature, seed or prompt count other than 512, 1.0, 9984, 900, or a checkpoint `config.json`
  other than the frozen one, is reported as a validity note.
- Every reply record must have exactly `swaptest.run`'s fields, a known prompt id, no duplicate, a stop reason that is
  a stop token ending its token ids or "cap", and token counts that agree; else the analysis stops.
- Fewer replies than prompts: a partial run; estimates use the complete rows.
- A reply whose `text` is not the decode of its token ids (stop token excluded) is counted in a validity note.
- Witnesses: at most three per (stratum, arm, form, reward graded now), each continuation quoted in at most 200
  characters.

## 6. What the numbers cannot say

One checkpoint (training's starting point) and one reply per prompt; the harness's eager PyTorch model sampled the
replies, not the training engine (ASSUMED equivalent up to numerical noise, as in stage 3); the record's text is
ASSUMED to equal the engine's decoded continuation (special tokens kept); the cap is 512, not 8,192; quota rows rest
on stage 2's page sample; nothing here describes the policy after training steps.
