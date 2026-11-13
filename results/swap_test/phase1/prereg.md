# Pre-registration: attention-gate swap test, phase 1

This document fixes the design, the statistics and the decision rule of phase 1 before the main run. The three
lines under "Record" are filled in on the machine before the main run starts (`run_phase1.sh` step 9 writes the
filled copy `prereg.md` next to the results and records its SHA-256 in the run log). Any change to the design,
the code or the prompts after that point is reported as a deviation.

## Question and hypotheses

The merge of `open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92` with two expert fine-tunes at full strength
(task arithmetic, coefficients 1.0 and 1.0) was reported in marin#9984 to produce replies that do not stop. The
published half-strength merge, `open-athena/Snowball-67B-A2B-Merge-TA0.5`, behaves normally. Phase 1 asks whether
replacing a few tensors of the half-strength merge by their full-strength values is enough to cause the failure.

- H_B: replacing the 78 attention-gate tensors (per layer `attn_gated_norm.down_proj`, `attn_gated_norm.up_proj`
  and `self_attn.attn_gate`, 26 layers) by their full-strength values raises R, the share of replies that reach
  the cap without a stop token, relative to the unmodified merge.
- H_C: the same for the 26 router weights (`mlp.router.weight` of every layer).
- H_D: the same for both groups together (104 tensors).

Each hypothesis is tested against the unmodified merge (arm A) with the rule below. A phase-1 result that shows no
effect does not refute the hypotheses on its own: the runaway has not yet been reproduced outside the serving
stack of marin#9984, and phase 2 (a rebuilt full-strength merge as positive control) is outside this document.

## Materials

| item | value |
|---|---|
| checkpoint | `open-athena/Snowball-67B-A2B-Merge-TA0.5`, commit `cd0605de0ab76473b2262a9545596436e726b27d` (502 shards, 134,157,855,008 bytes); its own `tokenizer.json`, `chat_template.jinja` and `generation_config.json` for every arm |
| patch | `gates_full_strength.safetensors` from release `v0.1.0` of `XiaoyuanWang111/grug-weight-atlas`, 70,831,936 bytes, SHA-256 `e8c7387ac84f01f3f1b1569397db07f99b9a4221ec78d411e86d84aa022b5946`; groups `attention_gates` (78 tensors) and `router_weights` (26 tensors) as listed in `swaptest/data/patch_manifest.json` |
| prompts | `openai/gsm8k`, configuration `main`, split `test` (1,319 rows), at the dataset commit recorded below |
| stand-in for validation | `open-athena/Snowflake-d512-Datakit-SFT-262K-2026.10.22`, commit `cbc90512edcb5e38bad11e82d46bb69c3ef98afa`, with its `vllm_validation.json` |

Before the main run, every patch tensor must have the shape and dtype of the checkpoint tensor it replaces and
must differ from it, and the name rules of the harness must select exactly the manifest's two groups.

## Arms

| arm | weights |
|---|---|
| A | Merge-TA0.5 as published |
| B | A with the 78 tensors of group `attention_gates` from the patch |
| C | A with the 26 tensors of group `router_weights` from the patch |
| D | A with all 104 patch tensors |

All other weights, in particular every routed expert, are shared by the four arms.

## Prompt selection

- The test split is read in file order; its rows are numbered 0 to 1,318.
- The row numbers are shuffled with Python's `random.Random(9984).shuffle(list(range(1319)))`. The first 48
  numbers of the shuffled list are the prompts, in ascending order. The same 48 prompts are used for every arm.
- An extension (see the decision rule) uses numbers 48 to 95 of the same shuffled list, so it never repeats a
  prompt.

## Rendering

- One user message `"Question: {question}\nAnswer:"` with no system prompt of our own, rendered by
  `transformers` 5.16.1 `apply_chat_template` with the checkpoint's template, `add_generation_prompt=True`, and
  thinking left at the template default (on).
- The rendered text is tokenized with `add_special_tokens=False`. Every prompt must hold exactly one start token,
  at position 0, the template's `Reasoning: /think` line, and end with the assistant header
  `<|start_header_id|>assistant<|end_header_id|>\n`; a prompt that fails this check stops the run.

## Sampling and decoding

- Temperature 1.0; no top-p, top-k, repetition or other penalties; one sample per prompt and arm.
- Tokens are drawn by the Gumbel-max rule: token = argmax(logits / T + g), with logits converted to float64 and g
  standard Gumbel noise over the whole vocabulary in float64, -log(-log(u)) of uniform draws u. The draws for
  reply position t (0 for the first new token) of the prompt with row number i come from a `torch.Generator`
  seeded with the BLAKE2b digest of size 8 of the text "9984:i:t", read little-endian with the top bit cleared.
  The noise does not depend on the arm, the batch or the row order, so arms share their random numbers and stay
  token-identical until their distributions differ.
- A reply ends when it samples token 128009 or 128001 (the `eos_token_id` list of `generation_config.json`) or
  when it holds 4,096 new tokens. The stop token counts as one of the new tokens.
- Weights, activations and the key/value cache are bfloat16; norms, the router, rotary embeddings and attention
  softmax accumulate in float32, as in the reference implementation. Short (sliding-window) layers attend to the
  last 2,048 positions.
- The four arms run in one mixed batch (192 rows), or in waves of a batch size that is a multiple of 4, prompt by
  prompt, with the arms of a prompt in the same wave. An interrupted run resumes: finished replies are kept, and
  unfinished ones continue from their saved tokens with the same noise.

## Primary statistic

R, the share of an arm's replies that reach 4,096 new tokens without a stop token (stop reason `cap`).

## Decision rule

For each arm X in B, C and D, compared with arm A (x capped replies of n per arm):

- difference: d = x_X / n_X - x_A / n_A;
- p: the one-sided Fisher exact test, P(capped count of X >= x_X) under the hypergeometric distribution with both
  margins of the 2 x 2 table fixed;
- interval: the Newcombe hybrid-score 95% interval for d (Newcombe 1998, method 10), built from the two Wilson
  intervals with z = 1.959964; its upper end is d + sqrt((u_X - x_X / n_X)^2 + (x_A / n_A - l_A)^2), where
  (l, u) are the Wilson ends (exactly 0 when x = 0 and exactly 1 when x = n).

Outcome, checked in this order:

1. **effect** if d >= 0.25 and p < 0.01;
2. **no effect** if the upper end of the interval is below 0.15;
3. otherwise **inconclusive**.

Each of the three arms is judged by this rule on its own; there is no further correction for the three
comparisons. If any arm is inconclusive, 48 more prompts (the extension above) are run once, for arm A and every
inconclusive arm, with the same settings, and the rule is applied again to all 96 prompts of each such arm; that
outcome is final, and an arm that is still inconclusive is reported as inconclusive.

Validity gate V1, evaluated on arm A over the same prompts: R_A <= 0.15 and arm A's GSM8K flexible accuracy
>= 0.80. If V1 fails, no arm is interpreted, whatever the rule gives.

GSM8K flexible accuracy: the reply text (stop token excluded) is cut at the first end-of-turn pattern
(`<|im_end|>`, `<|eot_id|>`, `<|end_of_text|>`, `<|endoftext|>`, `</s>`, `\nYou are an AI assistant`,
`Question:`, `\nQ:`, `\n[Question]`, `\nUser:`, `\nuser\n`, `\nAssistant:`); the answer is the last
`\boxed{number}`, else the last number after "answer is", "answer:" or "final answer", else the last number; it is
correct when it equals the number after `####` in the reference solution, compared as numbers after removing `$`
and thousands separators. Replies at the cap are scored on their capped text.

Interim look (optional) at step 2,048 of the main run: an arm may be called "no effect" early only if the call
still holds when every unfinished reply of that arm is counted as capped and every unfinished reply of arm A as
stopped. The interim look does not change the generation; the final analysis is run in any case and its outcome
is the result.

The rule and the gate are implemented in `swaptest/stats.py` and `swaptest/analyze.py` (thresholds 0.25, 0.01,
0.15, 0.15 and 0.80).

## Secondary statistics (reported, not used for the decision)

- median and quartiles of the new-token count per arm, with values at the cap marked as censored;
- which stop token ended each reply;
- how many replies contain `<|end_think|>`;
- GSM8K flexible accuracy of every arm on the capped text;
- repetition in the last 1,024 tokens (distinct 4-gram ratio and zlib compression ratio of the token ids), over
  all replies and over the replies at the cap;
- for each arm, how many replies equal arm A's reply and the median first position where they differ;
- teacher-forced readout on arm A's replies, scored under every arm: the mean negative log-likelihood of A's reply
  tokens and, for A's replies that ended with a stop token, the log-probability of that token and of any stop
  token at A's stop position.

## Stop conditions before the main run

| stop | condition | exit code of `run_phase1.sh` |
|---|---|---:|
| S0 | the prompt file cannot be built, or a stand-in check fails: the unit tests (including the oracle parity tests), the port against the reference implementation in float32 (max abs logit difference < 1e-4), the next-token log-probability of the vLLM record in bfloat16 (within 0.05 of -2.5997), the record's greedy chat samples (at least 3 of 4 equal over the first 32 tokens), or cached decode against a full forward over 2,600 tokens (top-1 agreement >= 99%, no NaN) | 10 |
| step 6 | the 67B does not load with the patch checks, is not resident, or leaves less than 80 GiB available | 11 |
| step 7 | a 67B check fails: the template check on three prompts, the port against the reference implementation in bfloat16 (top-1 equal at >= 63 of 64 positions), cached decode against a full forward over 2,600 tokens (>= 99%), or arm A in the mixed batch against arm A alone (>= 99%) | 12 |
| S1 | the projected main run (arms B and D running away) is longer than 8 hours | 13 |
| S2 | fewer than 6 of 8 arm A replies (cap 4,096) stop with 128009 or 128001 | 14 |
| review | the 8 arm A replies have not yet been read and judged to be worked GSM8K solutions | 15 (pause) |

V1 failing after the main run ends the script with exit code 19, after the full analysis is written.

## Reported whatever the outcome

- the analysis report and its JSON summary: per arm n, capped count, R, accuracy, `<|end_think|>` count, length
  quartiles, stop reasons, repetition and divergence statistics; V1; for B, C and D the difference, the Fisher p,
  the Newcombe interval and the outcome; the teacher-forced summary;
- every reply (token ids, decoded text, stop reason, length, seed, wall time) and the teacher-forced rows;
- the self-check summary (memory, every check, decode timings, fitted bandwidth and the projection), the stand-in
  validation log and the arm A quick-look replies;
- the run settings, all logs, this document as filled in and its SHA-256, and every deviation from it.

## Record

Filled in on the machine before the main run:

- Prompts file SHA-256: 126d243cd139ee57e9c66ec4df95e7296fdf035ecb7da7ac125fd5b106e8b0d2 (prompts_sha256)
- Harness tree SHA-256: d8e833efd3388ff454e5e5713dd157815d205fdca1ed2762a8e4f4aa863fe426 (harness_tree_sha256)
- GSM8K dataset revision: 740312add88f781978c0658806c59bc2815b9866 (gsm8k_revision)

The two hashes are printed by `python -m swaptest.hashes --prompts PROMPTS_FILE`. The harness tree hash covers
every file of the harness except this document and caches; `swaptest/hashes.py` gives its exact definition.
