# A CPU probe of Marin's assistant-prefix prompt

**TL;DR.** Marin's RL training renders an unfinished assistant message (a prefix the model must continue) through
the chat template, and the rendering drops a newline at the end of the prefix, while the grader joins the stored
prefix, newline included, with the reply. On the checkpoint that snowball-100k training starts from, tool-call
closing rows earned full credit in 0 of 100 sampled replies with the prompt as rendered and in 75 of 100 with the
newline kept. The design and the decision rule were fixed before sampling (`PREREG.md`).

| stratum | rows | full credit, prompt as rendered (T) | full credit, newline kept (U) | U - T, points (95% interval) | decision |
|---|---:|---:|---:|---|---|
| `tool/tool_delimiter/close_required` | 100 | 0% | 75% | +75 (66 to 83) | trimming costs reward |
| `code_fence/code_fence/close_required` | 200 | 6.5% | 16.5% | +10 (4 to 16) | inconclusive (capped replies) |
| `tool/repeated_opener_retention/opener_retention` | 100 | 58% | 40% | -18 (-31 to -6) | trimming raises reward |
| `code_fence/code_fence/continue_required` (control) | 100 | 0% | same prompt | | |

## What is compared

A structure-closure row of `open-athena/Snowball-Smartie-RL-2026.10.15.32-Data` holds a user instruction, an
assistant prefix, and the text that the prefix and a correct reply must form. Since marin#11062 these rows train
under the assistant-prefix harness, which builds the prompt with `assistant_prefix_token_ids`
(`vendor/skyrl/skyrl-train/skyrl_train/inference_engines/chat_template.py` at marin 80004cc06):
`tokenizer.apply_chat_template(messages, add_generation_prompt=False, continue_final_message=True)`.

The chat template trims assistant content. That removes the trailing space of the tag Hugging Face appends to
find the end of the message, and Hugging Face then right-strips the text before the tag. A prefix
`<tool_call>\n{...}\n` therefore reaches the model as `<tool_call>\n{...}`: the prompt ends in the token `}}`
(3500) where the stored prefix ends in the single token `}}\n` (11498). The grader scores the stored prefix
joined with the reply (`lib/verifyit/src/verifyit/modes/grade_structure_closure.py`).

- **Arm T**: the prompt as that function renders it.
- **Arm U**: the same text with the removed newline restored, tokenized as a whole.

Three of the 21 closure strata have prefixes that end in a newline, each in every row; no prefix starts with
whitespace. The fourth stratum of the probe is a control whose prefix ends in no whitespace, so both arms are one
prompt.

## Setup

- **Model**: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` at `fe666478`, the checkpoint that
  `experiments/post_training/inventory/train_quick_loop.py` starts snowball-100k from, with its own tokenizer and
  chat template.
- **Rows**: 200, 100, 100 and 100 rows of the four strata, drawn with a fixed seed from 12,288 closure rows decoded
  from 12 pages of the dataset's `train.parquet` (`as_analysed/FRAME_TABLE.md`).
- **Sampling**: `swap_test/` of this repository at commit `12f2727`, on CPU in bfloat16, one reply per prompt,
  temperature 1.0, seed 9984, at most 512 new tokens (the longest golden continuation has 183 tokens):

  ```
  python -m swaptest.run --checkpoint CHECKPOINT --prompts prompts.jsonl --out OUT --arms A --cap 512 \
      --batch-size 384 --seed 9984 --temperature 1.0
  ```

- **Grading**: `verifyit.candidate.grade_candidate` on a `StructureClosureSpec`, the grader of training; a blank
  reply scores 0. Each reply is graded against the stored prefix ("now") and against the prefix as its prompt
  shows it ("seen").
- **Decision rule** (`PREREG.md` section 4): an effect needs a difference of at least 10 points, a 95% interval on
  the same side of 0 (10,000 paired bootstrap replicates over rows) and an exact McNemar p below 0.05; "no effect"
  needs the whole interval inside (-10, +10) points. A label that changes when capped replies whose reward was
  not yet fixed take their extreme values becomes "inconclusive (capped replies)".

## Results

- **Tool-call closing.** With the prompt as rendered, 94 of 100 replies are empty: the model emits a stop token
  at once. With the newline kept, 74 of 100 replies are exactly `</tool_call>`. McNemar 75 to 0.
- **Code-fence closing.** 146 of 200 replies are empty with the prompt as rendered and none with the newline
  kept. Full credit rises from 13 to 33 of 200. Ten capped replies are unresolved; under their extreme
  assignments the difference is 9.5 to 14.5 points with every interval above 0, and 9.5 is below the 10-point
  threshold, so the pre-registered label is inconclusive.
- **Opener retention.** The prompt as rendered earns credit more often (58 against 40 of 100): the model writes
  the newline itself, and the grader accepts the doubled newline.
- **Grading against the seen prefix** instead of the stored one changes nothing on the tool rows and removes the
  13 credits of the code-fence rows: a grader-side change does not recover the lost reward.
- **Size.** By the page weights of `as_analysed/FRAME_TABLE.md` the three strata hold about 1,530 of the 11,634
  closure rows in snowball-100k's quota of 100,000 (about 570, 390 and 570).

Exploratory, not pre-registered: the counts of empty replies and of replies equal to the golden continuation.

## Reproduce

The table follows from the files here and Marin's grader. `regrade.py` needs numpy, tomli-w (the one dependency
of `lib/verifyit`) and a checkout of marin; the probe used 80004cc06:

```
python regrade.py --marin MARIN_CHECKOUT --grades grades.jsonl
```

It prints the four strata and ends with "grades equal grades.jsonl". The prompts follow from the rows' messages
and the tokenizer alone, with no model:

```
python render_check.py --tokenizer open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 \
    --revision fe666478454e79fe6027a0bab0388d9abd68f51a
```

It prints the token example above and reproduces the 500 arm-T and 400 arm-U prompts.

## Files

```
PREREG.md          the pre-registration; one local directory name is replaced by <marin worktree>
prompts.jsonl      900 prompts: messages, token ids, arm, stratum, and the row's prefix, expected text and format
replies.jsonl      the sampled replies (token ids and text), one per prompt; run_config.json records the run
grades.jsonl       per reply: both rewards, whether a capped reply is unresolved, empty, equal to the golden text
regrade.py         grades the replies with Marin's grader and prints the table
render_check.py    renders the prompts from the messages with the tokenizer alone
as_analysed/       RESULTS.md and FRAME_TABLE.md as the first analysis wrote them
```

The first analysis ran inside the working tree of a larger study of snowball-100k's graders, which `as_analysed/`
calls stage 2 and stage 3; its scripts depend on that tree and are not here. `regrade.py` recomputes sections 1
and 2 of `as_analysed/RESULTS.md`, and every per-reply reward equals the first analysis.

## Limits

One checkpoint, the one training starts from; one reply per prompt; this repository's CPU sampler in bfloat16 in
place of the training engine; a cap of 512 new tokens where training allows 8,192 (capped replies are reported
apart and bound the decisions); rows from 12 sampled pages of the source, so the quota sizes are estimates. The
probe says nothing about later training steps or other checkpoints.

## Data and licenses

`prompts.jsonl` holds 500 unchanged structure-closure rows of
[`open-athena/Snowball-Smartie-RL-2026.10.15.32-Data`](https://huggingface.co/datasets/open-athena/Snowball-Smartie-RL-2026.10.15.32-Data)
at revision `cb728a76`. Its card describes these rows as assembled by the Marin structural-closure campaign from
synthetic cases and BFCL-rollout-derived content, and lists the terms of its components under "License and
attribution"; those terms apply to the rows here. The replies were generated with a checkpoint released under
OpenMDW-1.1. The code in this folder is under the repository's Apache-2.0 license.
