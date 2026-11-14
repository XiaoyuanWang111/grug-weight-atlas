# Closure probe: results

Produced by `analyze_probe.py` under the frozen pre-registration (`PREREG.md`, `prereg_hashes.txt`). Reward is binary here, so a mean reward is the share of replies with full credit. Intervals: 95% percentile intervals from 10,000 paired bootstrap replicates over rows. Differences in points of full-credit share.

Run: cap 512, temperature 1.0, seed 9984, arm A, 900 replies.

## 1. Headline per newline stratum

T now = arm T (training's prompt) graded as training grades (untrimmed prefix + continuation); T seen = the same replies graded against the prefix the model saw; U now = arm U (untrimmed prompt) graded as training grades.

| Stratum | Rows | T now | T seen | U now | U − T now (renderer fix) | McNemar b/c, p | Decision | T seen − T now (grader fix) | McNemar b/c, p | Decision |
|---|---:|---|---|---|---|---|---|---|---|---|
| `code_fence/code_fence/close_required` | 200 | 6.5% [3.5%, 10.0%] | 0.0% [0.0%, 0.0%] | 16.5% [11.5%, 22.0%] | +10.0 [+4.0, +16.0] | 30/10, 0.0022 | **inconclusive (capped replies)** | -6.5 [-10.0, -3.5] | 0/13, 0.00024 | **inconclusive (capped replies)** |
| `tool/tool_delimiter/close_required` | 100 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 75.0% [66.0%, 83.0%] | +75.0 [+66.0, +83.0] | 75/0, 5.3e-23 | **trimming costs reward** | +0.0 [+0.0, +0.0] | 0/0, 1 | **no effect** |
| `tool/repeated_opener_retention/opener_retention` | 100 | 58.0% [48.0%, 68.0%] | 58.0% [48.0%, 68.0%] | 40.0% [30.0%, 49.0%] | -18.0 [-31.0, -6.0] | 13/31, 0.0096 | **trimming raises reward** | +0.0 [+0.0, +0.0] | 0/0, 1 | **no effect** |

Decision rule (PREREG section 4): an effect needs an estimate of at least 10 points, a 95% interval on the same side of 0 and McNemar p < 0.05; "no effect" needs the whole interval inside (−10, +10) points; anything else is inconclusive. A label that changes when capped replies whose reward was not yet fixed are set to their extreme values becomes "inconclusive (capped replies)".

Descriptive: U now − T seen (the two fixes compared):

- `code_fence/code_fence/close_required`: +16.5 [+11.5, +22.0], McNemar 33/0, p 2.3e-10
- `tool/tool_delimiter/close_required`: +75.0 [+66.0, +83.0], McNemar 75/0, p 5.3e-23
- `tool/repeated_opener_retention/opener_retention`: -18.0 [-31.0, -6.0], McNemar 13/31, p 0.0096

## 2. Negative control

`code_fence/code_fence/continue_required` (prefix ends in no whitespace; arm U = arm T, sampled once): full credit 0 of 100 = 0.0% [0.0%, 0.0%]; 8 capped replies not yet fixed. T now = T seen = U by construction.

## 3. Replies by category and form

Categories (as graded now): stop = ended with a stop token; cap_resolved = reached the cap after its closure region completed; cap_unresolved = reached the cap before that.

| Stratum | Arm | stop | cap_resolved | cap_unresolved | masked |
|---|---|---:|---:|---:|---:|
| `cf_close` | T | 198 | 1 | 1 | 0 |
| `cf_close` | U | 190 | 1 | 9 | 0 |
| `tool_close` | T | 98 | 0 | 2 | 0 |
| `tool_close` | U | 96 | 0 | 4 | 0 |
| `tool_opener` | T | 100 | 0 | 0 | 0 |
| `tool_opener` | U | 100 | 0 | 0 | 0 |
| `cf_continue_ctrl` | T | 91 | 1 | 8 | 0 |

Form classes (stage 3's `closure_form`): full now / full seen = replies with full credit graded against the untrimmed / the seen prefix; content = whitespace-free content rule.

| Stratum | Arm | Form | Replies | Full now | Full seen | Content | Capped |
|---|---|---|---:|---:|---:|---:|---:|
| `cf_close` | T | other | 183 | 0 | 0 | 0 | 1 |
| `cf_close` | T | exact | 7 | 7 | 0 | 7 | 0 |
| `cf_close` | T | trailing_text | 7 | 6 | 0 | 7 | 1 |
| `cf_close` | T | reasoning_markup | 3 | 0 | 0 | 0 | 0 |
| `cf_close` | U | other | 145 | 0 | 0 | 2 | 5 |
| `cf_close` | U | exact | 23 | 23 | 23 | 23 | 0 |
| `cf_close` | U | trailing_text | 20 | 8 | 8 | 20 | 2 |
| `cf_close` | U | reasoning_markup | 11 | 1 | 1 | 2 | 3 |
| `cf_close` | U | trailing_newline | 1 | 1 | 1 | 1 | 0 |
| `tool_close` | T | other | 97 | 0 | 0 | 0 | 0 |
| `tool_close` | T | reasoning_markup | 3 | 0 | 0 | 0 | 2 |
| `tool_close` | U | exact | 74 | 74 | 74 | 74 | 0 |
| `tool_close` | U | other | 23 | 0 | 0 | 0 | 2 |
| `tool_close` | U | reasoning_markup | 3 | 1 | 1 | 1 | 2 |
| `tool_opener` | T | leading_newline | 58 | 58 | 58 | 58 | 0 |
| `tool_opener` | T | other | 42 | 0 | 0 | 0 | 0 |
| `tool_opener` | U | other | 60 | 0 | 0 | 0 | 0 |
| `tool_opener` | U | exact | 40 | 40 | 40 | 40 | 0 |
| `cf_continue_ctrl` | T | other | 88 | 0 | 0 | 0 | 8 |
| `cf_continue_ctrl` | T | reasoning_markup | 12 | 0 | 0 | 0 | 1 |

Content right (whitespace-free) but reward 0 as graded now:

- `cf_close`: T 1 of 200; U 15 of 200
- `tool_close`: T 0 of 100; U 0 of 100
- `tool_opener`: T 0 of 100; U 0 of 100
- `cf_continue_ctrl`: T 0 of 100

## 4. Secondary: size in the curriculum

Rows of the 11,634 closure quota rows per pass whose full credit the trimming changes (quota rows of the stratum from FRAME_TABLE.md, held fixed, × the stratum's U − T estimate):

- `code_fence/code_fence/close_required`: 39 [16, 63] of 394
- `tool/tool_delimiter/close_required`: 426 [375, 472] of 569
- `tool/repeated_opener_retention/opener_retention`: -103 [-177, -34] of 572
- all three: 363 [273, 452] of 11,634

## 5. Validity notes

- none

## 6. What the numbers cannot say

One checkpoint (the one snowball-100k training starts from) and one reply per prompt; the harness, not the training engine, sampled the replies; the cap is far below training's 8,192 (replies that reach it are reported apart and bound the decision); the closure source is page-sampled, so the quota rows are estimates; the result says nothing about later training steps.
