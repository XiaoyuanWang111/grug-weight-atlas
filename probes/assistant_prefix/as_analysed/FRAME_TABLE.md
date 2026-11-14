# Closure probe: the model-free table

From stage 2's closure frame alone (12,288 decoded rows, `stage2/rows/closure_rows.jsonl`, read only); no model is involved. Produced by `frame_table.py` (worktree 80004cc06, checked clean). Final: it does not depend on the sampling run.

## 1. Which prefixes lose characters in the training prompt

Of the 21 strata of the frame, 3 have prefixes ending in whitespace: `code_fence/code_fence/close_required`, `tool/repeated_opener_retention/opener_retention`, `tool/tool_delimiter/close_required`. In each of them every row ends in exactly one newline, and no prefix in the frame starts with whitespace (VERIFIED, per row). The training renderer therefore removes exactly one `\n` from these rows and nothing from any other row.

| Stratum | Frame rows | Rows ending in whitespace | Removed suffix | Rows with leading whitespace |
|---|---:|---:|---|---:|
| `code_fence/code_fence/close_required` | 284 | 284 | `'\n'` × 284 | 0 |
| `code_fence/code_fence/continue_required` | 285 | 0 | none | 0 |
| `tool/repeated_opener_retention/opener_retention` | 618 | 618 | `'\n'` × 618 | 0 |
| `tool/tool_delimiter/close_required` | 615 | 615 | `'\n'` × 615 | 0 |

## 2. Size in the frame and in the quota

Share of the source with stage 2's page weights: the sampled pages [12, 13, 16, 18, 20, 23] (6,144 rows) stand for pages 12-26 and [35, 41, 44, 52, 64, 65] (6,144 rows) for pages 27-67, weighted 15:41 (code-fence rows start near page 27). Quota rows = share × 11,634 (ASSUMED: the quota draw is uniform over the source). The range takes, in each page group, the page with the lowest and the highest share.

| Stratum | Role | Frame rows (of 12,288) | Share of the frame | Share of the source | Range over pages | Quota rows (of 11,634) |
|---|---|---:|---:|---:|---|---:|
| `code_fence/code_fence/close_required` | primary | 284 | 2.3% | 3.4% | 2.9% to 3.5% | 394 |
| `tool/tool_delimiter/close_required` | tool | 615 | 5.0% | 4.9% | 4.8% to 5.0% | 569 |
| `tool/repeated_opener_retention/opener_retention` | tool | 618 | 5.0% | 4.9% | 4.8% to 5.0% | 572 |
| `code_fence/code_fence/continue_required` | control | 285 | 2.3% | 3.4% | 2.9% to 3.5% | 395 |
| **three newline strata** | | **1,517** | 12.3% | **13.2%** | | **1,534** |

## 3. What the grader gives to two fixed continuations

Real closure grader (`stage2/ops/graders.py` `closure`: verifyit `grade_candidate` on a `StructureClosureSpec`, which scores `score_closure_region(spec.prefix + candidate, expected, format)`, `grade_structure_closure.py` L39). **Untrimmed join** = the spec's prefix as stored, which is what training grades against; **trimmed join** = the spec's prefix with its trailing whitespace removed, i.e. the text the model saw (same expected text). Cell = rows with full credit / rows; every other row scored 0, none was masked unless stated.

**`code_fence/code_fence/close_required`** (primary; 284 rows)

| Continuation | Untrimmed join (graded now) | Trimmed join (as the model saw it) |
|---|---|---|
| golden | 284 / 284 | 0 / 284 |
| `\n` + golden | 0 / 284 | 284 / 284 |

**`tool/tool_delimiter/close_required`** (tool; 615 rows)

| Continuation | Untrimmed join (graded now) | Trimmed join (as the model saw it) |
|---|---|---|
| golden | 615 / 615 | 615 / 615 |
| `\n` + golden | 615 / 615 | 615 / 615 |

**`tool/repeated_opener_retention/opener_retention`** (tool; 618 rows)

| Continuation | Untrimmed join (graded now) | Trimmed join (as the model saw it) |
|---|---|---|
| golden | 618 / 618 | 618 / 618 |
| `\n` + golden | 618 / 618 | 618 / 618 |

**`code_fence/code_fence/continue_required`** (control; 285 rows)

| Continuation | Untrimmed join (graded now) | Trimmed join (as the model saw it) |
|---|---|---|
| golden | 285 / 285 | 285 / 285 |
| `\n` + golden | 0 / 285 | 0 / 285 |

Reading: on code-fence closure rows the grader gives full credit only to the continuation that is right for the text the model did not see (the golden fence alone, which in the model's view sits on the content line); the continuation that is right for what the model saw (newline, then the fence) always scores 0. The two tool strata accept both. The control stratum ends in no whitespace, so the two joins are the same text and the table only shows that a leading newline breaks its info string.
