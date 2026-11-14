# Provenance of the swap test results

## Which code ran

`swap_test/swaptest/hashes.py` defines the harness tree hash: the SHA-256 over every file of the harness except
the two pre-registration documents and caches. Each phase wrote the hash of its tree into its filled
pre-registration before its main run, and its run script stops if the tree changes after that point.

| harness tree SHA-256 | recorded in | ran | `swap_test/` at commit |
|---|---|---|---|
| `d8e833efd3388ff454e5e5713dd157815d205fdca1ed2762a8e4f4aa863fe426` | `phase1/prereg.md` | phase 1: arms A to D on 48 prompts (`phase1/main/`) and the steps before it | `00582a26573a33b70a014eae424d309a753905ef` |
| `09fbae2901d2da97e274226b9fef0be81393a3fa7dd47021fa059b6e2e4431a9` | `phase2/prereg_phase2.md` | phase 2: arms P, N and G on the same prompts (`phase2/runaway/`) and the steps before it | `2c309e6c42676d2771508de938bc99c1e974718e` |
| `88d7b7e9ba2de51433cf0b8f4c0428885f1c6fdd05e41ce1a32c82955e9833f6` | no run | nothing: the phase-2 tree with deviation 3 | `7adc2a1ddeb4eb5cfdb059fe813dcdfd11291164` |
| `f3f10e1aa86657f932b77de9fe2adbc8dd61caf25fa5fdd1679e93f60d3658a4` | `phase2/deviation4.md` | phase 2's capability runs: arms A and G on the task prompts (`phase2/tasks_A/`, `phase2/tasks_merge/`) and the capability analysis, issued by `phase2/run_phase2b.sh` | `d5def3aa5d0424d9be85e5f5e57b9f3410e03761` to `12f2727ff362b0a19c26512940ad92f2b18bf81a` |
| `25db15cc128b97b987aa1122e5380163efe77bfd5ced36486553ce533e85c6fc` | `phase2c/prereg_phase2c.md` | phase 2c (deviation 5): the quick look and arms A and G on the zero-shot MMLU-Pro prompts (`phase2c/quick_A/`, `phase2c/tasks_A/`, `phase2c/tasks_merge/`) and the capability analysis, run from commit `7a04ec2051079711c35119c13b30073251b59a13` | `549cfd6eaee94c07bfbb9c4b7652a083231b9669` and later |

To check a tree, extract it and hash it with the published `hashes.py`, from the repository root:

```
mkdir -p /tmp/tree && git archive 00582a26573a33b70a014eae424d309a753905ef swap_test | tar -x -C /tmp/tree
cd swap_test && python -m swaptest.hashes --prompts ../results/swap_test/phase1/prompts.jsonl --harness /tmp/tree/swap_test
```

`hashes.py` needs no third-party package. The phase-1 tree has no `PREREG_phase2.md`, so the published exclusion
rule gives the hash that phase 1 recorded. `PREREG_phase2c.md` is inside the hash of its tree: `run_phase2c.sh`
fills in a copy and leaves the file as it is.

The extension of arms A, B and D (`phase1/main_ext/`, `phase1/teacher_forced_ext.jsonl`, `phase1/analysis_96.*`)
and the replicate at batch size 96 (`phase1_rep96/`) ran after phase 1's last step, with the commands in
`swap_test/README.md`; their logs do not name the tree. Phase 2 ran from its own copy of the harness in its own
Python environment. The modules these runs call are byte-identical in the phase-1 and phase-2 trees: `run.py`,
`teacher_forced.py`, `analyze.py` and the modules they import (`model.py`, `engine.py`, `kvcache.py`,
`sampling.py`, `loading.py`, `checkpoint.py`, `config.py`, `patch.py`, `prompts.py`, `chat.py`, `stats.py`,
`gsm8k.py`, `records.py`). `loading.py` also imports `merge.py`, which changed between the two trees only in its
command-line handling, line wrapping and a `strict=True` on one `zip`, none of which a run with patch arms calls.
`git diff 00582a26573a33b70a014eae424d309a753905ef 2c309e6c42676d2771508de938bc99c1e974718e -- swap_test` shows
every change between the two trees.

## What changed after the runs

Commit `7adc2a1ddeb4eb5cfdb059fe813dcdfd11291164` (deviation 3 in `swap_test/DEVIATIONS.md`) changes one condition
in `analyze_phase2.py`: an arm is eligible for capability scoring only when it shows no effect against A, as
`PREREG_phase2.md` says; the code that ran phase 2 tested "not effect". `test_analyze_phase2.py` and
`DEVIATIONS.md` change with it. The runaway analysis as run (`phase2/runaway.txt`, `runaway.json`,
`runaway_decisions.txt`) names N as eligible and as the candidate. `phase2/runaway_reanalysis.txt`,
`runaway_reanalysis.json` and `runaway_decisions_reanalysis.txt` are the output of the published code on the same
replies and teacher-forced rows. They differ from the files as run only in eligibility: N is not eligible, and
the candidate is G. The capability runs came later and read `runaway_reanalysis.json`, so no other result
depends on the change.

Commit `d5def3aa5d0424d9be85e5f5e57b9f3410e03761` changes only the smoke test of `run_phase2.sh`
(`tests/test_phase2_script.py`). It had 4 GSM8K prompts, where 0 of 4 against 0 of 4 is inconclusive (upper end
of the Newcombe interval 0.49), so under deviation 3 no arm was eligible, the script stopped with exit code 39
instead of pausing before the capability runs, and the test failed; it passes on the phase-2 tree. With 40 prompts
the upper end is 0.088 and the test passes on the published tree.

Phase 2's capability runs (deviation 4) ran from the published tree at commit
`12f2727ff362b0a19c26512940ad92f2b18bf81a` through `phase2/run_phase2b.sh`, because `run_phase2.sh` accepts only
the tree of the filled pre-registration. The driver stopped unless the two trees differed in `swaptest/` in the
one eligibility line of `analyze_phase2.py` alone, and unless the task prompts, the merge spec and the filled
pre-registration had their recorded SHA-256. `phase2/deviation4.md` is the note written before the first
capability reply was sampled; `swap_test/DEVIATIONS.md` holds the same text.

Commits `67a01d345d3f7e6d50d800d8b3d3dbf29a25caa2`, `b8a26fcaf85019d04141684669946c4991a215d6` and
`549cfd6eaee94c07bfbb9c4b7652a083231b9669` (deviation 5) add the zero-shot MMLU-Pro prompt option to
`taskprompts.py`, the module `sameprompts.py` and the script `run_phase2c.sh`, and let the capability analysis of
`analyze_phase2.py` leave out replies to prompts outside its prompt file and take `--arms`. `run.py` and the
modules it imports do not change, and the grading rules do not change (`mmlu_pro.py` changes in its docstring
only).

The published code reproduces `phase1/analysis.txt`, `phase1/analysis.json`, `phase1/analysis_96.txt`,
`phase1/analysis_96.json`, `phase1_rep96/analysis.txt`, `phase1_rep96/analysis.json`, `phase2/capability.txt`,
`phase2/capability.json`, `phase2/capability_decisions.txt`, `phase2c/capability.txt`, `phase2c/capability.json`
and `phase2c/capability_decisions.txt` byte for byte from the replies, teacher-forced rows and grades here
(commands in `README.md`, section 1).

## Files

The files are copied unchanged from the runs, with these notes:

- `phase1/SHA256SUMS` was written at phase 1's last step. It also lists run logs and two summaries,
  `step07_selfcheck.json` and `step08_summary.json`, which are not here and were not in the copy of the run
  either. Every listed file that is here matches.
- `phase1/prereg.md.sha256` and `phase1/prereg_hashes.txt` were not in the copy of the run. They are rewritten in
  the format of `run_phase1.sh` step 9 from the values that `phase1/prereg.md` records, and their SHA-256 equal the
  lines of `phase1/SHA256SUMS` for them.
- `phase1/diag_numerics.log` is the output of the numerics diagnostic of deviation 2: top-1 disagreements and
  logit differences over 2,600 positions, KV-cached decode against a full forward and a full forward against a
  second one. The run wrote it next to the phase-1 directory.
- The paths in `phase2/merge_spec.json` and `phase2/merge_check.json` are the data directory of the run.
- `phase2/task_prompts.jsonl` and `phase2/quick_prompts.jsonl` are the capability prompt files: the first for the
  capability runs (`phase2/tasks_A/`, `phase2/tasks_merge/`, `grades.jsonl`, `capability.*`), the second for the
  quick look at arm A (`phase2/quick_A/`, `quick.json`, `quick_grades.jsonl`, `step16_review.txt`).
- `phase2/run_phase2b.sh` is the driver of the capability runs as it ran; its paths are those of the run.
- `phase2/SHA256SUMS` was written at the last step of the capability runs. It also lists run logs and
  `merge_spec_widened_bias.json`, which are not here. Every listed file that is here matches.
- `phase2c/` holds phase 2c as `run_phase2c.sh` wrote it, with the setting `RUNAWAY` naming
  `phase2/runaway_reanalysis.json`. `phase2c/task_prompts.jsonl` holds phase 2's 540 CRUXEval prompts unchanged
  and the 270 zero-shot MMLU-Pro prompts; `run_prompts.jsonl` is those 270 alone, the file the two runs read.
  `phase2c/SHA256SUMS` also lists run logs, which are not here; every listed file that is here matches.

Not included: the run logs (`logs/`, the `run.log` of each run and the output of the run scripts), the step
markers, the download logs, `phase1/logs/step10_check.txt` (its 20 lines are the first 20 lines of
`phase1/analysis.txt`) and `phase2/merge_spec_widened_bias.json` (byte-identical to `phase2/merge_spec.json`).
