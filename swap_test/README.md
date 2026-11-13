# swaptest

`swaptest` runs the attention-gate swap test on CPU. It loads the 67B mixture-of-experts checkpoint
`open-athena/Snowball-67B-A2B-Merge-TA0.5` once in bfloat16 (about 125 GiB) and decodes four arms in one batch.
The arms share every weight except a few small override tensors from a 70.8 MB patch file:

| arm | override |
|---|---|
| A | none |
| B | the 78 attention-gate tensors |
| C | the 26 router weights |
| D | both groups |

Each arm answers the same 48 GSM8K test prompts at temperature 1.0, with thinking on, the checkpoint's own
tokenizer and chat template, stop tokens 128001 and 128009, and a cap of 4,096 new tokens. Sampling noise is
shared across arms. The primary statistic is the share of replies that reach the cap without a stop token.
`PREREG.md` fixes the hypotheses, the statistics, the decision rule and the stop conditions.

The forward pass is a standalone port of Marin's eager PyTorch reference model with a preallocated key/value
cache. The tests check it against Marin's Levanter oracle and a transcription of the reference.

## Layout

```
pyproject.toml          package metadata, the extras "run" and "test", pytest and ruff settings
PREREG.md               the pre-registration
PREREG_phase2.md        the draft pre-registration of phase 2
swaptest/
  config.py             model configuration from config.json; stop tokens from generation_config.json
  checkpoint.py         shards loaded into memory; expected tensor names, shapes and dtypes
  patch.py              the two patch groups, arms A to D, checks against the checkpoint and the manifest
  model.py              forward pass with per-row arm overrides
  kvcache.py            key/value cache: every slot for long layers, a ring of 2,048 slots for short layers
  engine.py             chunked prefill and batched decoding that drops finished rows
  sampling.py           Gumbel-max sampler with noise seeded by (seed, prompt, position)
  loading.py            checkpoint, patch and model with a load and memory report
  chat.py               the checkpoint's tokenizer and chat template
  prompts.py            the GSM8K prompt file: seeded draw, rendering, one-start-token check
  gsm8k.py              flexible answer extraction and scoring
  run.py                generation run, one JSONL line per finished reply, resumable
  teacher_forced.py     log-probabilities of arm A's replies under every arm, resumable
  stats.py              Fisher exact test, Wilson and Newcombe intervals, decision rule
  analyze.py            primary and secondary statistics, validity gate V1, decisions
  checks.py             numerical checks and the decode-speed model
  validate_small.py     plan step 5: the small stand-in against its vLLM record
  selfcheck.py          plan steps 6 and 7: checks, memory and timing on the 67B
  skyrl_crosscheck.py   logits of Marin's reference model, for the parity checks
  verify_download.py    downloaded files against the saved and the live repository listings
  hashes.py             the two hashes the pre-registration records
  merge.py              phase 2: task-arithmetic merge built in memory with Marin's arithmetic; --check-against
  answers.py            phase 2: the part of a reply the capability graders read (final content)
  cruxeval.py           phase 2: CRUXEval prompts, answer reading and grading by execution
  mmlu_pro.py           phase 2: MMLU-Pro chain-of-thought prompts and answer extraction
  taskprompts.py        phase 2: the capability prompt file (seeded draw, rendering, checks)
  selfcheck_phase2.py   phase 2, plan step 17: checks, memory and timing on the merge built in memory
  analyze_phase2.py     phase 2: quick look, runaway decisions (necessity) and the capability comparison
  jsonfield.py          one field of a JSON file, for the scripts
  data/                 the patch manifest, five saved repository listings, the two CRUXEval prompt templates
scripts/
  bootstrap.sh          Python environment on a fresh Ubuntu 24.04 machine, CPU report, unit tests
  download.sh           pinned downloads and their verification
  run_phase1.sh         plan steps 5 to 12, resumable, one exit code per stop condition
  download_phase2.sh    phase 2's pinned downloads (B, E, E2, CRUXEval, MMLU-Pro) and their verification
  run_phase2.sh         phase 2, plan steps 14 to 24, resumable, one exit code per stop condition
tests/                  unit and end-to-end tests on tiny random checkpoints and made-up data, and their fixtures
```

## Tests

The tests need `torch`, `safetensors`, `numpy`, `pytest`, `jinja2` and `pyarrow`. They need neither
`transformers` nor network access nor model weights: they render the checkpoint's chat template
(`tests/fixtures/chat_template.jinja`) with jinja2 through a small byte-level tokenizer. The smoke test of
`run_phase2.sh` also needs `bash` (on Windows, Git's) and is skipped without it. From this directory:

```
python -m pip install -e ".[test]"
python -m pytest -q -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
```

## On the machine: plan steps 3 to 12

The machine runs Ubuntu 24.04 with 32 vCPUs, 256 GiB of memory, a CPU with AMX and AVX512-BF16, and a volume of
at least 750 GB. The commands below use `/data` for the data and results and `~/harness` for this directory.
Shell scripts write logs into files; none of them powers the machine off.

### Step 3: check the machine

The machine is launched elsewhere. After logging in:

```
lscpu | grep -o -w -E 'avx512_bf16|amx_bf16|amx_tile' | sort -u
free -g
cat /run/poweroff_at; shutdown --show
df -h /
```

The step worked when all three CPU flags are listed, `free -g` shows a total of about 250 GiB, a power-off is
scheduled, and the volume has at least 150 GB free (phase 1 downloads about 136 GB). Create the data directory:

```
sudo mkdir -p /data && sudo chown "$(id -un)" /data
```

If the large volume is a separate empty device rather than the root volume, find it with `lsblk`, then run
`sudo mkfs.ext4 /dev/DEVICE && sudo mkdir -p /data && sudo mount /dev/DEVICE /data && sudo chown "$(id -un)" /data`
with its name in place of `DEVICE`.

### Step 4: environment and downloads

Copy this directory to the machine (for example `scp -r harness USER@HOST:` from your computer), then:

```
bash ~/harness/scripts/bootstrap.sh 2>&1 | tee ~/bootstrap.log
bash ~/harness/scripts/download.sh /data small 2>&1 | tee /data/download_small.log
nohup bash ~/harness/scripts/download.sh /data large > /data/download_large.log 2>&1 &
```

- `bootstrap.sh` installs PyTorch 2.11.0 (CPU wheel) and the harness with `transformers` 5.16.1 into
  `~/.venvs/swaptest`, reports the CPU flags, memory and the planned power-off, and runs the unit tests.
- `download.sh ... small` fetches the patch (SHA-256 checked), the GSM8K test split, the stand-in
  `open-athena/Snowflake-d512-Datakit-SFT-262K-2026.10.22` with its `vllm_validation.json`, Marin's reference
  model source for the parity checks (SHA-256 checked), and the small files of the 67B checkpoint.
- `download.sh ... large` fetches the 502 weight shards, then checks every file's size and SHA-256 against the
  listing saved with the package and against the live listing at the pinned commit.
- Every verified item leaves `/data/markers/ITEM.verified`; a failed stage leaves
  `/data/markers/download_STAGE.failed`. Rerunning a stage skips verified items.
- The GSM8K commit is not pinned in advance. Unless `GSM8K_REV` is set, the commit of `main` at download time is
  used and recorded in `/data/gsm8k/REVISION`, and the pre-registration records it.

### Steps 5 to 12: the experiment

Start the phase-1 script; it may start while the large download is still running:

```
DATA_DIR=/data nohup bash ~/harness/scripts/run_phase1.sh > /data/run_phase1.out 2>&1 &
```

| step | what the script does | output under `/data/phase1` |
|---|---|---|
| 5 | builds the prompt file; runs the unit tests and the stand-in validation (checks b to e) | `prompts.jsonl`, `logs/step05_*` |
| 6, 7 | waits for the shards, loads the 67B with the patch, runs the memory, template, parity, cached-decode and mixed-arm checks, times decode steps and projects the main run | `step07_selfcheck.json`, `logs/step07_selfcheck.log` |
| 8 | arm A alone on 8 prompts, then pauses for reading | `step08_review.txt`, `step08_quick_A/` |
| 9 | fills in `PREREG.md` with the hashes and records the SHA-256 of the filled copy | `prereg.md`, `prereg_hashes.txt` |
| 10 | main run, 192 rows (arms A to D on 48 prompts) | `main/replies.jsonl`, `main/run.log` |
| 11 | teacher-forced pass on arm A's replies under every arm | `teacher_forced.jsonl` |
| 12 | analysis: V1 and the decision for B, C and D | `analysis.txt`, `analysis.json`, `SHA256SUMS` |

Follow progress with:

```
tail -n 20 /data/phase1/logs/run_phase1.log
tail -n 3 /data/phase1/main/run.log
```

The main run logs tokens per second and the number of active rows every 100 steps. The last line of
`logs/run_phase1.log` says how the script ended: `STOP (exit N): ...` names the stop condition, and
`phase 1 finished` means every step is done and V1 passed.

| exit | meaning | what to do |
|---:|---|---|
| 0 | all steps done, V1 passed | read `analysis.txt` |
| 1 | unexpected error or a crashed program | read the step's log, fix the cause, run the same command again |
| 2 | missing input, bad setting, or harness or prompts changed after step 9 | fix the setting; a change after step 9 is a deviation |
| 10 | stop S0: prompt file or stand-in validation failed | fix the harness before using the 67B |
| 11 | step 6: loading, patch or memory check failed | read `logs/step07_selfcheck.log` |
| 12 | step 7: a 67B check failed | read `logs/step07_selfcheck.log` |
| 13 | stop S1: projected main run longer than `MAX_HOURS` (8) | decide; see below |
| 14 | stop S2: fewer than 6 of 8 arm A replies stopped | debug the template, tokenizer or sampler |
| 15 | pause after step 8 | read `step08_review.txt`; see below |
| 16 | main run not finished | run the same command again; the run resumes |
| 17 | teacher-forced pass failed or incomplete | run the same command again; the pass resumes |
| 18 | analysis failed | read `analysis.txt` |
| 19 | V1 failed | no arm is interpreted; the analysis is written |
| 20 | an input is not downloaded | read `/data/download_*.log`, rerun `download.sh`, then this script |

After the pause at step 8, read the eight replies. If at least six stopped (the script checks this) and they read
as worked GSM8K solutions, continue:

```
DATA_DIR=/data STEP8_APPROVED=1 nohup bash ~/harness/scripts/run_phase1.sh > /data/run_phase1.out 2>&1 &
```

At stop S1, either continue with `ACCEPT_PROJECTION=1`, or change `ARMS` (for example `AB`), `CAP` or
`BATCH_SIZE`, delete `/data/phase1/markers/step07_checks.done` and run the script again so that steps 6 and 7 are
repeated with the new settings. A different `N_PROMPTS` needs a new `OUT_DIR`.

Every finished step leaves `/data/phase1/markers/STEP.done`. After an interruption (a crash, a reboot, an
out-of-memory kill), run the same command again: finished steps are skipped and the main run continues from its
last snapshot. If the main run runs out of memory, run it again with `BATCH_SIZE=96`: finished replies are kept
and the rest run in waves of 96 rows, each holding all arms of its prompts.

Settings are environment variables; `scripts/run_phase1.sh` lists them all with their defaults. The main ones are
`N_PROMPTS` (48), `CAP` (4096), `BATCH_SIZE` (192), `SEED` (9984), `ARMS` (ABCD) and `THREADS` (the number of
physical cores).

### Optional interim look at step 2,048

When `main/run.log` reports step 2,048 or later:

```
~/.venvs/swaptest/bin/python -m swaptest.analyze --prompts /data/phase1/prompts.jsonl \
    --replies /data/phase1/main/replies.jsonl --interim
```

An arm may be called "no effect" early only where the report says `early no-effect call allowed: True`. The look
changes nothing in the run.

### If an arm is inconclusive

The pre-registered rule adds 48 prompts once, for arm A and every inconclusive arm. With `EXT` set to arm A
followed by the inconclusive arms (for example `EXT=AD`):

```
PY=~/.venvs/swaptest/bin/python
P=/data/phase1
$PY -m swaptest.prompts --gsm8k /data/gsm8k/main/test-00000-of-00001.parquet --checkpoint /data/merge-ta0.5 \
    --out $P/prompts_ext.jsonl --count 48 --offset 48 --seed 9984
$PY -m swaptest.run --checkpoint /data/merge-ta0.5 --patch /data/patch/gates_full_strength.safetensors \
    --prompts $P/prompts_ext.jsonl --out $P/main_ext --arms $EXT --cap 4096 --batch-size $((48 * ${#EXT})) \
    --seed 9984 > $P/logs/ext_run.log 2>&1
$PY -m swaptest.teacher_forced --checkpoint /data/merge-ta0.5 --patch /data/patch/gates_full_strength.safetensors \
    --prompts $P/prompts_ext.jsonl --replies $P/main_ext/replies.jsonl --out $P/teacher_forced_ext.jsonl --arms $EXT
cat $P/teacher_forced.jsonl $P/teacher_forced_ext.jsonl > $P/teacher_forced_all.jsonl
$PY -m swaptest.analyze --prompts $P/prompts.jsonl $P/prompts_ext.jsonl \
    --replies $P/main/replies.jsonl $P/main_ext/replies.jsonl --teacher-forced $P/teacher_forced_all.jsonl \
    --arms $EXT --json $P/analysis_96.json > $P/analysis_96.txt
```

The outcomes in `analysis_96.txt` are final for the extended arms; the other arms keep their outcome from
`analysis.txt`.

### Step 12: hand-back

Copy `/data/phase1` (and `/data/run_phase1.out`) to your own computer, for example with
`scp -r USER@HOST:/data/phase1 .`, and check the copy with `sha256sum -c SHA256SUMS` inside it.

## Phase 2: plan steps 13 to 24

Phase 2 rebuilds the withheld full-strength merge P = B + 1.0 (E - B) + 1.0 (E2 - B) in memory, checks the same
code against the published Merge-TA0.5, and runs three arms over it: P, N (the 78 attention-gate tensors from
Merge-TA0.5) and G (the same tensors from B; G0 in the plan). It decides whether the gates are necessary for the
failure to stop, and whether an arm that stops normally scores higher than Merge-TA0.5 on CRUXEval and MMLU-Pro.
`PREREG_phase2.md` fixes the rules. Phase 1 must have finished with V1 passed, on the same data directory.

```
nohup bash ~/harness/scripts/download_phase2.sh /data > /data/download_phase2.log 2>&1 &
DATA_DIR=/data nohup bash ~/harness/scripts/run_phase2.sh > /data/run_phase2.out 2>&1 &
```

`download_phase2.sh` fetches the CRUXEval test file and the MMLU-Pro splits (stage `small`) and B, E and E2
(stage `large`, 402.5 GB; it needs 410 GB free), each pinned by commit and checked like phase 1's downloads. The
run script may start while the large stage is still running; it waits for each input.

| step | what the script does | output under `/data/phase2` |
|---|---|---|
| 14 | writes the merge spec; rebuilds Merge-TA0.5 (0.5, 0.5) and compares it bit for bit with the download | `merge_spec.json`, `merge_check.json`, `merge_check.txt` |
| 15 | task prompts (270 per target) and quick-look prompts (6 per task) | `task_prompts.jsonl`, `quick_prompts.jsonl` |
| 16 | arm A on the quick-look prompts, graded; then a pause for reading | `step16_review.txt`, `quick.json` |
| 17 | builds P with arms N and G; memory, cache, template, cached-decode and mixed-arm checks; timing; projections | `step17_selfcheck.json` |
| 18 | fills in `PREREG_phase2.md` and records the SHA-256 of the filled copy | `prereg_phase2.md`, `prereg_hashes.txt` |
| 19 | runaway run: P, N, G on phase 1's 48 GSM8K prompts | `runaway/replies.jsonl` |
| 20 | teacher-forced pass of arm A's phase-1 replies under P, N, G | `teacher_forced.jsonl` |
| 21 | runaway analysis: V2, necessity, eligibility; projection of the capability runs; then a pause | `runaway.txt`, `runaway.json`, `runaway_decisions.txt` |
| 22 | capability run of arm A | `tasks_A/replies.jsonl` |
| 23 | capability run of the scored arms (the candidate, and P when eligible) | `tasks_merge/replies.jsonl` |
| 24 | grading and the capability decision; checksums | `capability.txt`, `capability.json`, `capability_decisions.txt`, `grades.jsonl`, `SHA256SUMS` |

| exit | meaning | what to do |
|---:|---|---|
| 0 | all steps done | read `runaway.txt` and `capability.txt` |
| 1 | unexpected error or a crashed program | read the step's log, fix the cause, run the same command again |
| 2 | missing input or bad setting, or prompts, spec or harness changed after step 18 | fix the setting; a change after step 18 is a deviation |
| 20 | an input is not downloaded | read `/data/download_phase2.log`, rerun the download script |
| 21 | phase 1 has not finished, or its V1 failed | finish phase 1; with V1 failed phase 2 does not run |
| 30 | stop S3: the rebuilt Merge-TA0.5 differs from the download | read `merge_check.json` (first difference); P would not be the withheld merge |
| 31 | the merge, its arms or the memory check failed | read `logs/step17_selfcheck.log`; for the caches, lower `BATCH_SIZE` or `TASK_BATCH_SIZE` |
| 32 | another step-17 check failed | read `logs/step17_selfcheck.log` |
| 33 | stop S4: projected runaway run over `MAX_HOURS` (8) | decide; `ACCEPT_PROJECTION=1` continues |
| 34 | stop S5: quick look outside its limits | read `step16_review.txt`; debug the template, extraction or grader |
| 35 | pause after step 16 | read `step16_review.txt`; continue with `TASKS_APPROVED=1` |
| 36, 37 | runaway run or teacher-forced pass not finished | run the same command again; it resumes |
| 38 | runaway analysis failed | read `runaway.txt` |
| 39 | no arm stops normally enough for capability scoring | the runaway result is final |
| 40 | stop S6: projected capability runs over `MAX_TASK_HOURS` (20) | decide; `ACCEPT_TASK_PROJECTION=1` continues |
| 41 | a capability run not finished | run the same command again; it resumes |
| 42 | capability analysis failed | read `capability.txt` |
| 43 | V3 failed | no capability claim; the analysis is written |
| 44 | pause after step 21 | read `runaway.txt` and the projection; continue with `CAPABILITY_APPROVED=1` |

Every finished step leaves `/data/phase2/markers/STEP.done`; after an interruption the same command continues.
Settings are environment variables listed at the top of `scripts/run_phase2.sh`; the main ones are `N_TASK`
(270), `TASK_CAP` (4096), `TASK_BATCH_SIZE` (384), `BATCH_SIZE` (144) and `MMLU_PRO_IDS` (a file of question
ids to draw MMLU-Pro from; empty means all 12,032 documents, because marin#9984's 1,606-document subset is not
public).

The merge check alone, outside the script:

```
PY=~/.venvs/swaptest/bin/python
$PY -m swaptest.merge --spec /data/phase2/merge_spec.json --coefficients 0.5,0.5 --check-against /data/merge-ta0.5 \
    --json /data/phase2/merge_check.json
```
