#!/usr/bin/env bash
# Phase 2 of the swap test on one machine: plan steps 14 to 24, in order, resumable.
#
#   14  rebuild Merge-TA0.5 from B, E and E2 and compare it bit for bit with the download          (stop S3)
#   15  prompt files of the capability tasks (CRUXEval input and output, MMLU-Pro) and of their quick look
#   16  quick look: arm A (Merge-TA0.5) on a few prompts of each task; then a pause until they are read  (stop S5)
#   17  build the full-strength merge P in memory with arms N and G; checks, memory and decode timing    (stop S4)
#   18  pre-registration: PREREG_phase2.md filled in with the hashes, its SHA-256 logged
#   19  runaway run: arms P, N, G on phase 1's GSM8K prompts with phase 1's noise (resumable inside the step)
#   20  teacher-forced pass of arm A's phase-1 replies under P, N and G
#   21  runaway analysis: V2, necessity of the gates, the arms that get capability scores             (stop S6)
#       then a pause until the capability runs are approved (they cost about three times steps 14 to 21)
#   22  capability run of arm A on the task prompts
#   23  capability run of the scored arms (the candidate N or G, and P when it stops normally)
#   24  capability analysis: V3 and the decision "better merge than Merge-TA0.5"; checksums
#
# Every finished step writes OUT_DIR/markers/<step>.done and is skipped when that marker exists, so after an
# interruption the same command continues where it stopped. Output of each step goes to OUT_DIR/logs/; this
# script's own log is OUT_DIR/logs/run_phase2.log. Nothing here schedules, cancels or performs a power-off.
# Phase 1 must have finished with V1 passed: its prompt file, arm A's replies and its analysis are inputs.
#
# Usage: DATA_DIR=/data bash scripts/run_phase2.sh
#
# Settings (environment variables; defaults in brackets):
#   DATA_DIR            directory given to download.sh and download_phase2.sh (required)
#   PHASE1_DIR          phase 1's results [DATA_DIR/phase1]
#   OUT_DIR             results [DATA_DIR/phase2]
#   VENV_DIR            environment made by bootstrap.sh [$HOME/.venvs/swaptest]; PY its Python [VENV_DIR/bin/python]
#   ARMS                arms of the runaway run [PNG]  CAP  new-token cap of the runaway run [4096]
#   BATCH_SIZE          rows per wave of the runaway run [144]    SEED  sampling seed [9984]
#   N_TASK              prompts per capability task [270]          TASK_SEED  seed of the task draw [9984]
#   TASK_OFFSET         first draw number of the task prompts [0]  TASK_CAP  new-token cap of the task runs [4096]
#   TASK_BATCH_SIZE     rows per wave of the capability runs [384]
#   MMLU_PRO_IDS        file of MMLU-Pro question ids to draw from [empty: all 12,032 test documents]; marin#9984's
#                       1,606-document subset is not public, so by default the draw covers all documents
#   QUICK_PER_TASK      prompts per task in the quick look [6]     QUICK_OFFSET  their draw number [800 - QUICK_PER_TASK]
#   QUICK_MAX_CAPPED    quick-look replies allowed at the cap [2]  QUICK_MAX_MEDIAN  longest median length [2000]
#   QUICK_MIN_CORRECT   correct quick-look replies needed per task [2]
#   TASK_SIGMA          sigma of the log reply length assumed in the capability projection [0.8]
#   THREADS             torch threads [number of physical cores]
#   PREFILL_CHUNK       slots per prefill chunk [256]   PREFILL_ROWS   rows per prefill block [16]
#   MAX_HOURS           stop S4 above this projected runaway run [8]
#   MAX_TASK_HOURS      stop S6 above this projected total of the capability runs [20]
#   MIN_FREE_GIB        memory that must stay available with the merge built [60]
#   KV_HEADROOM_GIB     memory that must stay available beside the planned caches [16]
#   LONG_LEN, LONG_PREFILL, MIXED_STEPS, TIME_BATCHES, TIME_STEPS   checks of step 17 [2600, 2300, 32, 1,16,64,144, 20]
#   LEVEL               level of the composite interval [0.95]     BOOT_DRAWS  bootstrap draws [10000]
#   GRADE_WORKERS       parallel graders [8]
#   TASKS_APPROVED=1    the quick-look replies were read and are sound: continue past step 16
#   CAPABILITY_APPROVED=1     the capability runs (steps 22 to 24) are approved: continue past step 21
#   ACCEPT_PROJECTION=1      continue although the projected runaway run exceeds MAX_HOURS (logged)
#   ACCEPT_TASK_PROJECTION=1  continue although the projected capability runs exceed MAX_TASK_HOURS (logged)
#   WAIT_FOR_DOWNLOAD   1 waits for a running download script to verify an input, 0 stops at once [1]
#   DOWNLOAD_TIMEOUT_MIN  longest wait for one input [240]
#
# Exit codes:
#   0   all steps done; the capability decision is in capability_decisions.txt
#   1   unexpected error (the log names the command and line), or a step's program crashed
#   2   missing input, bad setting, or harness, prompts or merge spec changed after the pre-registration
#   20  an input is not downloaded: download failed, not started, or timed out
#   21  phase 1 has not finished, or its validity gate V1 failed
#   30  stop S3: the rebuilt Merge-TA0.5 differs from the download (P would not be the withheld merge)
#   31  step 17: the merge, its arms or the memory for the planned waves failed the load checks
#   32  step 17: another check failed (template, cached decode, mixed arms)
#   33  stop S4: the projected runaway run exceeds MAX_HOURS
#   34  stop S5: the quick look is outside its limits (replies at the cap, length, correct answers)
#   35  pause after step 16: read the quick-look replies, then rerun with TASKS_APPROVED=1
#   36  step 19: the runaway run did not finish (rerun to resume)
#   37  step 20: the teacher-forced pass failed or its output is incomplete (rerun to resume)
#   38  step 21: the runaway analysis failed
#   39  no arm qualifies for capability scoring: the runaway result is final (written); no capability runs
#   40  stop S6: the projected capability runs exceed MAX_TASK_HOURS
#   41  step 22 or 23: a capability run did not finish (rerun to resume)
#   42  step 24: the capability analysis failed
#   43  stop: V3 failed (the analysis is written; no capability claim)
#   44  pause after step 21: read runaway.txt and the projection, then rerun with CAPABILITY_APPROVED=1
set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the directory given to the download scripts}"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "DATA_DIR $DATA_DIR is not a directory" >&2
    exit 2
fi
PHASE1_DIR="${PHASE1_DIR:-$DATA_DIR/phase1}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/phase2}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/swaptest}"
PY="${PY:-$VENV_DIR/bin/python}"

ARMS="${ARMS:-PNG}"
CAP="${CAP:-4096}"
BATCH_SIZE="${BATCH_SIZE:-144}"
SEED="${SEED:-9984}"
N_TASK="${N_TASK:-270}"
TASK_SEED="${TASK_SEED:-9984}"
TASK_OFFSET="${TASK_OFFSET:-0}"
TASK_CAP="${TASK_CAP:-4096}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-384}"
MMLU_PRO_IDS="${MMLU_PRO_IDS:-}"
QUICK_PER_TASK="${QUICK_PER_TASK:-6}"
QUICK_OFFSET="${QUICK_OFFSET:-$((800 - QUICK_PER_TASK))}"
QUICK_MAX_CAPPED="${QUICK_MAX_CAPPED:-2}"
QUICK_MAX_MEDIAN="${QUICK_MAX_MEDIAN:-2000}"
QUICK_MIN_CORRECT="${QUICK_MIN_CORRECT:-2}"
TASK_SIGMA="${TASK_SIGMA:-0.8}"
PREFILL_CHUNK="${PREFILL_CHUNK:-256}"
PREFILL_ROWS="${PREFILL_ROWS:-16}"
MAX_HOURS="${MAX_HOURS:-8}"
MAX_TASK_HOURS="${MAX_TASK_HOURS:-20}"
MIN_FREE_GIB="${MIN_FREE_GIB:-60}"
KV_HEADROOM_GIB="${KV_HEADROOM_GIB:-16}"
LONG_LEN="${LONG_LEN:-2600}"
LONG_PREFILL="${LONG_PREFILL:-2300}"
MIXED_STEPS="${MIXED_STEPS:-32}"
TIME_BATCHES="${TIME_BATCHES:-1,16,64,144}"
TIME_STEPS="${TIME_STEPS:-20}"
LEVEL="${LEVEL:-0.95}"
BOOT_DRAWS="${BOOT_DRAWS:-10000}"
GRADE_WORKERS="${GRADE_WORKERS:-8}"
TASKS_APPROVED="${TASKS_APPROVED:-0}"
CAPABILITY_APPROVED="${CAPABILITY_APPROVED:-0}"
ACCEPT_PROJECTION="${ACCEPT_PROJECTION:-0}"
ACCEPT_TASK_PROJECTION="${ACCEPT_TASK_PROJECTION:-0}"
WAIT_FOR_DOWNLOAD="${WAIT_FOR_DOWNLOAD:-1}"
DOWNLOAD_TIMEOUT_MIN="${DOWNLOAD_TIMEOUT_MIN:-240}"
if [[ -z "${THREADS:-}" ]]; then  # physical cores: distinct (core, socket) pairs
    THREADS="$( (lscpu -p=CORE,SOCKET 2>/dev/null || true) | { grep -v '^#' || true; } | sort -u | wc -l)"
    if [[ "$THREADS" -lt 1 ]]; then
        THREADS="$(nproc)"
    fi
fi

MERGE_DIR="$DATA_DIR/merge-ta0.5"
B_DIR="$DATA_DIR/b"
E_DIR="$DATA_DIR/e"
E2_DIR="$DATA_DIR/e2"
CRUX_FILE="$DATA_DIR/cruxeval/test.jsonl"
MMLU_TEST="$DATA_DIR/mmlu_pro/data/test-00000-of-00001.parquet"
MMLU_VALIDATION="$DATA_DIR/mmlu_pro/data/validation-00000-of-00001.parquet"
GSM8K_PROMPTS="$PHASE1_DIR/prompts.jsonl"
PHASE1_REPLIES="$PHASE1_DIR/main/replies.jsonl"
SPEC="$OUT_DIR/merge_spec.json"
TASK_PROMPTS="$OUT_DIR/task_prompts.jsonl"
QUICK_PROMPTS="$OUT_DIR/quick_prompts.jsonl"
SELFCHECK_JSON="$OUT_DIR/step17_selfcheck.json"
RUNAWAY_DIR="$OUT_DIR/runaway"
TF_FILE="$OUT_DIR/teacher_forced.jsonl"
TASKS_A_DIR="$OUT_DIR/tasks_A"
TASKS_MERGE_DIR="$OUT_DIR/tasks_merge"
BIAS_PATTERN='model\.layers\.[0-9]+\.mlp\.router\.bias'

mkdir -p "$OUT_DIR/markers" "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/run_phase2.log"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
stop() {
    local code="$1"
    shift
    log "STOP (exit $code): $*"
    exit "$code"
}
on_error() {
    local rc=$? line="$1"
    log "ERROR (exit 1): a command failed with status $rc at line $line of run_phase2.sh"
    exit 1
}
trap 'on_error $LINENO' ERR

done_step() { [[ -f "$OUT_DIR/markers/$1.done" ]]; }
mark() {
    date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT_DIR/markers/$1.done"
    log "$1 done"
}
run_logged() {  # LOGFILE COMMAND...: append the command's output to LOGFILE; return its status
    local logfile="$1"
    shift
    log "running: $* (output in ${logfile#"$OUT_DIR"/})"
    printf '\n### %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$logfile"
    "$@" >>"$logfile" 2>&1
}
field() { "$PY" -m swaptest.jsonfield "$1" "$2"; }   # FILE KEY.PATH of a JSON file
decision() { awk -v k="$2" '$1 == k {print $2}' "$1"; }  # FILE KEY of a decisions file
greater() { awk -v a="$1" -v b="$2" 'BEGIN {exit !(a > b)}'; }  # A > B for decimal numbers

download_running() {
    local f pid
    for f in "$DATA_DIR"/markers/download_*.running "$DATA_DIR"/markers/download2_*.running; do
        [[ -f "$f" ]] || continue
        pid="$(cat "$f")"
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

wait_for() {  # ITEM: wait until a download script has verified DATA_DIR/markers/ITEM.verified
    local item="$1" waited=0 failed
    while [[ ! -f "$DATA_DIR/markers/$item.verified" ]]; do
        failed="$(cd "$DATA_DIR/markers" && ls -- download_*.failed download2_*.failed 2>/dev/null | tr '\n' ' ' || true)"
        if [[ -n "$failed" ]]; then
            stop 20 "input $item: a download failed ($failed); fix it and rerun the download script, then this script"
        fi
        if [[ "$WAIT_FOR_DOWNLOAD" != 1 ]] || ! download_running; then
            stop 20 "input $item is not downloaded and verified, and no download script is running"
        fi
        if ((waited >= DOWNLOAD_TIMEOUT_MIN * 60)); then
            stop 20 "input $item is still not verified after $DOWNLOAD_TIMEOUT_MIN min"
        fi
        if ((waited % 600 == 0)); then
            log "waiting for a download script to verify $item"
        fi
        sleep 60
        waited=$((waited + 60))
    done
}

current_hashes() {  # prints "<task prompts sha> <harness tree sha> <merge spec sha>"
    local tree
    tree="$("$PY" -m swaptest.hashes --prompts "$TASK_PROMPTS" --harness "$HARNESS_DIR" | awk '$1 == "harness_tree_sha256" {print $2}')"
    printf '%s %s %s\n' "$(sha256sum "$TASK_PROMPTS" | cut -d ' ' -f 1)" "$tree" "$(sha256sum "$SPEC" | cut -d ' ' -f 1)"
}

# ---------------------------------------------------------------------------------------------------------------
# preflight

for knob in CAP BATCH_SIZE SEED N_TASK TASK_SEED TASK_CAP TASK_BATCH_SIZE QUICK_PER_TASK QUICK_MAX_CAPPED \
    QUICK_MIN_CORRECT PREFILL_CHUNK PREFILL_ROWS THREADS MIN_FREE_GIB KV_HEADROOM_GIB LONG_LEN LONG_PREFILL \
    MIXED_STEPS TIME_STEPS BOOT_DRAWS GRADE_WORKERS DOWNLOAD_TIMEOUT_MIN TASK_OFFSET QUICK_OFFSET; do
    if [[ ! "${!knob}" =~ ^[0-9]+$ ]]; then
        stop 2 "$knob must be a whole number, got ${!knob}"
    fi
done
for knob in MAX_HOURS MAX_TASK_HOURS QUICK_MAX_MEDIAN TASK_SIGMA LEVEL; do
    if [[ ! "${!knob}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        stop 2 "$knob must be a number, got ${!knob}"
    fi
done
if [[ ! "$ARMS" =~ ^[PNG]+$ || "$ARMS" != *P* ]]; then
    stop 2 "ARMS must be letters from PNG including P, got $ARMS"
fi
if ((BATCH_SIZE % ${#ARMS} != 0)) || [[ ! "$TIME_BATCHES" =~ ^[0-9]+(,[0-9]+)*$ ]] || ((LONG_PREFILL >= LONG_LEN)); then
    stop 2 "BATCH_SIZE must be a multiple of the number of arms, TIME_BATCHES a list of numbers, LONG_PREFILL < LONG_LEN"
fi
if [[ ! -x "$PY" ]]; then
    stop 2 "no Python at $PY; run bootstrap.sh first"
fi
MMLU_ARGS=()
if [[ -n "$MMLU_PRO_IDS" ]]; then
    if [[ ! -s "$MMLU_PRO_IDS" ]]; then
        stop 2 "MMLU_PRO_IDS names $MMLU_PRO_IDS, which is missing or empty"
    fi
    MMLU_ARGS=(--mmlu-pro-ids "$MMLU_PRO_IDS")
fi
for f in "$GSM8K_PROMPTS" "$PHASE1_REPLIES" "$PHASE1_DIR/analysis.json" "$PHASE1_DIR/prereg_hashes.txt"; do
    if [[ ! -s "$f" ]]; then
        stop 21 "phase 1 has not finished: $f is missing (run run_phase1.sh to the end first)"
    fi
done
if [[ "$(field "$PHASE1_DIR/analysis.json" v1.passed)" != true ]]; then
    stop 21 "phase 1's validity gate V1 failed; arm A is not a usable reference, so phase 2 does not run"
fi
N_GSM8K="$(wc -l <"$GSM8K_PROMPTS")"

log "run_phase2.sh starting; harness $HARNESS_DIR, data $DATA_DIR, phase 1 $PHASE1_DIR, results $OUT_DIR"
log "settings: ARMS=$ARMS CAP=$CAP BATCH_SIZE=$BATCH_SIZE SEED=$SEED N_TASK=$N_TASK TASK_SEED=$TASK_SEED \
TASK_OFFSET=$TASK_OFFSET TASK_CAP=$TASK_CAP TASK_BATCH_SIZE=$TASK_BATCH_SIZE MMLU_PRO_IDS=${MMLU_PRO_IDS:-none} \
THREADS=$THREADS MAX_HOURS=$MAX_HOURS MAX_TASK_HOURS=$MAX_TASK_HOURS LEVEL=$LEVEL"
COMMON=(--threads "$THREADS")
RUN_OPTS=(--seed "$SEED" --prefill-chunk "$PREFILL_CHUNK" --prefill-rows "$PREFILL_ROWS" "${COMMON[@]}")

# ---------------------------------------------------------------------------------------------------------------
# step 14: rebuild Merge-TA0.5 and compare it with the download (stop S3)

if ! done_step step14; then
    log "step 14: rebuild Merge-TA0.5 (coefficients 0.5, 0.5) from B, E and E2 and compare it with the download"
    wait_for merge
    wait_for b
    wait_for e
    wait_for e2
    spec_args=(--base "$B_DIR" --donor "$E_DIR" --donor "$E2_DIR" --coefficients 1,1 --widen "$BIAS_PATTERN=bfloat16"
        --arm P --override N attention_gates "$MERGE_DIR" --override G attention_gates "$B_DIR")
    run_logged "$OUT_DIR/logs/step14_merge_check.log" "$PY" -m swaptest.merge "${spec_args[@]}" \
        --write-spec "$OUT_DIR/merge_spec_widened_bias.json"
    rc=0
    run_logged "$OUT_DIR/logs/step14_merge_check.log" "$PY" -m swaptest.merge \
        --spec "$OUT_DIR/merge_spec_widened_bias.json" --coefficients 0.5,0.5 --check-against "$MERGE_DIR" \
        --json "$OUT_DIR/merge_check.json" "${COMMON[@]}" || rc=$?
    case "$rc" in
        0)
            cp "$OUT_DIR/merge_spec_widened_bias.json" "$SPEC"
            echo "equal in every stored value" >"$OUT_DIR/merge_check.txt"
            ;;
        3)
            log "step 14: only the router biases differ, and the download keeps B's values there; rebuilding with \
router-bias coefficients 0 as pre-registered"
            run_logged "$OUT_DIR/logs/step14_merge_check.log" "$PY" -m swaptest.merge "${spec_args[@]}" \
                --tensor-coefficients "$BIAS_PATTERN=0,0" --write-spec "$OUT_DIR/merge_spec_bias_at_b.json"
            rc=0
            run_logged "$OUT_DIR/logs/step14_merge_check.log" "$PY" -m swaptest.merge \
                --spec "$OUT_DIR/merge_spec_bias_at_b.json" --coefficients 0.5,0.5 --check-against "$MERGE_DIR" \
                --json "$OUT_DIR/merge_check_bias_at_b.json" "${COMMON[@]}" || rc=$?
            if ((rc != 0)); then
                stop 30 "S3: with the router biases at B's values the rebuilt Merge-TA0.5 still differs (exit $rc); first difference $(field "$OUT_DIR/merge_check_bias_at_b.json" first_difference)"
            fi
            cp "$OUT_DIR/merge_spec_bias_at_b.json" "$SPEC"
            echo "equal in every stored value, with the router biases at B's values" >"$OUT_DIR/merge_check.txt"
            ;;
        1) stop 30 "S3: the rebuilt Merge-TA0.5 differs from the download; first difference $(field "$OUT_DIR/merge_check.json" first_difference)" ;;
        *) stop 1 "step 14: the merge check ended with status $rc; see logs/step14_merge_check.log" ;;
    esac
    log "step 14: rebuilt Merge-TA0.5 $(cat "$OUT_DIR/merge_check.txt")"
    mark step14
fi

# ---------------------------------------------------------------------------------------------------------------
# step 15: task prompt files

if ! done_step step15; then
    log "step 15: task prompts ($N_TASK per task) and quick-look prompts ($QUICK_PER_TASK per task)"
    wait_for merge_meta
    wait_for task_data
    data_args=(--cruxeval "$CRUX_FILE" --mmlu-pro-test "$MMLU_TEST" --mmlu-pro-validation "$MMLU_VALIDATION"
        --checkpoint "$MERGE_DIR" --seed "$TASK_SEED" "${MMLU_ARGS[@]}")
    run_logged "$OUT_DIR/logs/step15_prompts.log" "$PY" -m swaptest.taskprompts "${data_args[@]}" \
        --out "$TASK_PROMPTS" --count "$N_TASK" --offset "$TASK_OFFSET"
    run_logged "$OUT_DIR/logs/step15_prompts.log" "$PY" -m swaptest.taskprompts "${data_args[@]}" \
        --out "$QUICK_PROMPTS" --count "$QUICK_PER_TASK" --offset "$QUICK_OFFSET"
    mark step15
fi

# ---------------------------------------------------------------------------------------------------------------
# step 16: quick look with arm A (stop S5), then a pause until the replies are read

if ! done_step step16; then
    if ! done_step step16_run; then
        log "step 16: arm A on $(wc -l <"$QUICK_PROMPTS") quick-look prompts, cap $TASK_CAP"
        rc=0
        run_logged "$OUT_DIR/logs/step16_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" \
            --prompts "$QUICK_PROMPTS" --out "$OUT_DIR/quick_A" --arms A --cap "$TASK_CAP" \
            --batch-size "$(wc -l <"$QUICK_PROMPTS")" "${RUN_OPTS[@]}" || rc=$?
        if ((rc != 0)); then
            stop 1 "step 16: the quick-look run ended with status $rc; rerun to resume it"
        fi
        mark step16_run
    fi
    rc=0
    "$PY" -m swaptest.analyze_phase2 quicklook --prompts "$QUICK_PROMPTS" --replies "$OUT_DIR/quick_A/replies.jsonl" \
        --grades "$OUT_DIR/quick_grades.jsonl" --json "$OUT_DIR/quick.json" --max-capped "$QUICK_MAX_CAPPED" \
        --max-median "$QUICK_MAX_MEDIAN" --min-correct "$QUICK_MIN_CORRECT" --workers "$GRADE_WORKERS" \
        >"$OUT_DIR/step16_review.txt" 2>&1 || rc=$?
    case "$rc" in
        0) log "step 16: $(tail -n 1 "$OUT_DIR/step16_review.txt")" ;;
        3) stop 34 "S5: $(tail -n 1 "$OUT_DIR/step16_review.txt"); see step16_review.txt" ;;
        *) stop 1 "step 16: the quick-look analysis ended with status $rc; see step16_review.txt" ;;
    esac
    if [[ "$TASKS_APPROVED" == 1 ]]; then
        log "step 16: the quick-look replies were approved (TASKS_APPROVED=1)"
        mark step16
    else
        stop 35 "PAUSE after step 16: read $OUT_DIR/step16_review.txt; if the prompts, answers and grades are sound, rerun with TASKS_APPROVED=1"
    fi
fi

# ---------------------------------------------------------------------------------------------------------------
# step 17: the merge P in memory with arms N and G; checks, timing and projections (stops at 31, 32, S4)

if ! done_step step17; then
    if ! done_step step17_checks; then
        median="$(field "$OUT_DIR/quick.json" median_length)"
        log "step 17: build P with arms $ARMS in memory; checks and decode timing (task replies: measured median $median tokens)"
        batches="$TIME_BATCHES"
        if [[ ",$batches," != *",$BATCH_SIZE,"* ]]; then
            batches="$batches,$BATCH_SIZE"
        fi
        rm -f "$SELFCHECK_JSON"
        rc=0
        run_logged "$OUT_DIR/logs/step17_selfcheck.log" "$PY" -m swaptest.selfcheck_phase2 --checkpoint "$MERGE_DIR" \
            --merge "$SPEC" --prompts "$GSM8K_PROMPTS" --task-prompts "$TASK_PROMPTS" --arms "$ARMS" \
            --long-len "$LONG_LEN" --prefill-len "$LONG_PREFILL" --chunk "$PREFILL_CHUNK" --mixed-steps "$MIXED_STEPS" \
            --batches "$batches" --steps "$TIME_STEPS" --per-arm "$N_GSM8K" --cap "$CAP" --batch-size "$BATCH_SIZE" \
            --task-cap "$TASK_CAP" --task-batch-size "$TASK_BATCH_SIZE" --task-median "$median" \
            --task-sigma "$TASK_SIGMA" --max-hours "$MAX_HOURS" --min-free-gib "$MIN_FREE_GIB" \
            --kv-headroom-gib "$KV_HEADROOM_GIB" --seed "$SEED" --json "$SELFCHECK_JSON" "${COMMON[@]}" || rc=$?
        case "$rc" in
            0 | 4) mark step17_checks ;;
            3) stop 31 "step 17: building the merge or its arms, or the memory check failed; see logs/step17_selfcheck.log" ;;
            1)
                if [[ -f "$SELFCHECK_JSON" ]] && [[ "$(field "$SELFCHECK_JSON" passed.kv)" == false ]]; then
                    stop 31 "step 17: the planned waves do not fit in memory; lower BATCH_SIZE or TASK_BATCH_SIZE (see logs/step17_selfcheck.log)"
                fi
                if [[ -f "$SELFCHECK_JSON" ]]; then
                    stop 32 "step 17: a check failed ($(field "$SELFCHECK_JSON" passed)); see logs/step17_selfcheck.log"
                fi
                stop 1 "step 17: the self-check crashed (exit 1 without a summary); see logs/step17_selfcheck.log"
                ;;
            *) stop 1 "step 17: the self-check ended with status $rc (crash or out of memory?); see logs/step17_selfcheck.log" ;;
        esac
    fi
    worst="$(field "$SELFCHECK_JSON" speed.projected_hours.P_runs_away)"
    log "projected runaway run: $(printf '%.2f' "$worst") h with P running away; capability runs: one arm \
$(printf '%.2f' "$(field "$SELFCHECK_JSON" speed.projected_task_hours.1_arm)") h, two arms \
$(printf '%.2f' "$(field "$SELFCHECK_JSON" speed.projected_task_hours.2_arms)") h"
    if greater "$worst" "$MAX_HOURS"; then
        if [[ "$ACCEPT_PROJECTION" == 1 ]]; then
            log "the projected runaway run exceeds MAX_HOURS=$MAX_HOURS; continuing because ACCEPT_PROJECTION=1"
        else
            stop 33 "S4: the projected runaway run ($(printf '%.2f' "$worst") h) exceeds MAX_HOURS=$MAX_HOURS"
        fi
    fi
    mark step17
fi

# ---------------------------------------------------------------------------------------------------------------
# step 18: pre-registration

if ! done_step step18; then
    log "step 18: pre-registration"
    read -r task_sha tree_sha spec_sha <<<"$(current_hashes)"
    gsm8k_sha="$(sha256sum "$GSM8K_PROMPTS" | cut -d ' ' -f 1)"
    crux_rev="$(cat "$DATA_DIR/cruxeval/REVISION")"
    mmlu_rev="$(cat "$DATA_DIR/mmlu_pro/REVISION")"
    pool="all test documents"
    if [[ -n "$MMLU_PRO_IDS" ]]; then
        pool="listed ids, file SHA-256 $(sha256sum "$MMLU_PRO_IDS" | cut -d ' ' -f 1)"
    fi
    check="$(cat "$OUT_DIR/merge_check.txt")"
    for value in "$task_sha" "$tree_sha" "$spec_sha" "$gsm8k_sha"; do
        if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
            stop 2 "could not compute the hashes for the pre-registration ($task_sha / $tree_sha / $spec_sha / $gsm8k_sha)"
        fi
    done
    sed -e "s|^\(- Task prompts file SHA-256: \)_* \((task_prompts_sha256)\)$|\1$task_sha \2|" \
        -e "s|^\(- Merge spec SHA-256: \)_* \((merge_spec_sha256)\)$|\1$spec_sha \2|" \
        -e "s|^\(- Harness tree SHA-256: \)_* \((harness_tree_sha256)\)$|\1$tree_sha \2|" \
        -e "s|^\(- GSM8K prompts file SHA-256: \)_* \((gsm8k_prompts_sha256)\)$|\1$gsm8k_sha \2|" \
        -e "s|^\(- CRUXEval revision: \)_* \((cruxeval_revision)\)$|\1$crux_rev \2|" \
        -e "s|^\(- MMLU-Pro revision: \)_* \((mmlu_pro_revision)\)$|\1$mmlu_rev \2|" \
        -e "s|^\(- MMLU-Pro pool: \)_* \((mmlu_pro_pool)\)$|\1$pool \2|" \
        -e "s|^\(- Merge check: \)_* \((merge_check)\)$|\1$check \2|" \
        "$HARNESS_DIR/PREREG_phase2.md" >"$OUT_DIR/prereg_phase2.md.tmp"
    if grep -q '____' "$OUT_DIR/prereg_phase2.md.tmp"; then
        stop 2 "PREREG_phase2.md still holds blanks after filling in; its Record lines do not have the expected form"
    fi
    mv "$OUT_DIR/prereg_phase2.md.tmp" "$OUT_DIR/prereg_phase2.md"
    printf 'task_prompts_sha256 %s\nharness_tree_sha256 %s\nmerge_spec_sha256 %s\n' "$task_sha" "$tree_sha" \
        "$spec_sha" >"$OUT_DIR/prereg_hashes.txt"
    (cd "$OUT_DIR" && sha256sum prereg_phase2.md >prereg_phase2.md.sha256)
    log "prereg_phase2.md SHA-256 $(cut -d ' ' -f 1 "$OUT_DIR/prereg_phase2.md.sha256") (task prompts $task_sha, harness \
tree $tree_sha, merge spec $spec_sha)"
    mark step18
fi

read -r task_sha tree_sha spec_sha <<<"$(current_hashes)"
if [[ "$(printf 'task_prompts_sha256 %s\nharness_tree_sha256 %s\nmerge_spec_sha256 %s' "$task_sha" "$tree_sha" "$spec_sha")" \
    != "$(cat "$OUT_DIR/prereg_hashes.txt")" ]]; then
    stop 2 "the task prompts, the harness or the merge spec changed after the pre-registration (now $task_sha / $tree_sha / $spec_sha)"
fi

# ---------------------------------------------------------------------------------------------------------------
# step 19: runaway run

if ! done_step step19; then
    log "step 19: runaway run, arms $ARMS x $N_GSM8K GSM8K prompts, cap $CAP, batch size $BATCH_SIZE (progress in runaway/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/step19_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --merge "$SPEC" \
        --prompts "$GSM8K_PROMPTS" --out "$RUNAWAY_DIR" --arms "$ARMS" --cap "$CAP" --batch-size "$BATCH_SIZE" \
        "${RUN_OPTS[@]}" || rc=$?
    if ((rc != 0)); then
        stop 36 "step 19: the runaway run ended with status $rc; rerun this script to resume it"
    fi
    if [[ "$(wc -l <"$RUNAWAY_DIR/replies.jsonl")" -ne $((N_GSM8K * ${#ARMS})) ]]; then
        stop 36 "step 19: $(wc -l <"$RUNAWAY_DIR/replies.jsonl") replies, expected $((N_GSM8K * ${#ARMS})); rerun to resume"
    fi
    mark step19
fi

# ---------------------------------------------------------------------------------------------------------------
# step 20: teacher-forced pass

if ! done_step step20; then
    log "step 20: teacher-forced pass of arm A's phase-1 replies under arms $ARMS"
    rc=0
    run_logged "$OUT_DIR/logs/step20_teacher_forced.log" "$PY" -m swaptest.teacher_forced --checkpoint "$MERGE_DIR" \
        --merge "$SPEC" --prompts "$GSM8K_PROMPTS" --replies "$PHASE1_REPLIES" --out "$TF_FILE" --arms "$ARMS" \
        --reference A --chunk "$PREFILL_CHUNK" "${COMMON[@]}" || rc=$?
    if ((rc != 0)); then
        stop 37 "step 20: the teacher-forced pass ended with status $rc; rerun this script to resume it"
    fi
    rc=0
    run_logged "$OUT_DIR/logs/step20_teacher_forced.log" "$PY" -m swaptest.teacher_forced --check --merge "$SPEC" \
        --replies "$PHASE1_REPLIES" --out "$TF_FILE" --arms "$ARMS" --reference A || rc=$?
    if ((rc != 0)); then
        stop 37 "step 20: the teacher-forced output is incomplete or not finite; see logs/step20_teacher_forced.log"
    fi
    mark step20
fi

# ---------------------------------------------------------------------------------------------------------------
# step 21: runaway analysis (stop S6 on the projected capability runs)

if ! done_step step21; then
    log "step 21: runaway analysis"
    rc=0
    "$PY" -m swaptest.analyze_phase2 runaway --prompts "$GSM8K_PROMPTS" --replies "$PHASE1_REPLIES" \
        "$RUNAWAY_DIR/replies.jsonl" --teacher-forced "$TF_FILE" --json "$OUT_DIR/runaway.json" \
        --decisions "$OUT_DIR/runaway_decisions.txt" >"$OUT_DIR/runaway.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 38 "step 21: the runaway analysis ended with status $rc; see runaway.txt"
    fi
    mark step21
fi
log "runaway decisions: $(tr '\n' ' ' <"$OUT_DIR/runaway_decisions.txt")"
TASK_ARMS="$(decision "$OUT_DIR/runaway_decisions.txt" task_arms)"
if [[ "$TASK_ARMS" == none ]]; then
    (cd "$OUT_DIR" && find . -type f ! -name SHA256SUMS ! -path ./logs/run_phase2.log ! -path './markers/*' \
        ! -name '*.tmp' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
    stop 39 "no arm stops normally enough for capability scoring; the runaway result is final (runaway.txt, SHA256SUMS)"
fi
SCORED="${TASK_ARMS//,/}"
TASK_BATCH=$((TASK_BATCH_SIZE / ${#SCORED} * ${#SCORED}))

if ! done_step step21_projection; then
    one="$(field "$SELFCHECK_JSON" speed.projected_task_hours.1_arm)"
    if ((${#SCORED} == 1)); then
        merged="$one"
    else
        merged="$(field "$SELFCHECK_JSON" speed.projected_task_hours.2_arms)"
    fi
    total="$(awk -v a="$one" -v b="$merged" 'BEGIN {printf "%.2f", a + b}')"
    log "projected capability runs: arm A $(printf '%.2f' "$one") h, arms $SCORED $(printf '%.2f' "$merged") h, total $total h"
    if greater "$total" "$MAX_TASK_HOURS"; then
        if [[ "$ACCEPT_TASK_PROJECTION" == 1 ]]; then
            log "the projected capability runs exceed MAX_TASK_HOURS=$MAX_TASK_HOURS; continuing because ACCEPT_TASK_PROJECTION=1"
        else
            stop 40 "S6: the projected capability runs ($total h) exceed MAX_TASK_HOURS=$MAX_TASK_HOURS"
        fi
    fi
    mark step21_projection
fi
if [[ "$CAPABILITY_APPROVED" != 1 ]] && ! done_step step22; then
    stop 44 "PAUSE after step 21: the runaway result is in runaway.txt; the capability runs of A and $SCORED are projected above; rerun with CAPABILITY_APPROVED=1 to start them"
fi

# ---------------------------------------------------------------------------------------------------------------
# steps 22 and 23: capability runs

if ! done_step step22; then
    log "step 22: capability run of arm A on $(wc -l <"$TASK_PROMPTS") task prompts, cap $TASK_CAP (progress in tasks_A/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/step22_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --prompts "$TASK_PROMPTS" \
        --out "$TASKS_A_DIR" --arms A --cap "$TASK_CAP" --batch-size "$TASK_BATCH_SIZE" "${RUN_OPTS[@]}" || rc=$?
    if ((rc != 0)); then
        stop 41 "step 22: the capability run of arm A ended with status $rc; rerun this script to resume it"
    fi
    mark step22
fi

if ! done_step step23; then
    log "step 23: capability run of arms $SCORED on the task prompts, cap $TASK_CAP, batch size $TASK_BATCH (progress in tasks_merge/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/step23_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --merge "$SPEC" \
        --prompts "$TASK_PROMPTS" --out "$TASKS_MERGE_DIR" --arms "$SCORED" --cap "$TASK_CAP" --batch-size "$TASK_BATCH" \
        "${RUN_OPTS[@]}" || rc=$?
    if ((rc != 0)); then
        stop 41 "step 23: the capability run of arms $SCORED ended with status $rc; rerun this script to resume it"
    fi
    mark step23
fi

# ---------------------------------------------------------------------------------------------------------------
# step 24: capability analysis and checksums

if ! done_step step24; then
    log "step 24: capability analysis"
    rc=0
    "$PY" -m swaptest.analyze_phase2 capability --prompts "$TASK_PROMPTS" --replies "$TASKS_A_DIR/replies.jsonl" \
        "$TASKS_MERGE_DIR/replies.jsonl" --runaway "$OUT_DIR/runaway.json" --grades "$OUT_DIR/grades.jsonl" \
        --json "$OUT_DIR/capability.json" --decisions "$OUT_DIR/capability_decisions.txt" --level "$LEVEL" \
        --draws "$BOOT_DRAWS" --seed "$SEED" --workers "$GRADE_WORKERS" >"$OUT_DIR/capability.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 42 "step 24: the capability analysis ended with status $rc; see capability.txt"
    fi
    (cd "$OUT_DIR" && find . -type f ! -name SHA256SUMS ! -path ./logs/run_phase2.log ! -path './markers/*' \
        ! -name '*.tmp' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
    mark step24
fi

tee -a "$LOG" <"$OUT_DIR/capability.txt"
log "capability decisions: $(tr '\n' ' ' <"$OUT_DIR/capability_decisions.txt")"
if [[ "$(decision "$OUT_DIR/capability_decisions.txt" v3)" != passed ]]; then
    stop 43 "V3 failed: arm A's task scores or capped share are outside the gate; no capability claim (results in $OUT_DIR)"
fi
log "phase 2 finished; results in $OUT_DIR (runaway.txt, capability.txt, prereg_phase2.md, SHA256SUMS)"
