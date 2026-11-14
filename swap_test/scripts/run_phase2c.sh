#!/usr/bin/env bash
# Phase 2c of the attention-gate test: the MMLU-Pro target of phase 2's capability test, measured again with the
# zero-shot prompt of Marin's Evalchemy config for the task. PREREG_phase2c.md fixes the design. Phase 2 must have
# finished its capability runs on the same data directory.
#
# Usage: DATA_DIR=/data bash scripts/run_phase2c.sh
#
# Steps (a finished step writes OUT_DIR/markers/<step>.done; a rerun continues after the last finished step):
#   c1  prompt files: the task prompts (phase 2's CRUXEval prompts and the zero-shot MMLU-Pro prompts of phase 2's
#       documents), the run file (those MMLU-Pro prompts alone) and the quick-look file
#   c2  pre-registration: PREREG_phase2c.md filled in with the hashes, its SHA-256 logged
#   c3  quick look: arm A (Merge-TA0.5) on the quick-look prompts; stop S7; the cap rule writes cap.txt
#   c4  arm A on the run file
#   c5  V3 on arm A: its CRUXEval replies of phase 2 and its new MMLU-Pro replies; stop when V3 fails
#   c6  the scored arm (the merge of phase 2's spec) on the run file, with the same cap
#   c7  capability analysis, decision and checksums
#
# Settings (environment variables; defaults in brackets; the defaults are the pre-registered values):
#   DATA_DIR            (required) directory given to the download scripts
#   PHASE2_DIR          phase 2's results [$DATA_DIR/phase2]
#   OUT_DIR             results of phase 2c [$DATA_DIR/phase2c]
#   RUNAWAY             JSON summary of phase 2's runaway analysis [$PHASE2_DIR/runaway.json]
#   VENV_DIR            environment made by bootstrap.sh [$HOME/.venvs/swaptest]; PY its Python [VENV_DIR/bin/python]
#   SCORED              the scored arm [G]
#   N_TASK              MMLU-Pro prompts [270]              TASK_OFFSET   their first draw number [0]
#   QUICK_N             quick-look prompts [48]             QUICK_OFFSET  their first draw number [800]
#   QUICK_MIN_CORRECT   correct quick-look replies needed to go on (stop S7) [21]
#   QUICK_CAP_LIMIT     quick-look replies at the cap from which the raised cap is used [5]
#   TASK_CAP            cap in new tokens [4096]            CAP_RAISED    the raised cap [8192]
#   TASK_BATCH_SIZE     rows per wave of the main runs [270]
#   SEED                sampling and draw seed [9984]
#   PREFILL_CHUNK       slots per prefill chunk [256]       PREFILL_ROWS  rows per prefill block [16]
#   THREADS             torch threads [number of physical cores]
#   LEVEL               level of the composite interval [0.95]   BOOT_DRAWS  bootstrap draws [10000]
#   GRADE_WORKERS       parallel grading processes [8]
#
# Exit codes:
#   0   finished, V3 passed
#   1   a command failed
#   2   a precondition failed (settings, files of phase 2, changed inputs)
#   41  a sampling run failed; rerun this script to resume it
#   42  an analysis failed
#   43  stop: V3 failed (after step c5: the scored arm is not sampled; after step c7: no capability claim)
#   51  stop S7: the quick look has too few correct replies
set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the directory given to the download scripts}"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "DATA_DIR $DATA_DIR is not a directory" >&2
    exit 2
fi
PHASE2_DIR="${PHASE2_DIR:-$DATA_DIR/phase2}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/phase2c}"
RUNAWAY="${RUNAWAY:-$PHASE2_DIR/runaway.json}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/swaptest}"
PY="${PY:-$VENV_DIR/bin/python}"

SCORED="${SCORED:-G}"
N_TASK="${N_TASK:-270}"
TASK_OFFSET="${TASK_OFFSET:-0}"
QUICK_N="${QUICK_N:-48}"
QUICK_OFFSET="${QUICK_OFFSET:-800}"
QUICK_MIN_CORRECT="${QUICK_MIN_CORRECT:-21}"
QUICK_CAP_LIMIT="${QUICK_CAP_LIMIT:-5}"
TASK_CAP="${TASK_CAP:-4096}"
CAP_RAISED="${CAP_RAISED:-8192}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-270}"
SEED="${SEED:-9984}"
PREFILL_CHUNK="${PREFILL_CHUNK:-256}"
PREFILL_ROWS="${PREFILL_ROWS:-16}"
LEVEL="${LEVEL:-0.95}"
BOOT_DRAWS="${BOOT_DRAWS:-10000}"
GRADE_WORKERS="${GRADE_WORKERS:-8}"
if [[ -z "${THREADS:-}" ]]; then  # physical cores: distinct (core, socket) pairs
    THREADS="$( (lscpu -p=CORE,SOCKET 2>/dev/null || true) | { grep -v '^#' || true; } | sort -u | wc -l)"
    if [[ "$THREADS" -lt 1 ]]; then
        THREADS="$(nproc)"
    fi
fi

MERGE_DIR="$DATA_DIR/merge-ta0.5"
SPEC="$PHASE2_DIR/merge_spec.json"
PHASE2_PROMPTS="$PHASE2_DIR/task_prompts.jsonl"
PHASE2_REPLIES_A="$PHASE2_DIR/tasks_A/replies.jsonl"
PHASE2_REPLIES_MERGE="$PHASE2_DIR/tasks_merge/replies.jsonl"
CRUX_FILE="$DATA_DIR/cruxeval/test.jsonl"
MMLU_TEST="$DATA_DIR/mmlu_pro/data/test-00000-of-00001.parquet"
MMLU_VALIDATION="$DATA_DIR/mmlu_pro/data/validation-00000-of-00001.parquet"
TASK_PROMPTS="$OUT_DIR/task_prompts.jsonl"
RUN_PROMPTS="$OUT_DIR/run_prompts.jsonl"
QUICK_PROMPTS="$OUT_DIR/quick_prompts.jsonl"
TASKS_A_DIR="$OUT_DIR/tasks_A"
TASKS_MERGE_DIR="$OUT_DIR/tasks_merge"

mkdir -p "$OUT_DIR/markers" "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/run_phase2c.log"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
stop() {
    local code="$1"
    shift
    log "STOP (exit $code): $*"
    exit "$code"
}
on_error() {
    local rc=$? line="$1"
    log "ERROR (exit 1): a command failed with status $rc at line $line of run_phase2c.sh"
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
sha() { sha256sum "$1" | cut -d ' ' -f 1; }
tree_sha() {  # [DIR]: the harness tree hash of DIR, or of the tree Python imports swaptest from
    "$PY" -m swaptest.hashes --prompts "$PHASE2_PROMPTS" ${1:+--harness "$1"} | awk '$1 == "harness_tree_sha256" {print $2}'
}
record_hashes() {  # the lines of prereg_hashes.txt for the files as they are now
    printf 'task_prompts_sha256 %s\nrun_prompts_sha256 %s\nquick_prompts_sha256 %s\nmerge_spec_sha256 %s\n' \
        "$(sha "$TASK_PROMPTS")" "$(sha "$RUN_PROMPTS")" "$(sha "$QUICK_PROMPTS")" "$(sha "$SPEC")"
    printf 'harness_tree_sha256 %s\nphase2_replies_a_sha256 %s\nphase2_replies_g_sha256 %s\nrunaway_summary_sha256 %s\n' \
        "$(tree_sha "$HARNESS_DIR")" "$(sha "$PHASE2_REPLIES_A")" "$(sha "$PHASE2_REPLIES_MERGE")" "$(sha "$RUNAWAY")"
}

# ---------------------------------------------------------------------------------------------------------------
# preflight

for knob in N_TASK TASK_OFFSET QUICK_N QUICK_OFFSET QUICK_MIN_CORRECT QUICK_CAP_LIMIT TASK_CAP CAP_RAISED \
    TASK_BATCH_SIZE SEED PREFILL_CHUNK PREFILL_ROWS THREADS BOOT_DRAWS GRADE_WORKERS; do
    if [[ ! "${!knob}" =~ ^[0-9]+$ ]]; then
        stop 2 "$knob must be a whole number, got ${!knob}"
    fi
done
if [[ ! "$LEVEL" =~ ^[0-9]+([.][0-9]+)?$ || ! "$SCORED" =~ ^[PNG]$ ]]; then
    stop 2 "LEVEL must be a number and SCORED one of P, N, G (got $LEVEL, $SCORED)"
fi
if ((CAP_RAISED <= TASK_CAP)) || ((QUICK_OFFSET < TASK_OFFSET + N_TASK)); then
    stop 2 "CAP_RAISED must exceed TASK_CAP, and the quick-look draw must start after the task draw"
fi
if [[ ! -x "$PY" ]]; then
    stop 2 "no Python at $PY; run bootstrap.sh first"
fi
for f in "$PHASE2_PROMPTS" "$SPEC" "$RUNAWAY" "$PHASE2_REPLIES_A" "$PHASE2_REPLIES_MERGE" "$CRUX_FILE" "$MMLU_TEST" \
    "$MMLU_VALIDATION" "$DATA_DIR/mmlu_pro/REVISION" "$HARNESS_DIR/PREREG_phase2c.md"; do
    if [[ ! -s "$f" ]]; then
        stop 2 "$f is missing: phase 2 must have finished its capability runs on this data directory"
    fi
done
for item in merge merge_meta b e e2 task_data; do
    if [[ ! -f "$DATA_DIR/markers/$item.verified" ]]; then
        stop 2 "input $item is not downloaded and verified (see download.sh and download_phase2.sh)"
    fi
done
if [[ "$(tree_sha)" != "$(tree_sha "$HARNESS_DIR")" ]]; then
    stop 2 "Python imports swaptest from a tree other than $HARNESS_DIR"
fi
if [[ "$(field "$RUNAWAY" "phase2.eligible.$SCORED")" != true ]]; then
    stop 2 "the runaway summary $RUNAWAY does not name arm $SCORED eligible for capability scoring"
fi

log "run_phase2c.sh starting; harness $HARNESS_DIR, data $DATA_DIR, phase 2 $PHASE2_DIR, results $OUT_DIR"
log "settings: SCORED=$SCORED N_TASK=$N_TASK TASK_OFFSET=$TASK_OFFSET QUICK_N=$QUICK_N QUICK_OFFSET=$QUICK_OFFSET \
QUICK_MIN_CORRECT=$QUICK_MIN_CORRECT QUICK_CAP_LIMIT=$QUICK_CAP_LIMIT TASK_CAP=$TASK_CAP CAP_RAISED=$CAP_RAISED \
TASK_BATCH_SIZE=$TASK_BATCH_SIZE SEED=$SEED THREADS=$THREADS LEVEL=$LEVEL BOOT_DRAWS=$BOOT_DRAWS RUNAWAY=$RUNAWAY"
RUN_OPTS=(--seed "$SEED" --prefill-chunk "$PREFILL_CHUNK" --prefill-rows "$PREFILL_ROWS" --threads "$THREADS")

# ---------------------------------------------------------------------------------------------------------------
# step c1: prompt files

if ! done_step c1; then
    log "step c1: prompt files ($N_TASK zero-shot MMLU-Pro prompts, $QUICK_N quick-look prompts)"
    data_args=(--cruxeval "$CRUX_FILE" --mmlu-pro-test "$MMLU_TEST" --mmlu-pro-validation "$MMLU_VALIDATION"
        --checkpoint "$MERGE_DIR" --seed "$SEED" --mmlu-pro-zero-shot)
    run_logged "$OUT_DIR/logs/c1_prompts.log" "$PY" -m swaptest.taskprompts "${data_args[@]}" \
        --out "$TASK_PROMPTS" --count "$N_TASK" --offset "$TASK_OFFSET"
    run_logged "$OUT_DIR/logs/c1_prompts.log" "$PY" -m swaptest.taskprompts "${data_args[@]}" \
        --out "$RUN_PROMPTS" --count "$N_TASK" --offset "$TASK_OFFSET" --tasks mmlu_pro
    run_logged "$OUT_DIR/logs/c1_prompts.log" "$PY" -m swaptest.taskprompts "${data_args[@]}" \
        --out "$QUICK_PROMPTS" --count "$QUICK_N" --offset "$QUICK_OFFSET" --tasks mmlu_pro
    rc=0
    run_logged "$OUT_DIR/logs/c1_prompts.log" "$PY" -m swaptest.sameprompts --reference "$PHASE2_PROMPTS" \
        --file "$TASK_PROMPTS" --tasks cruxeval_input,cruxeval_output || rc=$?
    if ((rc != 0)); then
        stop 2 "step c1: the CRUXEval prompts differ from phase 2's (see logs/c1_prompts.log)"
    fi
    run_logged "$OUT_DIR/logs/c1_prompts.log" "$PY" -m swaptest.sameprompts --reference "$TASK_PROMPTS" \
        --file "$RUN_PROMPTS" --tasks mmlu_pro || rc=$?
    if ((rc != 0)) || [[ "$(wc -l <"$RUN_PROMPTS")" -ne "$N_TASK" || "$(wc -l <"$QUICK_PROMPTS")" -ne "$QUICK_N" ]]; then
        stop 2 "step c1: the run file or the quick-look file is not as planned (see logs/c1_prompts.log)"
    fi
    mark c1
fi

# ---------------------------------------------------------------------------------------------------------------
# step c2: pre-registration

if ! done_step c2; then
    log "step c2: pre-registration"
    record_hashes >"$OUT_DIR/prereg_hashes.txt.tmp"
    if [[ "$(grep -c -E '^[a-z0-9_]+ [0-9a-f]{64}$' "$OUT_DIR/prereg_hashes.txt.tmp")" -ne 8 ]]; then
        stop 2 "could not compute the hashes for the pre-registration (see prereg_hashes.txt.tmp)"
    fi
    value() { awk -v k="$1" '$1 == k {print $2}' "$OUT_DIR/prereg_hashes.txt.tmp"; }
    mmlu_rev="$(cat "$DATA_DIR/mmlu_pro/REVISION")"
    sed -e "s|^\(- Task prompts file SHA-256: \)_* \((task_prompts_sha256)\)$|\1$(value task_prompts_sha256) \2|" \
        -e "s|^\(- MMLU-Pro run file SHA-256: \)_* \((run_prompts_sha256)\)$|\1$(value run_prompts_sha256) \2|" \
        -e "s|^\(- Quick-look prompts file SHA-256: \)_* \((quick_prompts_sha256)\)$|\1$(value quick_prompts_sha256) \2|" \
        -e "s|^\(- Merge spec SHA-256: \)_* \((merge_spec_sha256)\)$|\1$(value merge_spec_sha256) \2|" \
        -e "s|^\(- Harness tree SHA-256: \)_* \((harness_tree_sha256)\)$|\1$(value harness_tree_sha256) \2|" \
        -e "s|^\(- Phase 2 replies of arm A SHA-256: \)_* \((phase2_replies_a_sha256)\)$|\1$(value phase2_replies_a_sha256) \2|" \
        -e "s|^\(- Phase 2 replies of arm G SHA-256: \)_* \((phase2_replies_g_sha256)\)$|\1$(value phase2_replies_g_sha256) \2|" \
        -e "s|^\(- Runaway summary SHA-256: \)_* \((runaway_summary_sha256)\)$|\1$(value runaway_summary_sha256) \2|" \
        -e "s|^\(- MMLU-Pro revision: \)_* \((mmlu_pro_revision)\)$|\1$mmlu_rev \2|" \
        "$HARNESS_DIR/PREREG_phase2c.md" >"$OUT_DIR/prereg_phase2c.md.tmp"
    if grep -q '____' "$OUT_DIR/prereg_phase2c.md.tmp"; then
        stop 2 "PREREG_phase2c.md still holds blanks after filling in; its Record lines do not have the expected form"
    fi
    mv "$OUT_DIR/prereg_phase2c.md.tmp" "$OUT_DIR/prereg_phase2c.md"
    mv "$OUT_DIR/prereg_hashes.txt.tmp" "$OUT_DIR/prereg_hashes.txt"
    (cd "$OUT_DIR" && sha256sum prereg_phase2c.md >prereg_phase2c.md.sha256)
    log "prereg_phase2c.md SHA-256 $(cut -d ' ' -f 1 "$OUT_DIR/prereg_phase2c.md.sha256")"
    mark c2
fi
if [[ "$(record_hashes)" != "$(cat "$OUT_DIR/prereg_hashes.txt")" ]]; then
    stop 2 "a prompt file, the merge spec, the harness, phase 2's replies or the runaway summary changed after the pre-registration"
fi
(cd "$OUT_DIR" && sha256sum -c prereg_phase2c.md.sha256 >/dev/null) || stop 2 "prereg_phase2c.md changed after it was filled in"

# ---------------------------------------------------------------------------------------------------------------
# step c3: quick look with arm A (stop S7) and the cap rule

if ! done_step c3; then
    log "step c3: arm A on $QUICK_N quick-look prompts, cap $TASK_CAP (progress in quick_A/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/c3_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --prompts "$QUICK_PROMPTS" \
        --out "$OUT_DIR/quick_A" --arms A --cap "$TASK_CAP" --batch-size "$QUICK_N" "${RUN_OPTS[@]}" || rc=$?
    if ((rc != 0)); then
        stop 41 "step c3: the quick-look run ended with status $rc; rerun this script to resume it"
    fi
    rc=0
    "$PY" -m swaptest.analyze_phase2 quicklook --prompts "$QUICK_PROMPTS" --replies "$OUT_DIR/quick_A/replies.jsonl" \
        --grades "$OUT_DIR/quick_grades.jsonl" --json "$OUT_DIR/quick.json" --max-capped "$QUICK_N" \
        --max-median "$CAP_RAISED" --min-correct 0 --workers "$GRADE_WORKERS" >"$OUT_DIR/quick_review.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 42 "step c3: the quick-look analysis ended with status $rc; see quick_review.txt"
    fi
    capped=$(($(field "$OUT_DIR/quick.json" replies) - $(field "$OUT_DIR/quick.json" stopped)))
    cap="$TASK_CAP"
    if ((capped >= QUICK_CAP_LIMIT)); then
        cap="$CAP_RAISED"
    fi
    printf '%s\n' "$cap" >"$OUT_DIR/cap.txt"
    log "step c3: $(field "$OUT_DIR/quick.json" correct.mmlu_pro) of $QUICK_N correct, $capped at the cap; cap of the main runs: $cap"
    mark c3
fi
quick_correct="$(field "$OUT_DIR/quick.json" correct.mmlu_pro)"
if ((quick_correct < QUICK_MIN_CORRECT)); then
    stop 51 "S7: $quick_correct of $QUICK_N quick-look replies are correct (at least $QUICK_MIN_CORRECT needed); no main run"
fi
CAP="$(cat "$OUT_DIR/cap.txt")"
if [[ "$CAP" != "$TASK_CAP" && "$CAP" != "$CAP_RAISED" ]]; then
    stop 2 "cap.txt holds $CAP, neither $TASK_CAP nor $CAP_RAISED"
fi

# ---------------------------------------------------------------------------------------------------------------
# step c4: arm A on the run file

if ! done_step c4; then
    log "step c4: arm A on $(wc -l <"$RUN_PROMPTS") MMLU-Pro prompts, cap $CAP (progress in tasks_A/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/c4_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --prompts "$RUN_PROMPTS" \
        --out "$TASKS_A_DIR" --arms A --cap "$CAP" --batch-size "$TASK_BATCH_SIZE" "${RUN_OPTS[@]}" || rc=$?
    if ((rc != 0)); then
        stop 41 "step c4: the run of arm A ended with status $rc; rerun this script to resume it"
    fi
    mark c4
fi

# ---------------------------------------------------------------------------------------------------------------
# step c5: V3 on arm A before the scored arm is sampled

ANALYSIS=(--prompts "$TASK_PROMPTS" --runaway "$RUNAWAY" --grades "$OUT_DIR/grades.jsonl" --level "$LEVEL"
    --draws "$BOOT_DRAWS" --seed "$SEED" --workers "$GRADE_WORKERS")
if ! done_step c5; then
    log "step c5: V3 on arm A"
    rc=0
    "$PY" -m swaptest.analyze_phase2 capability "${ANALYSIS[@]}" --arms "" --replies "$PHASE2_REPLIES_A" \
        "$TASKS_A_DIR/replies.jsonl" --json "$OUT_DIR/v3_arm_A.json" --decisions "$OUT_DIR/v3_arm_A_decisions.txt" \
        >"$OUT_DIR/v3_arm_A.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 42 "step c5: the analysis of arm A ended with status $rc; see v3_arm_A.txt"
    fi
    mark c5
fi
tee -a "$LOG" <"$OUT_DIR/v3_arm_A.txt"
if [[ "$(decision "$OUT_DIR/v3_arm_A_decisions.txt" v3)" != passed ]]; then
    stop 43 "V3 failed on arm A: arm $SCORED is not sampled; no capability claim (results in $OUT_DIR)"
fi

# ---------------------------------------------------------------------------------------------------------------
# step c6: the scored arm on the run file

if ! done_step c6; then
    log "step c6: arm $SCORED on the MMLU-Pro prompts, cap $CAP (progress in tasks_merge/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/c6_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --merge "$SPEC" \
        --prompts "$RUN_PROMPTS" --out "$TASKS_MERGE_DIR" --arms "$SCORED" --cap "$CAP" \
        --batch-size "$TASK_BATCH_SIZE" "${RUN_OPTS[@]}" || rc=$?
    if ((rc != 0)); then
        stop 41 "step c6: the run of arm $SCORED ended with status $rc; rerun this script to resume it"
    fi
    mark c6
fi

# ---------------------------------------------------------------------------------------------------------------
# step c7: capability analysis and checksums

if ! done_step c7; then
    log "step c7: capability analysis"
    rc=0
    "$PY" -m swaptest.analyze_phase2 capability "${ANALYSIS[@]}" --arms "$SCORED" --replies "$PHASE2_REPLIES_A" \
        "$PHASE2_REPLIES_MERGE" "$TASKS_A_DIR/replies.jsonl" "$TASKS_MERGE_DIR/replies.jsonl" \
        --json "$OUT_DIR/capability.json" --decisions "$OUT_DIR/capability_decisions.txt" \
        >"$OUT_DIR/capability.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 42 "step c7: the capability analysis ended with status $rc; see capability.txt"
    fi
    (cd "$OUT_DIR" && find . -type f ! -name SHA256SUMS ! -path ./logs/run_phase2c.log ! -path './markers/*' \
        ! -name '*.tmp' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
    mark c7
fi

tee -a "$LOG" <"$OUT_DIR/capability.txt"
log "capability decisions: $(tr '\n' ' ' <"$OUT_DIR/capability_decisions.txt")"
if [[ "$(decision "$OUT_DIR/capability_decisions.txt" v3)" != passed ]]; then
    stop 43 "V3 failed: no capability claim (results in $OUT_DIR)"
fi
log "phase 2c finished; results in $OUT_DIR (capability.txt, prereg_phase2c.md, SHA256SUMS)"
