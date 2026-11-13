#!/usr/bin/env bash
# Phase 1 of the swap test on one machine: steps 5 to 12 of the plan, in order, resumable.
#
#   5   prompt file, unit tests, validation on the small stand-in checkpoint        (stop S0)
#   6   load the 67B with the patch overrides; memory and patch checks              (one process with step 7)
#   7   67B checks and decode timing; projected duration of the main run            (stop S1)
#   8   arm A quick look on 8 prompts; then a pause until the replies are read      (stop S2)
#   9   pre-registration: PREREG.md filled in with the hashes, its SHA-256 logged
#   10  main run: every arm on every prompt (resumable inside the step)
#   11  teacher-forced pass on arm A's replies under every arm
#   12  analysis: validity gate V1 and the decision for every arm
#
# Every finished step writes OUT_DIR/markers/<step>.done and is skipped when that marker exists, so after an
# interruption the same command continues where it stopped. Output of each step goes to OUT_DIR/logs/; this
# script's own log is OUT_DIR/logs/run_phase1.log. Nothing here schedules, cancels or performs a power-off.
#
# Usage: DATA_DIR=/data bash scripts/run_phase1.sh
#
# Settings (environment variables; defaults in brackets):
#   DATA_DIR            directory given to download.sh (required)
#   OUT_DIR             results [DATA_DIR/phase1]
#   VENV_DIR            environment made by bootstrap.sh [$HOME/.venvs/swaptest]
#   N_PROMPTS           prompts per arm [48]            PROMPT_SEED    seed of the prompt draw [9984]
#   CAP                 new-token cap [4096]            SEED           sampling seed [9984]
#   ARMS                arms of the main run [ABCD]     BATCH_SIZE     rows per wave of the main run [192]
#   THREADS             torch threads [number of physical cores]
#   PREFILL_CHUNK       slots per prefill chunk [256]   PREFILL_ROWS   rows per prefill block [16]
#   MAX_HOURS           stop S1 above this projected main-run duration [8]
#   MIN_FREE_GIB        memory that must stay available with the 67B loaded [80]
#   QUICK_PROMPTS       prompts of the step-8 quick look [8]
#   QUICK_MIN_STOPPED   replies of the quick look that must stop (stop S2) [6]
#   LONG_LEN            tokens of the cached-against-full check in steps 5 and 7 [2600]
#   LONG_PREFILL        of which prefilled before single-token steps [2300]
#   MIXED_STEPS         decode steps of the mixed-arm check [32]
#   TIME_BATCHES        batch sizes timed in step 7 (BATCH_SIZE is added) [1,16,64,192]
#   TIME_STEPS          decode steps timed per batch size [20]
#   SKYRL_SRC           source of the reference model for the cross-checks [DATA_DIR/skyrl_src]
#   SKIP_SKYRL_CHECK=1  run without the cross-checks against the reference model (the skip is logged)
#   STEP8_APPROVED=1    the quick-look replies were read and are worked GSM8K solutions: continue past step 8
#   ACCEPT_PROJECTION=1 continue although the projected main run exceeds MAX_HOURS (the decision is logged)
#   WAIT_FOR_DOWNLOAD   1 waits for a running download.sh to verify an input, 0 stops at once [1]
#   DOWNLOAD_TIMEOUT_MIN  longest wait for one input [240]
#
# Exit codes:
#   0   all steps done and V1 passed
#   1   unexpected error (the log names the command and line), or a step's program crashed
#   2   missing input, bad setting, or harness or prompt file changed after the pre-registration
#   10  stop S0: the prompt file or the stand-in validation failed (step 5)
#   11  stop at step 6: loading, patch or memory check failed
#   12  stop at step 7: a 67B check failed (template, reference parity, cached decode, mixed arms)
#   13  stop S1: the projected main run exceeds MAX_HOURS
#   14  stop S2: fewer than QUICK_MIN_STOPPED of the quick-look replies stopped
#   15  pause after step 8: read the quick-look replies, then rerun with STEP8_APPROVED=1
#   16  step 10: the main run did not finish (rerun to resume)
#   17  step 11: the teacher-forced pass failed or its output is incomplete (rerun to resume)
#   18  step 12: the analysis failed
#   19  stop: V1 failed (the analysis is written; no arm is interpreted)
#   20  an input is not downloaded: download failed, not started, or timed out
set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the directory given to download.sh}"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "DATA_DIR $DATA_DIR is not a directory" >&2
    exit 2
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
OUT_DIR="${OUT_DIR:-$DATA_DIR/phase1}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/swaptest}"
PY="$VENV_DIR/bin/python"

N_PROMPTS="${N_PROMPTS:-48}"
PROMPT_SEED="${PROMPT_SEED:-9984}"
CAP="${CAP:-4096}"
SEED="${SEED:-9984}"
ARMS="${ARMS:-ABCD}"
BATCH_SIZE="${BATCH_SIZE:-192}"
PREFILL_CHUNK="${PREFILL_CHUNK:-256}"
PREFILL_ROWS="${PREFILL_ROWS:-16}"
MAX_HOURS="${MAX_HOURS:-8}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
QUICK_PROMPTS="${QUICK_PROMPTS:-8}"
QUICK_MIN_STOPPED="${QUICK_MIN_STOPPED:-6}"
LONG_LEN="${LONG_LEN:-2600}"
LONG_PREFILL="${LONG_PREFILL:-2300}"
MIXED_STEPS="${MIXED_STEPS:-32}"
TIME_BATCHES="${TIME_BATCHES:-1,16,64,192}"
TIME_STEPS="${TIME_STEPS:-20}"
SKYRL_SRC="${SKYRL_SRC:-$DATA_DIR/skyrl_src}"
SKIP_SKYRL_CHECK="${SKIP_SKYRL_CHECK:-0}"
STEP8_APPROVED="${STEP8_APPROVED:-0}"
ACCEPT_PROJECTION="${ACCEPT_PROJECTION:-0}"
WAIT_FOR_DOWNLOAD="${WAIT_FOR_DOWNLOAD:-1}"
DOWNLOAD_TIMEOUT_MIN="${DOWNLOAD_TIMEOUT_MIN:-240}"
if [[ -z "${THREADS:-}" ]]; then  # physical cores: distinct (core, socket) pairs
    THREADS="$( (lscpu -p=CORE,SOCKET 2>/dev/null || true) | { grep -v '^#' || true; } | sort -u | wc -l)"
    if [[ "$THREADS" -lt 1 ]]; then
        THREADS="$(nproc)"
    fi
fi

MERGE_DIR="$DATA_DIR/merge-ta0.5"
SMALL_DIR="$DATA_DIR/d512-sft"
PATCH_FILE="$DATA_DIR/patch/gates_full_strength.safetensors"
GSM8K_DIR="$DATA_DIR/gsm8k"
PROMPTS="$OUT_DIR/prompts.jsonl"
QUICK_DIR="$OUT_DIR/step08_quick_A"
MAIN_DIR="$OUT_DIR/main"
TF_FILE="$OUT_DIR/teacher_forced.jsonl"
SELFCHECK_JSON="$OUT_DIR/step07_selfcheck.json"

mkdir -p "$OUT_DIR/markers" "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/run_phase1.log"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
stop() {
    local code="$1"
    shift
    log "STOP (exit $code): $*"
    exit "$code"
}
on_error() {
    local rc=$? line="$1"
    log "ERROR (exit 1): a command failed with status $rc at line $line of run_phase1.sh"
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
in_harness() { (cd "$HARNESS_DIR" && "$@"); }
json_get() {  # FILE PYTHON-EXPRESSION over d, the parsed JSON
    "$PY" -c 'import json, sys; d = json.load(open(sys.argv[1], encoding="utf-8")); print(eval(sys.argv[2]))' "$1" "$2"
}

poweroff_note() {
    local raw at left
    if [[ -r /run/poweroff_at ]]; then
        raw="$(head -n 1 /run/poweroff_at)"
        if [[ "$raw" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
            at="${raw//[[:space:]]/}"
        else
            at="$(date -u -d "$raw" +%s 2>/dev/null || true)"
        fi
        if [[ -n "$at" ]]; then
            left=$((at - $(date -u +%s)))
            log "power-off planned at $(date -u -d "@$at" '+%Y-%m-%d %H:%M UTC'): $((left / 3600))h $(((left % 3600) / 60))m left"
        else
            log "power-off file /run/poweroff_at holds \"$raw\""
        fi
    fi
}

download_running() {
    local f pid
    for f in "$DATA_DIR"/markers/download_*.running; do
        [[ -f "$f" ]] || continue
        pid="$(cat "$f")"
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

wait_for() {  # ITEM: wait until download.sh has verified DATA_DIR/markers/ITEM.verified
    local item="$1" waited=0 failed
    while [[ ! -f "$DATA_DIR/markers/$item.verified" ]]; do
        failed="$(cd "$DATA_DIR/markers" && ls -- download_*.failed 2>/dev/null | tr '\n' ' ' || true)"
        if [[ -n "$failed" ]]; then
            stop 20 "input $item: a download failed ($failed); fix it and rerun download.sh, then this script"
        fi
        if [[ "$WAIT_FOR_DOWNLOAD" != 1 ]] || ! download_running; then
            stop 20 "input $item is not downloaded and verified, and no download.sh is running"
        fi
        if ((waited >= DOWNLOAD_TIMEOUT_MIN * 60)); then
            stop 20 "input $item is still not verified after $DOWNLOAD_TIMEOUT_MIN min"
        fi
        if ((waited % 600 == 0)); then
            log "waiting for download.sh to verify $item"
        fi
        sleep 60
        waited=$((waited + 60))
    done
}

current_hashes() {  # prints "<prompts sha> <harness tree sha>"
    local out
    out="$("$PY" -m swaptest.hashes --prompts "$PROMPTS" --harness "$HARNESS_DIR")"
    printf '%s %s\n' "$(awk '$1 == "prompts_sha256" {print $2}' <<<"$out")" \
        "$(awk '$1 == "harness_tree_sha256" {print $2}' <<<"$out")"
}

# ---------------------------------------------------------------------------------------------------------------
# preflight

for knob in N_PROMPTS PROMPT_SEED CAP SEED BATCH_SIZE PREFILL_CHUNK PREFILL_ROWS QUICK_PROMPTS QUICK_MIN_STOPPED \
    THREADS MIN_FREE_GIB DOWNLOAD_TIMEOUT_MIN LONG_LEN LONG_PREFILL MIXED_STEPS TIME_STEPS; do
    if [[ ! "${!knob}" =~ ^[0-9]+$ ]] || [[ "${!knob}" -eq 0 && "$knob" != MIN_FREE_GIB ]]; then
        stop 2 "$knob must be a positive whole number, got ${!knob}"
    fi
done
if [[ ! "$MAX_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    stop 2 "MAX_HOURS must be a number, got $MAX_HOURS"
fi
if [[ ! "$TIME_BATCHES" =~ ^[0-9]+(,[0-9]+)*$ ]] || ((LONG_PREFILL >= LONG_LEN)); then
    stop 2 "TIME_BATCHES must be a comma-separated list of numbers and LONG_PREFILL smaller than LONG_LEN"
fi
if [[ ! "$ARMS" =~ ^[ABCD]+$ || "$ARMS" != *A* ]]; then
    stop 2 "ARMS must be letters from ABCD including A, got $ARMS"
fi
if ((BATCH_SIZE % ${#ARMS} != 0)); then
    stop 2 "BATCH_SIZE ($BATCH_SIZE) must be a multiple of the number of arms (${#ARMS}) so that the arms of a prompt share a wave"
fi
if [[ ! -x "$PY" ]]; then
    stop 2 "no Python environment at $VENV_DIR; run bootstrap.sh first"
fi
if [[ "$SKIP_SKYRL_CHECK" == 1 ]]; then
    log "SKIP_SKYRL_CHECK=1: the cross-checks against the reference model (steps 5b and 7a) are skipped"
    SKYRL_ARGS=()
else
    SKYRL_ARGS=(--skyrl-src "$SKYRL_SRC")
fi

log "run_phase1.sh starting; harness $HARNESS_DIR, data $DATA_DIR, results $OUT_DIR"
log "settings: N_PROMPTS=$N_PROMPTS PROMPT_SEED=$PROMPT_SEED CAP=$CAP SEED=$SEED ARMS=$ARMS BATCH_SIZE=$BATCH_SIZE \
THREADS=$THREADS PREFILL_CHUNK=$PREFILL_CHUNK PREFILL_ROWS=$PREFILL_ROWS MAX_HOURS=$MAX_HOURS MIN_FREE_GIB=$MIN_FREE_GIB \
QUICK_PROMPTS=$QUICK_PROMPTS QUICK_MIN_STOPPED=$QUICK_MIN_STOPPED"
poweroff_note
COMMON=(--threads "$THREADS")

# ---------------------------------------------------------------------------------------------------------------
# step 5: prompt file, unit tests, stand-in validation (stop S0)

if ! done_step step05; then
    log "step 5: prompts, unit tests and validation on the stand-in"
    wait_for gsm8k
    wait_for merge_meta
    wait_for d512
    if [[ "$SKIP_SKYRL_CHECK" != 1 ]]; then
        wait_for skyrl
    fi
    if ! done_step prompts; then
        gsm8k_files=("$GSM8K_DIR"/main/test-*.parquet)
        if ((${#gsm8k_files[@]} != 1)) || [[ ! -f "${gsm8k_files[0]}" ]]; then
            stop 2 "expected exactly one GSM8K file $GSM8K_DIR/main/test-*.parquet"
        fi
        gsm8k="${gsm8k_files[0]}"
        rc=0
        run_logged "$OUT_DIR/logs/step05_prompts.log" "$PY" -m swaptest.prompts --gsm8k "$gsm8k" \
            --checkpoint "$MERGE_DIR" --out "$PROMPTS" --count "$N_PROMPTS" --seed "$PROMPT_SEED" || rc=$?
        if ((rc != 0)); then
            stop 10 "S0: the prompt file could not be built (exit $rc; template or tokenizer check); see logs/step05_prompts.log"
        fi
        mark prompts
    fi
    rc=0
    run_logged "$OUT_DIR/logs/step05a_unit_tests.log" in_harness "$PY" -m pytest -q -p no:cacheprovider || rc=$?
    if ((rc != 0)); then
        stop 10 "S0: unit tests failed (step 5a); see logs/step05a_unit_tests.log"
    fi
    rc=0
    run_logged "$OUT_DIR/logs/step05_validate_small.log" "$PY" -m swaptest.validate_small --checkpoint "$SMALL_DIR" \
        --record "$SMALL_DIR/vllm_validation.json" --prompts "$PROMPTS" --long-len "$LONG_LEN" \
        --prefill-len "$LONG_PREFILL" --chunk "$PREFILL_CHUNK" "${SKYRL_ARGS[@]}" "${COMMON[@]}" || rc=$?
    if ((rc != 0)); then
        stop 10 "S0: stand-in validation failed (step 5, exit $rc); see logs/step05_validate_small.log"
    fi
    mark step05
fi
if [[ "$(wc -l <"$PROMPTS")" -ne "$N_PROMPTS" ]]; then
    stop 2 "$PROMPTS holds $(wc -l <"$PROMPTS") prompts, N_PROMPTS is $N_PROMPTS; use a new OUT_DIR to change it"
fi

# ---------------------------------------------------------------------------------------------------------------
# steps 6 and 7: load the 67B, checks, timing and projection (one process; stops at step 6, step 7 and S1)

if ! done_step step07; then
    if ! done_step step07_checks; then
        log "steps 6 and 7: load the 67B with the patch, checks and decode timing"
        wait_for patch
        wait_for merge
        batches="$TIME_BATCHES"
        if [[ ",$batches," != *",$BATCH_SIZE,"* ]]; then
            batches="$batches,$BATCH_SIZE"
        fi
        rm -f "$SELFCHECK_JSON"
        rc=0
        run_logged "$OUT_DIR/logs/step07_selfcheck.log" "$PY" -m swaptest.selfcheck --checkpoint "$MERGE_DIR" \
            --patch "$PATCH_FILE" --prompts "$PROMPTS" "${SKYRL_ARGS[@]}" --batches "$batches" --steps "$TIME_STEPS" \
            --long-len "$LONG_LEN" --prefill-len "$LONG_PREFILL" --mixed-steps "$MIXED_STEPS" --arms "$ARMS" \
            --per-arm "$N_PROMPTS" --cap "$CAP" --batch-size "$BATCH_SIZE" --max-hours "$MAX_HOURS" \
            --min-free-gib "$MIN_FREE_GIB" --seed "$SEED" --chunk "$PREFILL_CHUNK" --json "$SELFCHECK_JSON" \
            "${COMMON[@]}" || rc=$?
        case "$rc" in
            0 | 4)
                mark step06
                mark step07_checks
                ;;
            3) stop 11 "step 6: loading, patch or memory check failed; see logs/step07_selfcheck.log" ;;
            1)
                if [[ -f "$SELFCHECK_JSON" ]] && [[ "$(json_get "$SELFCHECK_JSON" 'd["passed"].get("load", False)')" == True ]]; then
                    mark step06
                    stop 12 "step 7: a 67B check failed ($(json_get "$SELFCHECK_JSON" '[k for k, v in d["passed"].items() if not v]')); see logs/step07_selfcheck.log"
                fi
                stop 1 "steps 6 and 7: the self-check crashed (exit 1 without a summary); see logs/step07_selfcheck.log"
                ;;
            *) stop 1 "steps 6 and 7: the self-check ended with status $rc (crash or out of memory?); see logs/step07_selfcheck.log" ;;
        esac
    fi
    hours="$(json_get "$SELFCHECK_JSON" '", ".join(f"{k}: {v:.2f} h" for k, v in d["speed"]["projected_hours"].items())')"
    worst="$(json_get "$SELFCHECK_JSON" 'next(iter(d["speed"]["projected_hours"].values()))')"
    shown="$(printf '%.2f' "$worst")"
    log "projected main run: $hours"
    if "$PY" -c 'import sys; sys.exit(not float(sys.argv[1]) > float(sys.argv[2]))' "$worst" "$MAX_HOURS"; then
        if [[ "$ACCEPT_PROJECTION" == 1 ]]; then
            log "the projection ($shown h) exceeds MAX_HOURS=$MAX_HOURS; continuing because ACCEPT_PROJECTION=1"
        else
            stop 13 "S1: the projected main run ($shown h with arms B and D running away) exceeds MAX_HOURS=$MAX_HOURS"
        fi
    fi
    mark step07
fi

# ---------------------------------------------------------------------------------------------------------------
# step 8: arm A quick look (stop S2), then a pause until the replies are read

if ! done_step step08; then
    if ! done_step step08_run; then
        log "step 8: arm A alone on $QUICK_PROMPTS prompts, cap $CAP"
        rc=0
        run_logged "$OUT_DIR/logs/step08_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --prompts "$PROMPTS" \
            --out "$QUICK_DIR" --arms A --limit "$QUICK_PROMPTS" --cap "$CAP" --batch-size "$QUICK_PROMPTS" \
            --seed "$SEED" --prefill-chunk "$PREFILL_CHUNK" --prefill-rows "$PREFILL_ROWS" "${COMMON[@]}" || rc=$?
        if ((rc != 0)); then
            stop 1 "step 8: the quick-look run ended with status $rc; rerun to resume it"
        fi
        mark step08_run
    fi
    rc=0
    "$PY" -m swaptest.analyze --prompts "$PROMPTS" --replies "$QUICK_DIR/replies.jsonl" --limit "$QUICK_PROMPTS" \
        --min-stopped "$QUICK_MIN_STOPPED" --show-replies A --json "$OUT_DIR/step08_summary.json" \
        >"$OUT_DIR/step08_review.txt" 2>&1 || rc=$?
    stopped="$(json_get "$OUT_DIR/step08_summary.json" 'd["arms"]["A"]["n"] - d["arms"]["A"]["capped"]' 2>/dev/null || echo "?")"
    case "$rc" in
        0) log "step 8: $stopped of $QUICK_PROMPTS replies stopped; replies in step08_review.txt" ;;
        3) stop 14 "S2: only $stopped of $QUICK_PROMPTS arm A replies stopped (need $QUICK_MIN_STOPPED); see step08_review.txt" ;;
        *) stop 1 "step 8: the quick-look analysis ended with status $rc; see step08_review.txt" ;;
    esac
    if [[ "$STEP8_APPROVED" == 1 ]]; then
        log "step 8: the quick-look replies were approved (STEP8_APPROVED=1)"
        mark step08
    else
        stop 15 "PAUSE after step 8: read $OUT_DIR/step08_review.txt; if the replies are worked GSM8K solutions, rerun with STEP8_APPROVED=1"
    fi
fi

# ---------------------------------------------------------------------------------------------------------------
# step 9: pre-registration

if ! done_step step09; then
    log "step 9: pre-registration"
    read -r prompts_sha tree_sha <<<"$(current_hashes)"
    gsm8k_rev="$(cat "$GSM8K_DIR/REVISION")"
    if [[ ! "$prompts_sha" =~ ^[0-9a-f]{64}$ || ! "$tree_sha" =~ ^[0-9a-f]{64}$ || ! "$gsm8k_rev" =~ ^[0-9a-f]{40}$ ]]; then
        stop 2 "could not read the hashes or the GSM8K revision ($prompts_sha / $tree_sha / $gsm8k_rev)"
    fi
    sed -e "s/^\(- Prompts file SHA-256: \)_* \((prompts_sha256)\)$/\1$prompts_sha \2/" \
        -e "s/^\(- Harness tree SHA-256: \)_* \((harness_tree_sha256)\)$/\1$tree_sha \2/" \
        -e "s/^\(- GSM8K dataset revision: \)_* \((gsm8k_revision)\)$/\1$gsm8k_rev \2/" \
        "$HARNESS_DIR/PREREG.md" >"$OUT_DIR/prereg.md.tmp"
    if grep -q '____' "$OUT_DIR/prereg.md.tmp"; then
        stop 2 "PREREG.md still holds blanks after filling in; its Record lines do not have the expected form"
    fi
    mv "$OUT_DIR/prereg.md.tmp" "$OUT_DIR/prereg.md"
    printf 'prompts_sha256 %s\nharness_tree_sha256 %s\ngsm8k_revision %s\n' "$prompts_sha" "$tree_sha" "$gsm8k_rev" \
        >"$OUT_DIR/prereg_hashes.txt"
    (cd "$OUT_DIR" && sha256sum prereg.md >prereg.md.sha256)
    prereg_sha="$(cut -d ' ' -f 1 "$OUT_DIR/prereg.md.sha256")"
    mkdir -p "$MAIN_DIR"
    printf '%s pre-registration prereg.md SHA-256 %s (prompts %s, harness tree %s, GSM8K %s)\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$prereg_sha" "$prompts_sha" "$tree_sha" "$gsm8k_rev" >>"$MAIN_DIR/run.log"
    log "prereg.md SHA-256 $prereg_sha (prompts $prompts_sha, harness tree $tree_sha, GSM8K $gsm8k_rev)"
    mark step09
fi

# ---------------------------------------------------------------------------------------------------------------
# step 10: main run

if ! done_step step10; then
    read -r prompts_sha tree_sha <<<"$(current_hashes)"
    if [[ "prompts_sha256 $prompts_sha" != "$(sed -n 1p "$OUT_DIR/prereg_hashes.txt")" ||
        "harness_tree_sha256 $tree_sha" != "$(sed -n 2p "$OUT_DIR/prereg_hashes.txt")" ]]; then
        stop 2 "the prompt file or the harness changed after the pre-registration (now $prompts_sha / $tree_sha); see prereg_hashes.txt"
    fi
    log "step 10: main run, arms $ARMS x $N_PROMPTS prompts, cap $CAP, batch size $BATCH_SIZE (progress in main/run.log)"
    poweroff_note
    rc=0
    run_logged "$OUT_DIR/logs/step10_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --patch "$PATCH_FILE" \
        --prompts "$PROMPTS" --out "$MAIN_DIR" --arms "$ARMS" --cap "$CAP" --batch-size "$BATCH_SIZE" --seed "$SEED" \
        --prefill-chunk "$PREFILL_CHUNK" --prefill-rows "$PREFILL_ROWS" "${COMMON[@]}" || rc=$?
    if ((rc != 0)); then
        stop 16 "step 10: the main run ended with status $rc; rerun this script to resume it"
    fi
    rc=0
    "$PY" -m swaptest.analyze --prompts "$PROMPTS" --replies "$MAIN_DIR/replies.jsonl" \
        >"$OUT_DIR/logs/step10_check.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 16 "step 10: the reply file is incomplete (analysis status $rc); rerun this script to resume"
    fi
    log "step 10: $(wc -l <"$MAIN_DIR/replies.jsonl") replies"
    mark step10
fi

# ---------------------------------------------------------------------------------------------------------------
# step 11: teacher-forced pass

if ! done_step step11; then
    log "step 11: teacher-forced pass on arm A's replies under arms $ARMS"
    rc=0
    run_logged "$OUT_DIR/logs/step11_teacher_forced.log" "$PY" -m swaptest.teacher_forced --checkpoint "$MERGE_DIR" \
        --patch "$PATCH_FILE" --prompts "$PROMPTS" --replies "$MAIN_DIR/replies.jsonl" --out "$TF_FILE" \
        --arms "$ARMS" --reference A --chunk "$PREFILL_CHUNK" "${COMMON[@]}" || rc=$?
    if ((rc != 0)); then
        stop 17 "step 11: the teacher-forced pass ended with status $rc; rerun this script to resume it"
    fi
    rc=0
    run_logged "$OUT_DIR/logs/step11_teacher_forced.log" "$PY" -m swaptest.teacher_forced --check \
        --replies "$MAIN_DIR/replies.jsonl" --out "$TF_FILE" --arms "$ARMS" --reference A || rc=$?
    if ((rc != 0)); then
        stop 17 "step 11: the teacher-forced output is incomplete or not finite; see logs/step11_teacher_forced.log"
    fi
    mark step11
fi

# ---------------------------------------------------------------------------------------------------------------
# step 12: analysis

if ! done_step step12; then
    log "step 12: analysis"
    rc=0
    "$PY" -m swaptest.analyze --prompts "$PROMPTS" --replies "$MAIN_DIR/replies.jsonl" --teacher-forced "$TF_FILE" \
        --json "$OUT_DIR/analysis.json" >"$OUT_DIR/analysis.txt" 2>&1 || rc=$?
    if ((rc != 0)); then
        stop 18 "step 12: the analysis ended with status $rc; see analysis.txt"
    fi
    (cd "$OUT_DIR" && find . -type f ! -name SHA256SUMS ! -path ./logs/run_phase1.log ! -path './markers/*' \
        ! -name '*.tmp' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
    mark step12
fi

tee -a "$LOG" <"$OUT_DIR/analysis.txt"
log "decisions: $(json_get "$OUT_DIR/analysis.json" '", ".join(a + ": " + c["outcome"] for a, c in d["comparisons"].items()) or "none"')"
if [[ "$(json_get "$OUT_DIR/analysis.json" 'd["v1"]["passed"]')" != True ]]; then
    stop 19 "V1 failed (R_A $(json_get "$OUT_DIR/analysis.json" 'round(d["v1"]["rate"], 3)'), accuracy $(json_get "$OUT_DIR/analysis.json" 'round(d["v1"]["accuracy"], 3)')): no arm is interpreted; results in $OUT_DIR"
fi
log "phase 1 finished; results in $OUT_DIR (analysis.txt, analysis.json, prereg.md, SHA256SUMS)"
