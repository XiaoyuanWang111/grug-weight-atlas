#!/bin/bash
# Phase 2b on the machine: steps 22 to 24 of run_phase2.sh (capability runs of arm A and of the candidate G, then
# the capability analysis), issued from the published tree. run_phase2.sh itself refuses a tree other than the one
# in the filled pre-registration, and that tree holds the eligibility condition that deviation 3 corrects.
# The commands and settings are those of run_phase2.sh; the checks below stop the driver when the generating
# modules, the task prompts or the merge spec differ from the pre-registered ones.
# Resumable: every finished step writes /data/phase2/markers_2b/<step>.done. Log: /data/phase2/logs/run_phase2b.log
# Usage: nohup setsid bash a1_p2b_run.sh > /data/p2b.out 2>&1 &
set -uo pipefail

PUB="$HOME/harness_pub/swap_test"      # published tree (commit in $HOME/harness_pub/COMMIT)
OLD="$HOME/harness_p2"                 # tree of the filled pre-registration of phase 2
DATA_DIR=/data
OUT_DIR="$DATA_DIR/phase2"
PHASE1_DIR="$DATA_DIR/phase1"
PY="$HOME/.venvs/swaptest2/bin/python"
MERGE_DIR="$DATA_DIR/merge-ta0.5"
SPEC="$OUT_DIR/merge_spec.json"
TASK_PROMPTS="$OUT_DIR/task_prompts.jsonl"
GSM8K_PROMPTS="$PHASE1_DIR/prompts.jsonl"
PHASE1_REPLIES="$PHASE1_DIR/main/replies.jsonl"
RUNAWAY_DIR="$OUT_DIR/runaway"
TF_FILE="$OUT_DIR/teacher_forced.jsonl"
TASKS_A_DIR="$OUT_DIR/tasks_A"
TASKS_MERGE_DIR="$OUT_DIR/tasks_merge"
SCORED=G
SEED=9984
TASK_CAP=4096
TASK_BATCH_SIZE=384
PREFILL_CHUNK=256
PREFILL_ROWS=16
LEVEL=0.95
BOOT_DRAWS=10000
GRADE_WORKERS=8
THREADS="$( (lscpu -p=CORE,SOCKET 2>/dev/null || true) | { grep -v '^#' || true; } | sort -u | wc -l)"
PUB_REANALYSIS_SHA=dd2475fc97876e1fa0489a8fd35ad039b77f97343525bbfb65cfd2fa08036570
MARK="$OUT_DIR/markers_2b"
LOG="$OUT_DIR/logs/run_phase2b.log"
mkdir -p "$MARK" "$OUT_DIR/logs"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
stop() { local code="$1"; shift; log "STOP (exit $code): $*"; exit "$code"; }
done_step() { [[ -f "$MARK/$1.done" ]]; }
mark() { date -u +%Y-%m-%dT%H:%M:%SZ >"$MARK/$1.done"; log "$1 done"; }
run_logged() {
    local logfile="$1"; shift
    log "running: $* (output in ${logfile#"$OUT_DIR"/})"
    printf '\n### %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$logfile"
    "$@" >>"$logfile" 2>&1
}

cd "$PUB" || stop 2 "no published tree at $PUB"
export PYTHONPATH="$PUB"
log "run_phase2b starting; tree $PUB at commit $(cat "$HOME/harness_pub/COMMIT"), threads $THREADS, scored arm $SCORED"

# --- checks before any capability reply is sampled
where="$("$PY" -c 'import swaptest, os; print(os.path.dirname(os.path.abspath(swaptest.__file__)))')"
[[ "$where" == "$PUB/swaptest" ]] || stop 2 "python imports swaptest from $where, not from the published tree"
differ="$(diff -rq "$OLD/swaptest" "$PUB/swaptest" -x __pycache__ | sed 's/^/  /')"
log "module files that differ between the pre-registered tree and the published tree:"
printf '%s\n' "$differ" | tee -a "$LOG"
if [[ "$(printf '%s\n' "$differ" | grep -c .)" -ne 1 || "$differ" != *"analyze_phase2.py differ"* ]]; then
    stop 2 "the trees differ in more than analyze_phase2.py"
fi
[[ "$(diff "$OLD/swaptest/analyze_phase2.py" "$PUB/swaptest/analyze_phase2.py" | grep -c '^[<>]')" -eq 2 ]] \
    || stop 2 "analyze_phase2.py differs in more than the one eligibility line"
tree_sha="$("$PY" -m swaptest.hashes --prompts "$TASK_PROMPTS" --harness "$PUB" | awk '$1 == "harness_tree_sha256" {print $2}')"
log "published tree SHA-256 $tree_sha"
have="$(printf 'task_prompts_sha256 %s\nmerge_spec_sha256 %s' "$(sha256sum "$TASK_PROMPTS" | cut -d ' ' -f 1)" "$(sha256sum "$SPEC" | cut -d ' ' -f 1)")"
want="$(grep -E '^(task_prompts|merge_spec)_sha256 ' "$OUT_DIR/prereg_hashes.txt")"
[[ "$have" == "$want" ]] || stop 2 "the task prompts or the merge spec differ from the pre-registration"
(cd "$OUT_DIR" && sha256sum -c prereg_phase2.md.sha256 >/dev/null) || stop 2 "prereg_phase2.md changed"
log "task prompts, merge spec and filled pre-registration are unchanged"

if ! done_step reanalysis; then
    "$PY" -m swaptest.analyze_phase2 runaway --prompts "$GSM8K_PROMPTS" --replies "$PHASE1_REPLIES" \
        "$RUNAWAY_DIR/replies.jsonl" --teacher-forced "$TF_FILE" --json "$OUT_DIR/runaway_reanalysis.json" \
        --decisions "$OUT_DIR/runaway_decisions_reanalysis.txt" >"$OUT_DIR/runaway_reanalysis.txt" 2>&1 \
        || stop 38 "the runaway reanalysis failed; see runaway_reanalysis.txt"
    mark reanalysis
fi
got="$(sha256sum "$OUT_DIR/runaway_reanalysis.json" | cut -d ' ' -f 1)"
[[ "$got" == "$PUB_REANALYSIS_SHA" ]] || stop 2 "runaway_reanalysis.json is $got, the published file is $PUB_REANALYSIS_SHA"
task_arms="$(awk '$1 == "task_arms" {print $2}' "$OUT_DIR/runaway_decisions_reanalysis.txt")"
[[ "$task_arms" == "$SCORED" ]] || stop 2 "the reanalysis names task arms $task_arms, the driver expects $SCORED"
log "runaway reanalysis equals the published file; task arms $task_arms"
[[ -s "$OUT_DIR/deviation4.md" ]] || stop 2 "deviation4.md is missing: write it before the capability runs"
(cd "$OUT_DIR" && sha256sum deviation4.md >deviation4.md.sha256)
log "deviation4.md SHA-256 $(cut -d ' ' -f 1 "$OUT_DIR/deviation4.md.sha256")"

COMMON=(--threads "$THREADS")
RUN_OPTS=(--seed "$SEED" --prefill-chunk "$PREFILL_CHUNK" --prefill-rows "$PREFILL_ROWS" "${COMMON[@]}")
TASK_BATCH=$((TASK_BATCH_SIZE / ${#SCORED} * ${#SCORED}))

if ! done_step step22; then
    log "step 22: capability run of arm A on $(wc -l <"$TASK_PROMPTS") task prompts, cap $TASK_CAP (progress in tasks_A/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/step22_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --prompts "$TASK_PROMPTS" \
        --out "$TASKS_A_DIR" --arms A --cap "$TASK_CAP" --batch-size "$TASK_BATCH_SIZE" "${RUN_OPTS[@]}" || rc=$?
    ((rc == 0)) || stop 41 "step 22: the capability run of arm A ended with status $rc; rerun this script to resume it"
    mark step22
fi

if ! done_step step23; then
    log "step 23: capability run of arms $SCORED on the task prompts, cap $TASK_CAP, batch size $TASK_BATCH (progress in tasks_merge/run.log)"
    rc=0
    run_logged "$OUT_DIR/logs/step23_run.log" "$PY" -m swaptest.run --checkpoint "$MERGE_DIR" --merge "$SPEC" \
        --prompts "$TASK_PROMPTS" --out "$TASKS_MERGE_DIR" --arms "$SCORED" --cap "$TASK_CAP" --batch-size "$TASK_BATCH" \
        "${RUN_OPTS[@]}" || rc=$?
    ((rc == 0)) || stop 41 "step 23: the capability run of arms $SCORED ended with status $rc; rerun this script to resume it"
    mark step23
fi

if ! done_step step24; then
    log "step 24: capability analysis"
    rc=0
    "$PY" -m swaptest.analyze_phase2 capability --prompts "$TASK_PROMPTS" --replies "$TASKS_A_DIR/replies.jsonl" \
        "$TASKS_MERGE_DIR/replies.jsonl" --runaway "$OUT_DIR/runaway_reanalysis.json" --grades "$OUT_DIR/grades.jsonl" \
        --json "$OUT_DIR/capability.json" --decisions "$OUT_DIR/capability_decisions.txt" --level "$LEVEL" \
        --draws "$BOOT_DRAWS" --seed "$SEED" --workers "$GRADE_WORKERS" >"$OUT_DIR/capability.txt" 2>&1 || rc=$?
    ((rc == 0)) || stop 42 "step 24: the capability analysis ended with status $rc; see capability.txt"
    (cd "$OUT_DIR" && find . -type f ! -name SHA256SUMS ! -path './logs/run_phase2*.log' ! -path './markers/*' \
        ! -path './markers_2b/*' ! -name '*.tmp' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
    mark step24
fi

tee -a "$LOG" <"$OUT_DIR/capability.txt"
log "capability decisions: $(tr '\n' ' ' <"$OUT_DIR/capability_decisions.txt")"
log "phase 2b finished"
