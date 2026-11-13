#!/usr/bin/env bash
# Download every input that phase 2 adds to phase 1's, each pinned to an exact revision, and verify it.
#
# Usage: bash scripts/download_phase2.sh DATA_DIR [small|large|all]
#   small  the CRUXEval test file and the MMLU-Pro test and validation splits (about 10 MB), with a row-count check
#   large  the base checkpoint B and the two experts E and E2: 79 + 39 + 39 weight shards, 402,473,493,040 bytes
#   all    small, then large (default)
# Phase 1's download.sh supplies Merge-TA0.5 (weights, tokenizer, chat template) and the GSM8K test split.
#
# Layout added under DATA_DIR:
#   b/  e/  e2/        B, E and E2 at B_REV, E_REV, E2_REV
#   cruxeval/          test.jsonl and REVISION
#   mmlu_pro/          data/test-00000-of-00001.parquet, data/validation-00000-of-00001.parquet and REVISION
#   markers/           <item>.verified for every verified item; download2_<stage>.running while a stage runs,
#                      download2_<stage>.failed after a stage failed (phase 1's markers keep the prefix download_)
#
# Settings (environment variables; defaults in brackets):
#   VENV_DIR        environment made by bootstrap.sh [$HOME/.venvs/swaptest]
#   VERIFY_SHA256   1 checks the SHA-256 of every LFS file against the listings, 0 checks sizes only [1]
#   MAX_WORKERS     parallel file downloads [8]
#   CRUXEVAL_REV    commit of cruxeval-org/cruxeval [see below]
#   MMLU_PRO_REV    commit of TIGER-Lab/MMLU-Pro [see below]
#   MIN_FREE_GB     free space the large stage needs before it starts [410]
set -euo pipefail

DATA_DIR="${1:?usage: download_phase2.sh DATA_DIR [small|large|all]}"
STAGE="${2:-all}"
case "$STAGE" in
    small | large | all) ;;
    *)
        echo "usage: download_phase2.sh DATA_DIR [small|large|all]" >&2
        exit 2
        ;;
esac

VENV_DIR="${VENV_DIR:-$HOME/.venvs/swaptest}"
PY="${PY:-$VENV_DIR/bin/python}"
VERIFY_SHA256="${VERIFY_SHA256:-1}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MIN_FREE_GB="${MIN_FREE_GB:-410}"

# The checkpoints at the commits read while the experiment was planned; their safetensors files have the same
# SHA-256 as at the commits marin#9984 pins (fdc3c3c5, fb4f8051, acb97183).
B_REPO="open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"
B_REV="80975627aebeaec3ef946ad609d9c1d95458ebd8"
E_REPO="open-athena/Snowball-67B-A2B-CruxEval-Expert-E-Step272"
E_REV="2eb988bb5df70378950d983ac8e70dae550cf6d5"
E2_REPO="open-athena/Snowball-67B-A2B-MMLUPro-Expert-E2-Step254"
E2_REV="0f22c03aff076f46c292196cf0d03a9bc4ab81c5"
# Dataset revisions: the commits that Marin's canonical CRUXEval and MMLU-Pro tasks pin at marin 80004cc06. The
# MMLU-Pro commit is also the one marin#9984 re-joined its labels to. The CRUXEval commit that #9984's evaluator
# read is not recorded; the test file is not known to have changed.
CRUXEVAL_REPO="cruxeval-org/cruxeval"
CRUXEVAL_REV="${CRUXEVAL_REV:-b96af0450242eb4da433032b90998f25588a5d0f}"
MMLU_PRO_REPO="TIGER-Lab/MMLU-Pro"
MMLU_PRO_REV="${MMLU_PRO_REV:-b189ec765aa7ed75c8acfea42df31fdae71f97be}"

MARKERS="$DATA_DIR/markers"
mkdir -p "$MARKERS"
export HF_HOME="$DATA_DIR/.hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH

log() { printf '%s download2[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STAGE" "$*"; }

finish() {
    local rc=$?
    rm -f "$MARKERS/download2_$STAGE.running"
    if ((rc != 0)); then
        date -u +%Y-%m-%dT%H:%M:%SZ >"$MARKERS/download2_$STAGE.failed"
        log "FAILED with status $rc; marker $MARKERS/download2_$STAGE.failed written; fix the cause and rerun"
    fi
}
trap finish EXIT
rm -f "$MARKERS/download2_$STAGE.failed"
if [[ "$STAGE" == all ]]; then
    rm -f "$MARKERS/download2_small.failed" "$MARKERS/download2_large.failed"
fi
echo "$$" >"$MARKERS/download2_$STAGE.running"

if [[ ! -x "$PY" ]]; then
    log "no Python environment at $VENV_DIR; run bootstrap.sh first"
    exit 2
fi
if [[ -x "$VENV_DIR/bin/hf" ]]; then
    HF=("$VENV_DIR/bin/hf" download)
elif [[ -x "$VENV_DIR/bin/huggingface-cli" ]]; then
    HF=("$VENV_DIR/bin/huggingface-cli" download)
else
    log "neither hf nor huggingface-cli is installed in $VENV_DIR"
    exit 2
fi
SHA_FLAG=()
if [[ "$VERIFY_SHA256" == 1 ]]; then
    SHA_FLAG=(--sha256)
fi

verified() { [[ -f "$MARKERS/$1.verified" ]]; }
mark() {
    date -u +%Y-%m-%dT%H:%M:%SZ >"$MARKERS/$1.verified"
    log "$1 verified"
}

record_revision() {  # DIR REV: write DIR/REVISION, refusing to mix two commits in one directory
    local dir="$1" rev="$2"
    mkdir -p "$dir"
    if [[ -s "$dir/REVISION" && "$(cat "$dir/REVISION")" != "$rev" ]]; then
        log "$dir/REVISION holds $(cat "$dir/REVISION"), not $rev; remove $dir to change the commit"
        exit 1
    fi
    echo "$rev" >"$dir/REVISION"
}

fetch_cruxeval() {
    local dir="$DATA_DIR/cruxeval"
    record_revision "$dir" "$CRUXEVAL_REV"
    log "cruxeval: downloading $CRUXEVAL_REPO at $CRUXEVAL_REV"
    "${HF[@]}" "$CRUXEVAL_REPO" --repo-type dataset --revision "$CRUXEVAL_REV" --include "test.jsonl" \
        --local-dir "$dir" --max-workers "$MAX_WORKERS"
    "$PY" -m swaptest.verify_download --repo "$CRUXEVAL_REPO" --revision "$CRUXEVAL_REV" --repo-type dataset \
        --dir "$dir" --include "test.jsonl" "${SHA_FLAG[@]}"
    mark cruxeval
}

fetch_mmlu_pro() {
    local dir="$DATA_DIR/mmlu_pro" pattern
    record_revision "$dir" "$MMLU_PRO_REV"
    log "mmlu_pro: downloading $MMLU_PRO_REPO at $MMLU_PRO_REV"
    for pattern in "data/test-*.parquet" "data/validation-*.parquet"; do
        "${HF[@]}" "$MMLU_PRO_REPO" --repo-type dataset --revision "$MMLU_PRO_REV" --include "$pattern" \
            --local-dir "$dir" --max-workers "$MAX_WORKERS"
    done
    "$PY" -m swaptest.verify_download --repo "$MMLU_PRO_REPO" --revision "$MMLU_PRO_REV" --repo-type dataset \
        --dir "$dir" --include "data/test-*.parquet" "data/validation-*.parquet" "${SHA_FLAG[@]}"
    mark mmlu_pro
}

fetch_task_data() {  # the published row counts and fields of the three files
    "$PY" -m swaptest.taskprompts --check-data --cruxeval "$DATA_DIR/cruxeval/test.jsonl" \
        --mmlu-pro-test "$DATA_DIR/mmlu_pro/data/test-00000-of-00001.parquet" \
        --mmlu-pro-validation "$DATA_DIR/mmlu_pro/data/validation-00000-of-00001.parquet"
    mark task_data
}

fetch_model() {  # ITEM REPO REV: download a checkpoint and check it against the saved and the Hub listing
    local item="$1" repo="$2" rev="$3" dir="$DATA_DIR/$1"
    log "$item: downloading $repo at $rev into $dir"
    "${HF[@]}" "$repo" --revision "$rev" --local-dir "$dir" --max-workers "$MAX_WORKERS"
    log "$item: checking every file against the saved and the Hub listing (SHA-256: $VERIFY_SHA256)"
    "$PY" -m swaptest.verify_download --repo "$repo" --revision "$rev" --dir "$dir" --workers "$MAX_WORKERS" \
        "${SHA_FLAG[@]}"
    mark "$item"
}
fetch_b() { fetch_model b "$B_REPO" "$B_REV"; }
fetch_e() { fetch_model e "$E_REPO" "$E_REV"; }
fetch_e2() { fetch_model e2 "$E2_REPO" "$E2_REV"; }

check_space() {
    local avail
    if verified b && verified e && verified e2; then
        return 0
    fi
    avail="$(df -P -B 1000000000 "$DATA_DIR" | awk 'NR == 2 {print $4}')"
    if ((avail < MIN_FREE_GB)); then
        log "only $avail GB free under $DATA_DIR; the three checkpoints need about 403 GB (MIN_FREE_GB=$MIN_FREE_GB)"
        exit 1
    fi
    log "$avail GB free under $DATA_DIR"
}

run_item() {  # run fetch_<item> unless the item is verified already
    # Called as a plain command (never inside "if", "||" or "&&"), so "set -e" stays in force inside fetch_<item>.
    if verified "$1"; then
        log "$1 verified earlier; skipped"
        return 0
    fi
    "fetch_$1"
}

log "start; data directory $DATA_DIR"
if [[ "$STAGE" == small || "$STAGE" == all ]]; then
    run_item cruxeval
    run_item mmlu_pro
    run_item task_data
fi
if [[ "$STAGE" == large || "$STAGE" == all ]]; then
    check_space
    run_item b
    run_item e
    run_item e2
fi
df -h "$DATA_DIR" | tail -n 1
log "done: $(cd "$MARKERS" && ls -- *.verified 2>/dev/null | tr '\n' ' ')"
