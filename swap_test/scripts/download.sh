#!/usr/bin/env bash
# Download every input of phase 1, each pinned to an exact revision or checksum, and verify it.
#
# Usage: bash scripts/download.sh DATA_DIR [small|large|all]
#   small  the patch file, the GSM8K test split, the small stand-in checkpoint with its vLLM record, the source of
#          the reference model for the cross-checks, and the small files of the 67B checkpoint (configs,
#          tokenizer, chat template, shard index); about 1.6 GB
#   large  the 67B checkpoint's 502 weight shards (134,157,855,008 bytes)
#   all    small, then large (default)
#
# Layout under DATA_DIR:
#   merge-ta0.5/       open-athena/Snowball-67B-A2B-Merge-TA0.5 at MERGE_REV
#   d512-sft/          open-athena/Snowflake-d512-Datakit-SFT-262K-2026.10.22 at SMALL_REV
#   patch/             gates_full_strength.safetensors
#   gsm8k/             main/test-00000-of-00001.parquet and REVISION (the dataset commit used)
#   skyrl_src/         skyrl_train/... from Marin at MARIN_COMMIT
#   markers/           <item>.verified for every verified item; download_<stage>.running while a stage runs,
#                      download_<stage>.failed after a stage failed
#   .hf_home/          the Hugging Face cache and settings used here (no token is read or sent)
#
# Settings (environment variables; defaults in brackets):
#   VENV_DIR        environment made by bootstrap.sh [$HOME/.venvs/swaptest]
#   GSM8K_REV       commit of the openai/gsm8k dataset repository [see GSM8K_REV below]
#   VERIFY_SHA256   1 checks the SHA-256 of every LFS file against the listings, 0 checks sizes only [1]
#   MAX_WORKERS     parallel file downloads [8]
set -euo pipefail

DATA_DIR="${1:?usage: download.sh DATA_DIR [small|large|all]}"
STAGE="${2:-all}"
case "$STAGE" in
    small | large | all) ;;
    *)
        echo "usage: download.sh DATA_DIR [small|large|all]" >&2
        exit 2
        ;;
esac

VENV_DIR="${VENV_DIR:-$HOME/.venvs/swaptest}"
PY="$VENV_DIR/bin/python"
VERIFY_SHA256="${VERIFY_SHA256:-1}"
MAX_WORKERS="${MAX_WORKERS:-8}"

MERGE_REPO="open-athena/Snowball-67B-A2B-Merge-TA0.5"
MERGE_REV="cd0605de0ab76473b2262a9545596436e726b27d"
SMALL_REPO="open-athena/Snowflake-d512-Datakit-SFT-262K-2026.10.22"
SMALL_REV="cbc90512edcb5e38bad11e82d46bb69c3ef98afa"
PATCH_URL="https://github.com/XiaoyuanWang111/grug-weight-atlas/releases/download/v0.1.0/gates_full_strength.safetensors"
PATCH_SHA256="e8c7387ac84f01f3f1b1569397db07f99b9a4221ec78d411e86d84aa022b5946"
PATCH_BYTES=70831936
GSM8K_REPO="openai/gsm8k"
GSM8K_ROWS=1319
# ---------------------------------------------------------------------------------------------------------------
# GSM8K_REV: NOT PINNED IN ADVANCE. Set it to a commit of the openai/gsm8k dataset repository to pin the split.
# When it is empty, the commit that "main" points to at download time is used and recorded in gsm8k/REVISION;
# every later step reads the split from that file, and the pre-registration records the commit.
GSM8K_REV="${GSM8K_REV:-}"
# ---------------------------------------------------------------------------------------------------------------
MARIN_COMMIT="80004cc0654537cb39f386539f91cd28db653e4e"
SKYRL_BASE="https://raw.githubusercontent.com/marin-community/marin/$MARIN_COMMIT/vendor/skyrl/skyrl-train"
# SHA-256 and path of each reference-model file, read from the Marin repository at MARIN_COMMIT.
SKYRL_FILES="00ccc19d22efb86204e9069968f44e1cdbb914d61010f9672d83cb79d021458f  skyrl_train/models/grug_moe.py
04ea62be367db5ebbf447e192190c9234e58e611ce019233761ca607cfc78061  skyrl_train/models/ep_gradient.py
9075094ec8c552b43a3df7b0eb9774d92056064e33f291803a7807f0d4ebcb32  skyrl_train/models/grug_query_bias.py
46fee23275f7724be114d737f2bd13c515f0e24993c76d3ce126450ee2851149  skyrl_train/models/layers/moe_routing.py
b0d113ffcd3947016288de63f71d621570f2027268afef6420c3956899498358  skyrl_train/models/router_instrumentation.py
f4a2990186c2a44cb97c6ea3352938c9fb2e23f3d3bcfc132588dc7a2cbad3f2  skyrl_train/utils/flash_attention.py"

MARKERS="$DATA_DIR/markers"
mkdir -p "$MARKERS"
export HF_HOME="$DATA_DIR/.hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH

log() { printf '%s download[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STAGE" "$*"; }

finish() {
    local rc=$?
    rm -f "$MARKERS/download_$STAGE.running"
    if ((rc != 0)); then
        date -u +%Y-%m-%dT%H:%M:%SZ >"$MARKERS/download_$STAGE.failed"
        log "FAILED with status $rc; marker $MARKERS/download_$STAGE.failed written; fix the cause and rerun"
    fi
}
trap finish EXIT
rm -f "$MARKERS/download_$STAGE.failed"
if [[ "$STAGE" == all ]]; then
    rm -f "$MARKERS/download_small.failed" "$MARKERS/download_large.failed"
fi
echo "$$" >"$MARKERS/download_$STAGE.running"

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

fetch_patch() {
    local dest="$DATA_DIR/patch/gates_full_strength.safetensors"
    mkdir -p "$DATA_DIR/patch"
    if [[ ! -f "$dest" ]] || ! echo "$PATCH_SHA256  $dest" | sha256sum -c --status -; then
        log "patch: downloading $PATCH_URL"
        curl -fL --retry 5 --retry-delay 10 -o "$dest.part" "$PATCH_URL"
        mv "$dest.part" "$dest"
    fi
    if [[ "$(stat -c %s "$dest")" != "$PATCH_BYTES" ]] || ! echo "$PATCH_SHA256  $dest" | sha256sum -c --status -; then
        log "patch: SHA-256 or size of $dest differs from $PATCH_SHA256 ($PATCH_BYTES bytes)"
        exit 1
    fi
    log "patch: SHA-256 $PATCH_SHA256, $PATCH_BYTES bytes"
    mark patch
}

fetch_gsm8k() {
    local dir="$DATA_DIR/gsm8k" rev="$GSM8K_REV" files
    mkdir -p "$dir"
    if [[ -z "$rev" && -s "$dir/REVISION" ]]; then
        rev="$(cat "$dir/REVISION")"
        log "gsm8k: using the commit recorded in $dir/REVISION: $rev"
    fi
    if [[ -z "$rev" ]]; then
        rev="$("$PY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().dataset_info(sys.argv[1]).sha)' \
            "$GSM8K_REPO")"
        log "gsm8k: GSM8K_REV is not set; main of $GSM8K_REPO is commit $rev, which is used and recorded"
    fi
    if [[ -s "$dir/REVISION" && "$(cat "$dir/REVISION")" != "$rev" ]]; then
        log "gsm8k: $dir/REVISION holds $(cat "$dir/REVISION"), not $rev; remove $dir to change the commit"
        exit 1
    fi
    echo "$rev" >"$dir/REVISION"
    "${HF[@]}" "$GSM8K_REPO" --repo-type dataset --revision "$rev" --include "main/test-*.parquet" \
        --local-dir "$dir" --max-workers "$MAX_WORKERS"
    "$PY" -m swaptest.verify_download --repo "$GSM8K_REPO" --revision "$rev" --repo-type dataset --dir "$dir" \
        --include "main/test-*.parquet" "${SHA_FLAG[@]}"
    files=("$dir"/main/test-*.parquet)
    if ((${#files[@]} != 1)) || [[ ! -f "${files[0]}" ]]; then
        log "gsm8k: expected exactly one main/test-*.parquet file in $dir"
        exit 1
    fi
    "$PY" -c 'import sys, pyarrow.parquet as pq; n = pq.read_metadata(sys.argv[1]).num_rows; print(f"gsm8k: {n} rows in {sys.argv[1]}"); sys.exit(n != int(sys.argv[2]))' \
        "${files[0]}" "$GSM8K_ROWS"
    mark gsm8k
}

fetch_hf_model() {  # item repo revision dir [extra options for the download and the check]
    local item="$1" repo="$2" rev="$3" dir="$4"
    shift 4
    log "$item: downloading $repo at $rev into $dir"
    "${HF[@]}" "$repo" --revision "$rev" --local-dir "$dir" --max-workers "$MAX_WORKERS" "$@"
}

fetch_d512() {
    fetch_hf_model d512 "$SMALL_REPO" "$SMALL_REV" "$DATA_DIR/d512-sft"
    "$PY" -m swaptest.verify_download --repo "$SMALL_REPO" --revision "$SMALL_REV" --dir "$DATA_DIR/d512-sft" \
        "${SHA_FLAG[@]}"
    if [[ ! -s "$DATA_DIR/d512-sft/vllm_validation.json" ]]; then
        log "d512: vllm_validation.json is missing"
        exit 1
    fi
    mark d512
}

fetch_skyrl() {
    local dir="$DATA_DIR/skyrl_src" path
    mkdir -p "$dir"
    while read -r _ path; do
        mkdir -p "$dir/$(dirname "$path")"
        curl -fsSL --retry 5 --retry-delay 10 -o "$dir/$path" "$SKYRL_BASE/$path"
    done <<<"$SKYRL_FILES"
    printf '%s\n' "$SKYRL_FILES" >"$dir/SHA256SUMS"
    (cd "$dir" && sha256sum -c SHA256SUMS)
    mark skyrl
}

fetch_merge_meta() {
    fetch_hf_model merge_meta "$MERGE_REPO" "$MERGE_REV" "$DATA_DIR/merge-ta0.5" --exclude "*.safetensors"
    "$PY" -m swaptest.verify_download --repo "$MERGE_REPO" --revision "$MERGE_REV" --dir "$DATA_DIR/merge-ta0.5" \
        --exclude "*.safetensors" "${SHA_FLAG[@]}"
    mark merge_meta
}

fetch_merge() {
    fetch_hf_model merge "$MERGE_REPO" "$MERGE_REV" "$DATA_DIR/merge-ta0.5"
    log "merge: checking 512 files against the saved and the Hub listing (SHA-256: $VERIFY_SHA256)"
    "$PY" -m swaptest.verify_download --repo "$MERGE_REPO" --revision "$MERGE_REV" --dir "$DATA_DIR/merge-ta0.5" \
        --workers "$MAX_WORKERS" "${SHA_FLAG[@]}"
    mark merge
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
    run_item patch
    run_item gsm8k
    run_item skyrl
    run_item merge_meta
    run_item d512
fi
if [[ "$STAGE" == large || "$STAGE" == all ]]; then
    run_item merge
fi
df -h "$DATA_DIR" | tail -n 1
log "done: $(cd "$MARKERS" && ls -- *.verified 2>/dev/null | tr '\n' ' ')"
