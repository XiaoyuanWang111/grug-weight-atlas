#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 machine for the swap test: Python environment with CPU-only PyTorch, the harness
# installed in editable mode, a report of the CPU's bfloat16 instructions and of the memory, and one run of the
# unit tests.
#
# Usage: bash scripts/bootstrap.sh            (as root, or as a user who may use sudo)
#
# Settings (environment variables; defaults in brackets):
#   VENV_DIR        where the environment is created [$HOME/.venvs/swaptest]
#   TORCH_VERSION   CPU wheel of PyTorch to install [2.11.0]
#   TORCH_INDEX     package index of the CPU wheels [https://download.pytorch.org/whl/cpu]
#   SKIP_TESTS=1    do not run the unit tests
#
# This script never schedules or cancels a power-off; it only reports the one recorded in /run/poweroff_at.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/swaptest}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
SKIP_TESTS="${SKIP_TESTS:-0}"
PY="$VENV_DIR/bin/python"

say() { printf '\n== %s\n' "$*"; }

poweroff_note() {
    # The launcher writes the planned power-off time to /run/poweroff_at, as epoch seconds or as a date string.
    local file=/run/poweroff_at raw at now left
    if [[ -r "$file" ]]; then
        raw="$(head -n 1 "$file")"
        if [[ "$raw" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
            at="${raw//[[:space:]]/}"
        else
            at="$(date -u -d "$raw" +%s 2>/dev/null || true)"
        fi
        if [[ -n "$at" ]]; then
            now="$(date -u +%s)"
            left=$((at - now))
            if ((left > 0)); then
                printf 'power-off planned at %s, in %dh %02dm\n' "$(date -u -d "@$at" '+%Y-%m-%d %H:%M UTC')" \
                    $((left / 3600)) $(((left % 3600) / 60))
            else
                printf 'power-off time %s has passed\n' "$(date -u -d "@$at" '+%Y-%m-%d %H:%M UTC')"
            fi
        else
            printf 'power-off file %s holds "%s", which is not a time this script can read\n' "$file" "$raw"
        fi
    else
        echo "no /run/poweroff_at on this machine"
    fi
    if [[ -r /run/systemd/shutdown/scheduled ]]; then
        echo "systemd shutdown schedule: $(tr '\n' ' ' < /run/systemd/shutdown/scheduled)"
    fi
}

if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    echo "run this script as root or as a user who may use sudo" >&2
    exit 1
fi

say "power-off"
poweroff_note

say "system packages"
export DEBIAN_FRONTEND=noninteractive
"${SUDO[@]}" apt-get update -q
"${SUDO[@]}" apt-get install -y -q python3 python3-venv python3-pip ca-certificates curl util-linux

python3 - <<'EOF'
import sys
if sys.version_info < (3, 12):
    sys.exit(f"Python 3.12 or newer is needed, found {sys.version.split()[0]}")
print(f"system Python {sys.version.split()[0]}")
EOF

say "Python environment in $VENV_DIR"
if [[ ! -x "$PY" ]]; then
    python3 -m venv "$VENV_DIR"
fi
"$PY" -m pip install --quiet --upgrade pip
# PyTorch first, from the CPU-only index, so that no later requirement pulls a CUDA build from the default index.
"$PY" -m pip install --quiet --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
# The harness with its run and test extras (transformers pinned there, huggingface_hub with hf_xet, pyarrow,
# loguru for the reference-model cross-check, pytest, jinja2); safetensors and numpy are core requirements.
"$PY" -m pip install --quiet -e "$HARNESS_DIR[run,test]"
"$PY" - <<'EOF'
import importlib.metadata as md
import sys

import torch

if torch.version.cuda is not None:
    sys.exit(f"torch {torch.__version__} is a CUDA build; the CPU wheel is required")
pins = {"transformers", "huggingface_hub", "hf_xet", "safetensors", "numpy", "pyarrow", "jinja2", "pytest", "loguru"}
print(f"torch {torch.__version__} (CPU build), oneDNN available: {torch.backends.mkldnn.is_available()}, "
      f"CPU capability: {torch.backends.cpu.get_cpu_capability()}, default threads: {torch.get_num_threads()}")
for name in sorted(pins):
    try:
        print(f"  {name} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name}: not installed")
if not torch.backends.mkldnn.is_available():
    sys.exit("oneDNN is not available in this torch build")
EOF

say "CPU and memory"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)):' || true
flags="$(LC_ALL=C lscpu | grep -i '^flags:' || true)"
missing=0
for flag in avx512_bf16 amx_bf16 amx_tile; do
    if grep -qw "$flag" <<<"$flags"; then
        echo "  $flag: present"
    else
        echo "  $flag: MISSING"
        missing=1
    fi
done
if ((missing)); then
    echo "  warning: without these instructions bfloat16 matrix products take slow paths"
fi
awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2}
     END {printf "  memory: %.1f GiB total, %.1f GiB available (the 67B needs about 125 GiB plus the KV cache)\n",
          t / 1048576, a / 1048576}' /proc/meminfo
free -g
cores="$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)"
echo "  physical cores: $cores (run_phase1.sh uses this many torch threads unless THREADS is set)"

if [[ "$SKIP_TESTS" != 1 ]]; then
    say "unit tests"
    (cd "$HARNESS_DIR" && "$PY" -m pytest -q -p no:cacheprovider)
fi

say "power-off"
poweroff_note
echo
echo "bootstrap finished; next: bash $HARNESS_DIR/scripts/download.sh DATA_DIR small"
