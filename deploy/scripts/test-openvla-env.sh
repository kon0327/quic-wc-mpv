#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
env_prefix="${QUIC_WC_ENV_PREFIX:-$project_root/.conda/openvla-gpu}"
export LIBERO_CONFIG_PATH="$project_root/.cache/libero"
export XDG_CACHE_HOME="$project_root/.cache"
export HF_HOME="$project_root/.cache/huggingface"
export NUMBA_CACHE_DIR="$project_root/.cache/numba"
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$LIBERO_CONFIG_PATH" "$HF_HOME" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd "$project_root"
"$env_prefix/bin/python" -m pip check
"$env_prefix/bin/python" deploy/scripts/check_openvla_env.py "$@"
