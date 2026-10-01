#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
env_prefix="${QUIC_WC_ENV_PREFIX:-$project_root/.conda/openvla-gpu}"
clone_source="${QUIC_WC_CLONE_SOURCE:-/home/kony/miniconda3/envs/openvla}"
export PIP_CACHE_DIR="$project_root/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

if [[ ! -f "$env_prefix/conda-meta/history" ]]; then
    if [[ -e "$env_prefix" ]]; then
        echo "Incomplete/existing environment at $env_prefix; inspect it before retrying." >&2
        exit 1
    fi
    if [[ -f "$clone_source/conda-meta/history" ]]; then
        conda create --prefix "$env_prefix" --clone "$clone_source" --offline --copy --yes
    else
        echo "Working source environment missing: $clone_source" >&2
        echo "Set QUIC_WC_CLONE_SOURCE to a compatible environment; see the replay snapshot for rebuilding." >&2
        exit 1
    fi
fi

pip_options=()
if [[ "${QUIC_WC_OFFLINE:-0}" == 1 ]]; then
    pip_options=(--no-index --find-links "$project_root/.cache/wheelhouse")
fi
"$env_prefix/bin/python" -m pip install --no-deps --no-build-isolation \
    "${pip_options[@]}" \
    --editable "$project_root/external/openvla" \
    --editable "$project_root/external/libero"
# LIBERO has a namespace outer directory that its legacy find_packages() misses.
# Add this project's source root explicitly instead of inheriting an ovlab path.
"$env_prefix/bin/python" - "$project_root" <<'PY'
from pathlib import Path
import sys
import sysconfig
source = Path(sys.argv[1]) / "external/libero"
(Path(sysconfig.get_paths()["purelib"]) / "quic_wc_libero_source.pth").write_text(str(source) + "\n")
PY
"$env_prefix/bin/python" -m pip install \
    "${pip_options[@]}" \
    --constraint "$project_root/deploy/environments/openvla/constraints.txt" \
    --requirement "$project_root/deploy/environments/openvla/libero-requirements.txt"
"$env_prefix/bin/python" -m pip check
echo "Environment ready: $env_prefix"
echo "GPU checks: bash $project_root/deploy/scripts/test-openvla-env.sh"
