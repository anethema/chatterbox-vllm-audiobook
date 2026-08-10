#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${project_dir}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python environment not found at ${project_dir}/.venv" >&2
    echo "Run ${project_dir}/install_linux.sh first." >&2
    exit 1
fi

cd "${project_dir}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export CHATTERBOX_CFG_SCALE="${CHATTERBOX_CFG_SCALE:-0.5}"

exec "${python_bin}" "${project_dir}/gradio_tts_app.py"
