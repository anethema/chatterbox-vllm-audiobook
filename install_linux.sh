#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_dir}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer supports Linux and WSL2 only." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "An NVIDIA GPU and working NVIDIA driver are required." >&2
    echo "Install or repair the driver, then confirm that nvidia-smi succeeds." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1 || \
   ! command -v ffmpeg >/dev/null 2>&1 || \
   ! command -v ffprobe >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Install curl, FFmpeg, and FFprobe with your distribution's package manager." >&2
        exit 1
    fi

    apt_command=(apt-get)
    if [[ "${EUID}" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required to install curl and FFmpeg." >&2
            exit 1
        fi
        apt_command=(sudo apt-get)
    fi

    "${apt_command[@]}" update
    "${apt_command[@]}" install -y curl ffmpeg
fi

uv_bin="$(command -v uv || true)"
if [[ -z "${uv_bin}" && -x "${HOME}/.local/bin/uv" ]]; then
    uv_bin="${HOME}/.local/bin/uv"
fi

if [[ -z "${uv_bin}" ]]; then
    echo "Installing uv with the official Astral installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv_bin="${HOME}/.local/bin/uv"
fi

if [[ ! -x "${uv_bin}" ]]; then
    echo "uv was not found after installation." >&2
    echo "Restart the shell and rerun this installer." >&2
    exit 1
fi

echo "Creating the Python environment and installing dependencies..."
"${uv_bin}" sync --locked --python 3.12

echo "Verifying CUDA, PyTorch, vLLM, and FFmpeg..."
"${project_dir}/.venv/bin/python" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available():
    raise SystemExit("PyTorch was installed, but CUDA is not available")

print(f"PyTorch {torch.__version__}")
print(f"PyTorch CUDA runtime {torch.version.cuda}")
print(f"vLLM {vllm.__version__}")
print(f"GPU {torch.cuda.get_device_name(0)}")
PY

ffmpeg -version | sed -n '1p'
ffprobe -version | sed -n '1p'

echo
echo "Installation complete. Start the interface with:"
echo "  ${project_dir}/run_chatterbox_vllm.sh"
