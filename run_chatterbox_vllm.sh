#!/usr/bin/env bash
set -euo pipefail

cd /home/anethema/chatterbox-vllm-standalone
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export CHATTERBOX_CFG_SCALE="${CHATTERBOX_CFG_SCALE:-0.5}"

exec .venv/bin/python gradio_tts_app.py
