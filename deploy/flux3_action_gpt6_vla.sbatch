#!/usr/bin/env bash
#SBATCH --job-name=gpt6-flux-action
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLA_PORT="${FLUX_ACTION_VLA_PORT:-8000}"
PIPELINE_PORT="${GPT6_VLA_PORT:-8001}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid/variants/fp8r}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
CODEX_BIN="${CODEX_BIN:-/home/yitongl/.local/bin/codex}"
cd "$REPO"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
export TMPDIR="${FLUX_ACTION_TMPDIR:-$CACHE_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor/gpt6-flux-action}"
export QUACK_CACHE_DIR="${QUACK_CACHE_DIR:-$CACHE_ROOT/quack/gpt6-flux-action}"
export FLUX_ACTION_FP8_GEMM=quack
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$QUACK_CACHE_DIR" logs outputs
ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS%%,*}"

VLA_LOG="logs/gpt6-vla-upstream-${SLURM_JOB_ID}.log"
PIPELINE_LOG="outputs/gpt6-vla-${SLURM_JOB_ID}.jsonl"

cleanup() {
  if [[ -n "${VLA_PID:-}" ]]; then
    kill "$VLA_PID" 2>/dev/null || true
    wait "$VLA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$REPO/.venv/bin/flux-action" serve-robolab \
  --checkpoint "$CHECKPOINT" \
  --setting num_inference_steps=3 \
  --host 127.0.0.1 \
  --port "$VLA_PORT" \
  --warmup 2 \
  --log-dir "outputs/gpt6-vla-upstream/${SLURM_JOB_ID}" \
  >"$VLA_LOG" 2>&1 &
VLA_PID=$!

"$REPO/.venv/bin/python" - "$VLA_PORT" "$VLA_PID" "$VLA_LOG" <<'PY'
import os
import socket
import sys
import time

port, pid, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + 2700
while time.monotonic() < deadline:
    if not os.path.exists(f"/proc/{pid}"):
        raise SystemExit(f"VLA exited before readiness; see {log}")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"timed out waiting for VLA; see {log}")
PY

printf 'ws://%s:%s\n' "$(hostname -f)" "$PIPELINE_PORT" \
  | tee "outputs/gpt6-vla-${SLURM_JOB_ID}.endpoint"
exec "$REPO/.venv/bin/python" deploy/gpt6_vla_proxy.py \
  --planner-backend codex \
  --upstream "ws://127.0.0.1:${VLA_PORT}" \
  --host 0.0.0.0 \
  --port "$PIPELINE_PORT" \
  --codex-bin "$CODEX_BIN" \
  --codex-work-root "$TMPDIR" \
  --model "${CODEX_PLANNER_MODEL:-gpt-6-astra}" \
  --reasoning-effort "${GPT6_REASONING_EFFORT:-low}" \
  --image-detail "${GPT6_IMAGE_DETAIL:-high}" \
  --log-path "$PIPELINE_LOG"
