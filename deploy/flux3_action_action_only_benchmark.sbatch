#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-vla-bench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid/variants/sd}"
OBSERVATION="${FLUX_ACTION_OBSERVATION:-outputs/public-droid/observation.npz}"
OBSERVATION_META="${FLUX_ACTION_OBSERVATION_META:-outputs/public-droid/observation.json}"
WARMUP="${FLUX_ACTION_WARMUP:-10}"
ITERATIONS="${FLUX_ACTION_ITERATIONS:-30}"
PORT="${FLUX_ACTION_PORT:-$((18000 + SLURM_JOB_ID % 1000))}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
RUN_DIR="$REPO/outputs/action-only-benchmark/${SLURM_JOB_ID}"
cd "$REPO"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
export TMPDIR="${FLUX_ACTION_TMPDIR:-$CACHE_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
CODE_STORAGE_ROOT="$(readlink -f "$HOME/code")"
export HF_HOME="${HF_HOME:-$CODE_STORAGE_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$RUN_DIR" logs

# The node allocation may expose four GB200 GPUs, but this benchmark deliberately uses exactly one.
ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS%%,*}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$REPO/.venv/bin/flux-action" serve-robolab \
  --checkpoint "$CHECKPOINT" \
  --setting predict_future_video=false \
  --no-compile-dit \
  --warmup 3 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --log-dir "$RUN_DIR/requests" \
  >"$RUN_DIR/server.out" 2>"$RUN_DIR/server.err" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 120); do
  if "$REPO/.venv/bin/python" -c \
    'import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(("127.0.0.1", int(sys.argv[1]))))' \
    "$PORT"; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"
  fi
  sleep 5
done
test "$READY" -eq 1

TASK_CAPTION="$($REPO/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$OBSERVATION_META")"
"$REPO/.venv/bin/python" deploy/benchmark_service.py \
  --endpoint "ws://127.0.0.1:$PORT" \
  --observation "$OBSERVATION" \
  --prompt "$TASK_CAPTION" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS" \
  --output "$RUN_DIR/result.json"

kill "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT
echo "FLUX3_ACTION_ACTION_ONLY_BENCHMARK=$RUN_DIR/result.json"
