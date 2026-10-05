#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-1g-nvfp4
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PORT="${FLUX_ACTION_PORT:-8000}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid/variants/fp8r}"
NFE="${FLUX_ACTION_NFE:-}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
cd "$REPO"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
export TMPDIR="${FLUX_ACTION_TMPDIR:-$CACHE_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor/flux3-action-nvfp4-stats512}"
export QUACK_CACHE_DIR="${QUACK_CACHE_DIR:-$CACHE_ROOT/quack/flux3-action-nvfp4}"
export FLUX_ACTION_FP8_GEMM=nvfp4
export FLUX_ACTION_NVFP4_SCOPE="${FLUX_ACTION_NVFP4_SCOPE:-all}"
export FLUX_ACTION_FP8_QUANT=triton
export FLUX_ACTION_QK_EPILOGUE=triton
export FLUX_ACTION_ACTIVE_EPILOGUE=triton
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$QUACK_CACHE_DIR" logs

ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS%%,*}"
ENDPOINT_FILE="outputs/flux3-action-singlecard-nvfp4-${SLURM_JOB_ID}.endpoint"
printf 'ws://%s:%s\n' "$(hostname -f)" "$PORT" | tee "$ENDPOINT_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

SERVER_ARGS=(
  serve-robolab
  --checkpoint "$CHECKPOINT"
  --host 0.0.0.0
  --port "$PORT"
  --warmup 2
  --log-dir "outputs/robolab-singlecard-nvfp4/${SLURM_JOB_ID}"
)
if [[ -n "$NFE" ]]; then
  SERVER_ARGS+=(--setting "num_inference_steps=$NFE")
fi

exec "$REPO/.venv/bin/flux-action" "${SERVER_ARGS[@]}"
