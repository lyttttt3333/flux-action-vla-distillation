#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-tp2cfg
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PORT="${FLUX_ACTION_PORT:-8000}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
cd "$REPO"
if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
export TMPDIR="${FLUX_ACTION_TMPDIR:-$CACHE_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR"

ENDPOINT_FILE="outputs/flux3-action-service-${SLURM_JOB_ID}.endpoint"
printf 'ws://%s:%s\n' "$(hostname -f)" "$PORT" | tee "$ENDPOINT_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

exec .venv/bin/torchrun --standalone --nproc-per-node=4 deploy/distributed_service.py \
  --checkpoint "$CHECKPOINT" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --warmup 3
