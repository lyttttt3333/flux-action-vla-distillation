#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-vla-distill
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
RUN_DIR="${FLUX_ACTION_RUN_DIR:-$REPO/outputs/droid/action-only-sd-distill-seed42}"
STEPS="${FLUX_ACTION_STEPS:-5000}"
cd "$REPO"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
# Python's forkserver appends a long random socket name.  The repository's absolute Lustre path is
# already close to AF_UNIX's limit, so keep socket-only temp state under a short node-local path.
# Model/data/package caches and checkpoints remain on Lustre.
export TMPDIR="${FLUX_ACTION_SOCKET_TMPDIR:-/tmp/fa-${SLURM_JOB_ID}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
CODE_STORAGE_ROOT="$(readlink -f "$HOME/code")"
export HF_HOME="${HF_HOME:-$CODE_STORAGE_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLUX_ACTION_PG_TIMEOUT_MINUTES=30
export PATH="$REPO/.tools/ffmpeg/bin:$PATH"
mkdir -p "$TMPDIR" "$RUN_DIR" "$CACHE_ROOT" logs

test -x "$REPO/.tools/ffmpeg/bin/ffmpeg"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

"$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=4 -m flux_action.cli train \
  --config configs/droid/train_action_only.json \
  --override output_dir="$RUN_DIR" \
  --override steps="$STEPS" \
  --override windows_per_rank=8 \
  --override grad_accumulation=2 \
  --override num_workers=8 \
  --override prefetch_factor=2 \
  --override shard_size=4 \
  --override checkpoint_every=500 \
  --override keep_checkpoints=2 \
  --override keep_every=1000 \
  --override log_every=10 \
  --override resume=auto

echo "FLUX3_ACTION_ACTION_ONLY_DISTILL_TRAIN=$RUN_DIR"
