#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-late-prune-smoke
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
RUN_DIR="${FLUX_ACTION_RUN_DIR:-$REPO/outputs/late-video-prune-smoke/${SLURM_JOB_ID}}"
SOURCE_ROOT="${FLUX_ACTION_SOURCE_ROOT:-outputs/public-droid/source}"
INDEX_DIR="${FLUX_ACTION_INDEX_DIR:-outputs/public-droid/index}"
STEPS="${FLUX_ACTION_STEPS:-2}"
cd "$REPO"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
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

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

"$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=4 -m flux_action.cli train \
  --config configs/droid/train_late_video_prune.json \
  --override source_root="$SOURCE_ROOT" \
  --override index_dir="$INDEX_DIR" \
  --override output_dir="$RUN_DIR" \
  --override steps="$STEPS" \
  --override cooldown_start=null \
  --override windows_per_rank=1 \
  --override grad_accumulation=1 \
  --override num_workers=1 \
  --override in_process_loader=true \
  --override decoder=pyav \
  --override checkpoint_every=2 \
  --override log_every=1 \
  --override keep_checkpoints=1 \
  --override keep_every=null \
  --override resume=none

echo "FLUX3_ACTION_LATE_PRUNE_DISTILL_SMOKE=$RUN_DIR"
