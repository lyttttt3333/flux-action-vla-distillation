#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-vla-distill-smoke
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
RUN_DIR="$REPO/outputs/action-only-distill-smoke/${SLURM_JOB_ID}"
SOURCE_ROOT="${FLUX_ACTION_SOURCE_ROOT:-outputs/public-droid/source}"
INDEX_DIR="${FLUX_ACTION_INDEX_DIR:-outputs/public-droid/index}"
STEPS="${FLUX_ACTION_STEPS:-2}"
WINDOWS_PER_RANK="${FLUX_ACTION_WINDOWS_PER_RANK:-1}"
GRAD_ACCUMULATION="${FLUX_ACTION_GRAD_ACCUMULATION:-1}"
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
export FLUX_ACTION_PG_TIMEOUT_MINUTES=30
mkdir -p "$TMPDIR" "$RUN_DIR" logs

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

"$REPO/.venv/bin/torchrun" --standalone --nproc_per_node=4 -m flux_action.cli train \
  --config configs/droid/train_action_only.json \
  --override source_root="$SOURCE_ROOT" \
  --override index_dir="$INDEX_DIR" \
  --override output_dir="$RUN_DIR" \
  --override steps="$STEPS" \
  --override cooldown_start=null \
  --override windows_per_rank="$WINDOWS_PER_RANK" \
  --override grad_accumulation="$GRAD_ACCUMULATION" \
  --override num_workers=1 \
  --override in_process_loader=true \
  --override decoder=pyav \
  --override shard_size=4 \
  --override checkpoint_every=2 \
  --override log_every=1 \
  --override keep_checkpoints=1 \
  --override keep_every=null \
  --override resume=none

echo "FLUX3_ACTION_ACTION_ONLY_DISTILL_SMOKE=$RUN_DIR"
