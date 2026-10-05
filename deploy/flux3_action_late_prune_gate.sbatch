#!/usr/bin/env bash
#SBATCH --job-name=flux3-late-prune-gate
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

FLUX_REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CHECKPOINT="${FLUX_ACTION_TRAINING_CHECKPOINT:-$FLUX_REPO/outputs/droid/late-video-prune-k8-distill-seed42/step-500}"
EXPORT="${FLUX_ACTION_EXPORT:-$FLUX_REPO/outputs/droid/late-video-prune-k8-distill-seed42/export-step500-ema0p10-bf16}"
cd "$FLUX_REPO"

CODE_STORAGE_ROOT="$(readlink -f "$HOME/code")"
export HF_HOME="${HF_HOME:-$CODE_STORAGE_ROOT/.cache/huggingface}"

if [[ ! -f "$EXPORT/manifest.json" ]]; then
  "$FLUX_REPO/.venv/bin/flux-action" export-checkpoint \
    --checkpoint "$CHECKPOINT" \
    --output "$EXPORT" \
    --profile ema_0p10 \
    --dtype bfloat16
fi

export ISAAC_IMAGE="${ISAAC_IMAGE:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/.cache/enroot/isaaclab-2.3.0-arm64.sqsh}"
export FLUX_ACTION_CHECKPOINT="$EXPORT"
export FLUX_ACTION_NFE=1
export FLUX_ACTION_COMPILE_DIT=0
export ROBOLAB_TASKS=BananaInBowlTask
export ROBOLAB_NUM_ENVS=10
export ROBOLAB_NUM_RUNS=1
export ROBOLAB_VIDEO_MODE=all
export ROBOLAB_RUN_NAME="late_prune_k8_step500_ema0p10_banana_10seeds_${SLURM_JOB_ID}"

source "$FLUX_REPO/deploy/flux3_action_robolab.sbatch"
