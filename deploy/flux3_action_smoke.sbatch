#!/usr/bin/env bash
#SBATCH --job-name=flux3-action-smoke
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

REPO="${FLUX_ACTION_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid}"
OBSERVATION="${FLUX_ACTION_OBSERVATION:-outputs/public-droid/observation.npz}"
OBSERVATION_META="${FLUX_ACTION_OBSERVATION_META:-outputs/public-droid/observation.json}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$REPO/.cache}"
cd "$REPO"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
export TMPDIR="${FLUX_ACTION_TMPDIR:-$CACHE_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ALLOCATED_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS%%,*}"

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

TASK_CAPTION="$($REPO/.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["task"])' "$OBSERVATION_META")"
RUN_DIR="outputs/droid/inference-${SLURM_JOB_ID}"

"$REPO/.venv/bin/flux-action" infer \
  --checkpoint "$CHECKPOINT" \
  --observation "$OBSERVATION" \
  --task "$TASK_CAPTION" \
  --output "$RUN_DIR"

RUN_DIR="$RUN_DIR" "$REPO/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

run_dir = Path(os.environ["RUN_DIR"])
actions = np.load(run_dir / "actions.npy")
report = json.loads((run_dir / "report.json").read_text())
assert actions.shape == (1, 32, 8), actions.shape
assert np.isfinite(actions).all()
print(f"validated actions={actions.shape} dtype={actions.dtype}")
print(json.dumps(report, indent=2)[:4000])
PY

echo "FLUX3_ACTION_SMOKE_OK=$RUN_DIR"
