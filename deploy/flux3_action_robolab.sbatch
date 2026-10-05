#!/usr/bin/env bash
#SBATCH --job-name=flux3-robolab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

FLUX_REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ROOT="${AGENT_DEPLOY_ROOT:-$(dirname "$FLUX_REPO")}"
ROBOLAB_REPO="${ROBOLAB_REPO:-$ROOT/robolab}"
ISAAC_IMAGE="${ISAAC_IMAGE:?Set ISAAC_IMAGE to the Isaac Lab enroot image path}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$ROOT/.cache}"
ISAAC_ROOT_CACHE="${ISAAC_ROOT_CACHE:-$CACHE_ROOT/isaac-root}"
ISAAC_KIT_CACHE="${ISAAC_KIT_CACHE:-$CACHE_ROOT/isaac-kit}"

PORT="${FLUX_ACTION_PORT:-8000}"
TASKS="${ROBOLAB_TASKS:-BananaInBowlTask}"
NUM_ENVS="${ROBOLAB_NUM_ENVS:-1}"
NUM_RUNS="${ROBOLAB_NUM_RUNS:-1}"
RUN_NAME="${ROBOLAB_RUN_NAME:-flux3_${SLURM_JOB_ID}}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid}"
NFE="${FLUX_ACTION_NFE:-}"
COMPILE_DIT="${FLUX_ACTION_COMPILE_DIT:-1}"
SERVER_READY_TIMEOUT="${FLUX_ACTION_READY_TIMEOUT:-2700}"
VIDEO_MODE="${ROBOLAB_VIDEO_MODE:-none}"

mkdir -p "$FLUX_REPO/logs" "$ROBOLAB_REPO/logs" "$ISAAC_ROOT_CACHE" "$ISAAC_KIT_CACHE"
if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi
export TMPDIR="${FLUX_ACTION_TMPDIR:-$CACHE_ROOT/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
if [[ "$CHECKPOINT" == *fp8r* ]]; then
  export FLUX_ACTION_FP8_GEMM="${FLUX_ACTION_FP8_GEMM:-quack}"
  export FLUX_ACTION_FP8_QUANT="${FLUX_ACTION_FP8_QUANT:-triton}"
  export FLUX_ACTION_QK_EPILOGUE="${FLUX_ACTION_QK_EPILOGUE:-triton}"
  export FLUX_ACTION_ACTIVE_EPILOGUE="${FLUX_ACTION_ACTIVE_EPILOGUE:-triton}"
  export QUACK_CACHE_DIR="${QUACK_CACHE_DIR:-$CACHE_ROOT/quack/flux3-action-row-scaled-fp8-r2}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor/flux3-action-quack-e2e}"
  mkdir -p "$QUACK_CACHE_DIR"
else
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor/flux3-action-robolab}"
fi
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

SERVER_LOG="$FLUX_REPO/logs/robolab-server-${SLURM_JOB_ID}.log"
SIM_LOG="$ROBOLAB_REPO/logs/flux3-robolab-${SLURM_JOB_ID}.log"
QUERY_LOG="$FLUX_REPO/outputs/robolab-sim/${SLURM_JOB_ID}"

SERVER_ARGS=(
  serve-robolab
  --checkpoint "$CHECKPOINT"
  --host 0.0.0.0
  --port "$PORT"
  --warmup 2
  --log-dir "$QUERY_LOG"
)
if [[ -n "$NFE" ]]; then
  SERVER_ARGS+=(--setting "num_inference_steps=$NFE")
fi
if [[ "$COMPILE_DIT" == 0 ]]; then
  SERVER_ARGS+=(--no-compile-dit)
fi

cleanup() {
  if [[ -n "${SERVER_STEP_PID:-}" ]]; then
    kill "$SERVER_STEP_PID" 2>/dev/null || true
    wait "$SERVER_STEP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "node=$(hostname) task=$TASKS num_envs=$NUM_ENVS num_runs=$NUM_RUNS checkpoint=$CHECKPOINT nfe=${NFE:-checkpoint-default} compile_dit=$COMPILE_DIT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 bash -lc \
  "cd '$FLUX_REPO' && exec '$FLUX_REPO/.venv/bin/flux-action' ${SERVER_ARGS[*]@Q}" \
  >"$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!

"$FLUX_REPO/.venv/bin/python" - "$PORT" "$SERVER_STEP_PID" "$SERVER_LOG" "$SERVER_READY_TIMEOUT" <<'PY'
import os
import socket
import sys
import time

port = int(sys.argv[1])
pid = int(sys.argv[2])
log = sys.argv[3]
deadline = time.monotonic() + int(sys.argv[4])
while time.monotonic() < deadline:
    if not os.path.exists(f"/proc/{pid}"):
        raise SystemExit(f"policy server step exited before readiness; see {log}")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            print(f"policy server ready on 127.0.0.1:{port}", flush=True)
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"timed out waiting for policy server; see {log}")
PY

# The ARM64 Isaac image does not bundle two host-driver assets needed by
# NGX/OptiX.  Keep a job-local copy because Enroot cannot reliably follow the
# versioned host symlink when only the SONAME is mounted.
DRIVER_EXTRA="$CACHE_ROOT/nvidia-driver-extra-${SLURM_JOB_ID}"
mkdir -p "$DRIVER_EXTRA"
NGX_HOST_LIB="$(readlink -f /usr/lib/aarch64-linux-gnu/libnvidia-ngx.so.1)"
NGX_VERSIONED_NAME="$(basename "$NGX_HOST_LIB")"
cp "$NGX_HOST_LIB" "$DRIVER_EXTRA/$NGX_VERSIONED_NAME"
ln -sfn "$NGX_VERSIONED_NAME" "$DRIVER_EXTRA/libnvidia-ngx.so.1"
cp /usr/share/nvidia/nvoptix.bin "$DRIVER_EXTRA/nvoptix.bin"

CONTAINER_MOUNTS="$ROOT:/workspace/agent_deploy,$ISAAC_ROOT_CACHE:/root/.cache,$ISAAC_KIT_CACHE:/isaac-sim/kit/cache,$DRIVER_EXTRA:/opt/nvidia-driver-extra,$DRIVER_EXTRA/nvoptix.bin:/usr/share/nvidia/nvoptix.bin"

CUDA_VISIBLE_DEVICES=1 srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --container-image="$ISAAC_IMAGE" \
  --container-writable \
  --container-mounts="$CONTAINER_MOUNTS" \
  --container-workdir=/workspace/agent_deploy/robolab \
  --container-env=ACCEPT_EULA,OMNI_KIT_ACCEPT_EULA,CUDA_VISIBLE_DEVICES \
  bash -lc "export LD_LIBRARY_PATH=/opt/nvidia-driver-extra:\${LD_LIBRARY_PATH:-}; \
    export PYTHONPATH=/workspace/agent_deploy/robolab/.container-site:/workspace/agent_deploy/robolab; \
    exec /workspace/isaaclab/_isaac_sim/python.sh policies/cosmos3/run.py \
      --headless --device cuda:1 --remote-host 127.0.0.1 --remote-port '$PORT' \
      --task $TASKS --num-envs '$NUM_ENVS' --num-runs '$NUM_RUNS' \
      --video-mode '$VIDEO_MODE' --output-folder-name '$RUN_NAME' \
      --kit_args=--/rtx/indirectDiffuse/enabled=true" \
  >"$SIM_LOG" 2>&1

echo "simulation_log=$SIM_LOG"
echo "server_log=$SERVER_LOG"
echo "query_log=$QUERY_LOG"
echo "results=$ROBOLAB_REPO/output/$RUN_NAME"
