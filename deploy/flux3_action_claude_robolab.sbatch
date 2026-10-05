#!/usr/bin/env bash
#SBATCH --job-name=claude-opus-robolab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

FLUX_REPO="${FLUX_ACTION_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ROOT="${AGENT_DEPLOY_ROOT:-$(dirname "$FLUX_REPO")}"
ROBOLAB_REPO="${ROBOLAB_REPO:-$ROOT/robolab}"
ISAAC_IMAGE="${ISAAC_IMAGE:?Set ISAAC_IMAGE to the Isaac Lab enroot image path}"
CACHE_ROOT="${FLUX_ACTION_CACHE_ROOT:-$ROOT/.cache}"
ISAAC_ROOT_CACHE="${ISAAC_ROOT_CACHE:-$CACHE_ROOT/isaac-root}"
ISAAC_KIT_CACHE="${ISAAC_KIT_CACHE:-$CACHE_ROOT/isaac-kit}"

VLA_PORT="${FLUX_ACTION_PORT:-8000}"
PIPELINE_PORT="${CLAUDE_VLA_PORT:-8001}"
TASKS="${ROBOLAB_TASKS:-BananaInBowlTask}"
NUM_ENVS="${ROBOLAB_NUM_ENVS:-1}"
NUM_RUNS="${ROBOLAB_NUM_RUNS:-1}"
RUN_NAME="${ROBOLAB_RUN_NAME:-claude_opus_5_5_${SLURM_JOB_ID}}"
VIDEO_MODE="${ROBOLAB_VIDEO_MODE:-all}"
CHECKPOINT="${FLUX_ACTION_CHECKPOINT:-outputs/droid/variants/fp8r}"
NFE="${FLUX_ACTION_NFE:-4}"
CLAUDE_BIN="${CLAUDE_BIN:-/home/yitongl/.local/bin/claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-5-5}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-high}"
READY_TIMEOUT="${FLUX_ACTION_READY_TIMEOUT:-2700}"

mkdir -p \
  "$FLUX_REPO/logs" \
  "$FLUX_REPO/outputs" \
  "$ROBOLAB_REPO/logs" \
  "$ISAAC_ROOT_CACHE" \
  "$ISAAC_KIT_CACHE"

if [[ -n "${FLUX_ACTION_ENV_SETUP:-}" ]]; then
  source "$FLUX_ACTION_ENV_SETUP"
fi

export TMPDIR="${FLUX_ACTION_TMPDIR:-/tmp/claude-robolab-${SLURM_JOB_ID}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONUNBUFFERED=1
export HF_HOME="${FLUX_ACTION_HF_HOME:-$(dirname "$ROOT")/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export FLUX_ACTION_FP8_GEMM="${FLUX_ACTION_FP8_GEMM:-quack}"
export FLUX_ACTION_FP8_QUANT="${FLUX_ACTION_FP8_QUANT:-triton}"
export FLUX_ACTION_QK_EPILOGUE="${FLUX_ACTION_QK_EPILOGUE:-triton}"
export FLUX_ACTION_ACTIVE_EPILOGUE="${FLUX_ACTION_ACTIVE_EPILOGUE:-triton}"
export QUACK_CACHE_DIR="${QUACK_CACHE_DIR:-$CACHE_ROOT/quack/flux3-action-row-scaled-fp8-r2}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor/flux3-action-quack-e2e}"
mkdir -p "$TMPDIR" "$QUACK_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

VLA_LOG="$FLUX_REPO/logs/claude-robolab-vla-${SLURM_JOB_ID}.log"
PROXY_LOG="$FLUX_REPO/logs/claude-robolab-proxy-${SLURM_JOB_ID}.log"
PLANNER_JSONL="$FLUX_REPO/outputs/claude-robolab-${SLURM_JOB_ID}.jsonl"
SIM_LOG="$ROBOLAB_REPO/logs/claude-robolab-${SLURM_JOB_ID}.log"
QUERY_LOG="$FLUX_REPO/outputs/claude-robolab-vla/${SLURM_JOB_ID}"

cleanup() {
  if [[ -n "${PROXY_PID:-}" ]]; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLA_PID:-}" ]]; then
    kill "$VLA_PID" 2>/dev/null || true
    wait "$VLA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "node=$(hostname) tasks=$TASKS num_envs=$NUM_ENVS num_runs=$NUM_RUNS checkpoint=$CHECKPOINT nfe=$NFE planner=$CLAUDE_MODEL effort=$CLAUDE_EFFORT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 "$FLUX_REPO/.venv/bin/flux-action" serve-robolab \
  --checkpoint "$CHECKPOINT" \
  --setting "num_inference_steps=$NFE" \
  --host 127.0.0.1 \
  --port "$VLA_PORT" \
  --warmup 2 \
  --log-dir "$QUERY_LOG" \
  >"$VLA_LOG" 2>&1 &
VLA_PID=$!

"$FLUX_REPO/.venv/bin/python" - "$VLA_PORT" "$VLA_PID" "$VLA_LOG" "$READY_TIMEOUT" <<'PY'
import os
import socket
import sys
import time

port, pid, log, timeout = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    if not os.path.exists(f"/proc/{pid}"):
        raise SystemExit(f"VLA exited before readiness; see {log}")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            print(f"VLA ready on 127.0.0.1:{port}", flush=True)
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"timed out waiting for VLA; see {log}")
PY

CUDA_VISIBLE_DEVICES="" "$FLUX_REPO/.venv/bin/python" "$FLUX_REPO/deploy/gpt6_vla_proxy.py" \
  --planner-backend claude \
  --upstream "ws://127.0.0.1:${VLA_PORT}" \
  --host 0.0.0.0 \
  --port "$PIPELINE_PORT" \
  --claude-bin "$CLAUDE_BIN" \
  --codex-work-root "$TMPDIR/claude-planner" \
  --model "$CLAUDE_MODEL" \
  --reasoning-effort "$CLAUDE_EFFORT" \
  --jpeg-quality 90 \
  --timeout 300 \
  --max-retries 1 \
  --max-budget-usd 0.25 \
  --log-path "$PLANNER_JSONL" \
  >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

"$FLUX_REPO/.venv/bin/python" - "$PIPELINE_PORT" "$PROXY_PID" "$PROXY_LOG" <<'PY'
import os
import socket
import sys
import time

port, pid, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    if not os.path.exists(f"/proc/{pid}"):
        raise SystemExit(f"planner proxy exited before readiness; see {log}")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            print(f"Claude -> VLA proxy ready on 127.0.0.1:{port}", flush=True)
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"timed out waiting for planner proxy; see {log}")
PY

# Restore host driver assets that the ARM64 Isaac image does not bundle.
DRIVER_EXTRA="$CACHE_ROOT/nvidia-driver-extra-${SLURM_JOB_ID}"
mkdir -p "$DRIVER_EXTRA"
cp -L /usr/lib/aarch64-linux-gnu/libnvidia-ngx.so.1 "$DRIVER_EXTRA/libnvidia-ngx.so.1"
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
      --headless --device cuda:1 --remote-host 127.0.0.1 --remote-port '$PIPELINE_PORT' \
      --task $TASKS --num-envs '$NUM_ENVS' --num-runs '$NUM_RUNS' \
      --video-mode '$VIDEO_MODE' --output-folder-name '$RUN_NAME' \
      --kit_args=--/rtx/indirectDiffuse/enabled=true" \
  >"$SIM_LOG" 2>&1

echo "simulation_log=$SIM_LOG"
echo "vla_log=$VLA_LOG"
echo "proxy_log=$PROXY_LOG"
echo "planner_jsonl=$PLANNER_JSONL"
echo "vla_queries=$QUERY_LOG"
echo "results=$ROBOLAB_REPO/output/$RUN_NAME"
