# FLUX 3 Action deployment

The deployment uses the released DROID BF16 policy and the official OpenPI-compatible
RoboLab WebSocket server. Model and encoder revisions are pinned by the downloaded policy
package; jobs run with Hugging Face offline mode enabled. The long-running service enables
the compiled DiT/VAE path and uses cuDNN SDPA for the long video and joint-attention streams.
Positive and negative classifier-free-guidance branches with compatible text buckets are evaluated
as one batch, while incompatible buckets automatically retain the serial fallback.
Its first warm-up compiles the static request shapes and can take roughly three minutes on GB200.

The latency deployment uses all four GB200s as two tensor-parallel replicas. Each TP=2 replica
runs one classifier-free-guidance branch, and the guided predictions are reduced before every
UniPC update. Image/state payloads and the steady prompt-control path use NCCL tensors; Gloo is
used only when the prompt changes. The four-step UniPC arithmetic is compiled and fused without
changing its schedule or the four DiT evaluations.

Run and validate one recorded observation:

```bash
sbatch -A <slurm-account> deploy/flux3_action_smoke.sbatch
```

After the smoke test succeeds, start the four-GPU service for 24 hours:

```bash
sbatch -A <slurm-account> deploy/flux3_action_distributed.sbatch
```

`deploy/flux3_action_service.sbatch` remains the slower single-GPU fallback.

For the existing three-NFE FP8 latency tier, start:

```bash
sbatch -A <slurm-account> deploy/flux3_action_singlecard_fast.sbatch
```

This selects the released rowwise-FP8 checkpoint, tuned SM100 Quack GEMMs,
and three UniPC evaluations. On the recorded DROID request it measured
133.63 ms mean and 134.99 ms P95 over 30 steady-state requests. It is an
approximate latency tier: the checkpoint is FP8 and reducing four NFE to
three changes the solver result. Install the Blackwell kernel dependency with
`uv sync --extra encoders --extra serve --extra blackwell`. The cluster QOS
allocates the whole four-GPU node, but the script masks it to one visible GPU;
the model and measured latency are strictly single-card.

For the single-card four-NFE NVFP4 latency tier, start:

```bash
sbatch -A <slurm-account> deploy/flux3_action_singlecard_nvfp4.sbatch
```

This launcher retains the released Base FP8r checkpoint and four-NFE sampler,
but executes the repeated DiT GEMMs with native SM100 block-scaled NVFP4. On
the recorded DROID request, 10 warmups followed by 30 measured requests gave
117.97 ms wall mean, 116.21 ms p50, and 121.92 ms p95 on one GB200. Set
`FLUX_ACTION_NVFP4_SCOPE=input` or `output` for mixed FP8r/NVFP4 diagnosis;
the deployment default is `all`. Treat full NVFP4 as an experimental latency
tier: the final paired 40-episode RoboLab smoke benchmark measured 24/40
successes versus 26/40 for FP8r. The output-only scope measured 27/40 but does
not meet the 120 ms latency target.

The FP8r launchers also set `FLUX_ACTION_FP8_QUANT=triton`. For the fixed DROID
geometry this enables a single-pass SwiGLU + rowwise-E4M3 kernel: it keeps the
producer row live through the amax reduction and writes the quantized MLP input
without creating a BF16 intermediate or recomputing SwiGLU. Set the variable to
`torch` for the portable Inductor fallback. This optimization does not alter the
checkpoint, NFE count, quantization rule, or action tensor shape.

`FLUX_ACTION_QK_EPILOGUE=triton` additionally enables the fixed-geometry Q/K
epilogue. It reads Q and K directly from the packed projection once and fuses
RMSNorm with interleaved RoPE; set it to `torch` for the portable fallback.
Its reduction and FMA ordering differs from Inductor's generated kernel, so it
is deterministic but not bit-exact; use the Torch fallback when exact replay
against the unfused backend is required.

`FLUX_ACTION_ACTIVE_EPILOGUE=triton` enables the fixed DROID joint-block input
producer. The five streams' shift/scale tensors are packed once per NFE, then
each block computes LayerNorm, applies the indexed modulation, takes the row
amax, and writes E4M3 plus its FP32 scale in one kernel. This removes the BF16
active tensor and its separate row-quant kernel. It is deterministic, but its
LayerNorm reduction order is not bit-exact with Inductor; set it to `torch` for
the portable/exact-backend fallback.

The service writes its `ws://node:port` address to
`outputs/flux3-action-service-<job-id>.endpoint`. It accepts the OpenPI request schema
documented in `docs/setup.md`; the service predicts action chunks but never sends commands
to robot hardware by itself.

Benchmark the steady state with the same recorded observation on one persistent connection:

```bash
endpoint=$(cat outputs/flux3-action-service-<job-id>.endpoint)
.venv/bin/python deploy/benchmark_service.py \
  --endpoint "$endpoint" \
  --observation outputs/public-droid/observation.npz \
  --prompt 'Pour the contents of the yellow cup into the pink bowl' \
  --warmup 5 --iterations 30 \
  --output benchmarks/droid-service-<job-id>.json
```

The warm-up count matters: the first real prompt may use a different text bucket than the service's
synthetic compile warm-up and can trigger one additional graph capture. See `benchmarks/RESULTS.md`
for the measured GB200 baseline and optimized results.

The Slurm launchers resolve the repository from their own location. Override
`FLUX_ACTION_REPO`, `FLUX_ACTION_CHECKPOINT`, `FLUX_ACTION_CACHE_ROOT`, or
`FLUX_ACTION_ENV_SETUP` when your checkout, weights, cache, or environment setup script lives
elsewhere. The RoboLab launcher additionally requires `ISAAC_IMAGE` and accepts
`ROBOLAB_REPO`/`AGENT_DEPLOY_ROOT` overrides.

For the exact GB200 topology, ARM64 Isaac container, dependency overlay,
headless-rendering caveats, launch command, and completed closed-loop runs, see
[`ROBOLAB_GB200.md`](ROBOLAB_GB200.md).

## Serial GPT-6 Astra -> FLUX Action pipeline

`deploy/gpt6_vla_proxy.py` is an OpenPI-compatible WebSocket proxy. For every
observation it first sends the current camera views, robot state, overall task, and previous
subtask to GPT-6 Astra. Astra is forced to call a structured planner tool that returns one
non-numeric immediate subtask. The proxy then rewrites the prompt and synchronously calls the
unchanged FLUX Action VLA for its 32-step action chunk. There is no FastWAM component and no
parallel/overlapped execution.

Launch the optimized single-GB200 VLA and serial proxy together:

```bash
export OPENAI_API_KEY='...'
sbatch -A <slurm-account> --export=ALL deploy/flux3_action_gpt6_vla.sbatch
```

The endpoint is written to `outputs/gpt6-vla-<job-id>.endpoint`. Each response retains the
normal `action` and `server_timing` fields and adds `gpt6_plan` plus `pipeline_timing` with
planner, VLA RTT, and total latency. Images are sent to the OpenAI API on every request in the
strict serial configuration. The key is read only from the environment and is never logged.

Test the WebSocket plumbing without an API request by pointing the proxy at an existing VLA:

```bash
.venv/bin/python deploy/gpt6_vla_proxy.py \
  --upstream ws://127.0.0.1:8000 --port 8001 \
  --fixed-subtask 'Move the gripper above the yellow cup'
```
