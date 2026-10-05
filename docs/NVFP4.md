# GB200 NVFP4 inference path

This fork adds an experimental single-GPU SM100 inference backend for the
released FLUX.3 Action FP8r checkpoints. It does not introduce a new model
checkpoint. Released row-FP8 weights are restored to BF16 once at load time
and packed into NVFP4 E2M1 data with E4M3 block scales. Runtime activations are
quantized into the same block-scaled layout and executed by Quack's native
SM100 GEMMs.

## Implementation

The main integration is in
`src/flux_action/models/transformer_inf_fp8r.py`. The custom Triton producers
are in `src/flux_action/kernels/`:

- `fp8r_quant.py`: fused SwiGLU and rowwise E4M3 production;
- `qk_norm_rope.py`: packed Q/K RMSNorm and interleaved RoPE;
- `joint_active_fp8r.py`: joint LayerNorm, modulation and FP8 production;
- `nvfp4_quant.py`: SM100-compatible block-scaled E2M1 packing;
- `nvfp4_producers.py`: direct joint-active and SwiGLU NVFP4 producers.

The NVFP4 quantizer matches Quack's canonical packed data and scale layout on
the supported fixed DROID shapes. Model-level NVFP4 execution is deterministic
under repeat execution, but it is not numerically identical to FP8r.

Install the Blackwell dependency and select the backend with:

```bash
uv sync --extra encoders --extra serve --extra blackwell
export FLUX_ACTION_FP8_GEMM=nvfp4
export FLUX_ACTION_NVFP4_SCOPE=all
export FLUX_ACTION_FP8_QUANT=triton
export FLUX_ACTION_QK_EPILOGUE=triton
export FLUX_ACTION_ACTIVE_EPILOGUE=triton
```

`FLUX_ACTION_NVFP4_SCOPE=input` and `output` retain one side of the repeated
DiT projections in FP8r for accuracy/performance diagnosis.

## Single-card results

All measurements below use one visible NVIDIA GB200 and return a 32-by-8
action chunk. Cluster launchers may reserve a complete node, but mask all GPUs
except the first allocation entry from the model process.

| Checkpoint | Backend | NFE | Wall mean | Wall p50 | DiT p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Base FP8r | fused FP8r | 4 | 144.92 ms | 144.48 ms | 128.95 ms |
| Base FP8r | full NVFP4 | 4 | **117.97 ms** | **116.21 ms** | 102.74 ms mean |
| Step-distilled FP8r | fused FP8r | 1 | 36.03 ms | 35.52 ms | 23.56 ms |
| Step-distilled FP8r | full NVFP4 | 1 | **30.80 ms** | **30.22 ms** | **18.04 ms** |

The one-step test used ten warmups and 100 measured requests. Full NVFP4
reduced median request latency by 14.9% and median DiT latency by 23.5%. A
same-input, same-seed diagnostic measured 0.0431 mean and 0.1172 maximum
absolute action difference relative to fused FP8r.

The four-step full-NVFP4 path was also evaluated in a paired 40-episode
RoboLab smoke benchmark: 24/40 successes versus 26/40 for FP8r. It is therefore
an experimental latency tier, not a lossless replacement. The one-step NVFP4
combination has only received latency and same-seed drift checks so far; it has
not yet received a new closed-loop simulation qualification.

## Launch

The Slurm launcher preserves the checkpoint's native NFE unless
`FLUX_ACTION_NFE` is explicitly set:

```bash
# Base checkpoint, native 4 NFE
sbatch --export=ALL,FLUX_ACTION_CHECKPOINT=outputs/droid/variants/fp8r \
  deploy/flux3_action_singlecard_nvfp4.sbatch

# Step-distilled checkpoint, native 1 NFE and no runtime CFG
sbatch --export=ALL,FLUX_ACTION_CHECKPOINT=outputs/droid/variants/sd-fp8r \
  deploy/flux3_action_singlecard_nvfp4.sbatch
```

Model weights, recorded observations, compiler caches, profiler traces and
simulation videos are intentionally excluded from the repository.
