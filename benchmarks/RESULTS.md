# Recorded DROID steady-state results on GB200

Input: `outputs/public-droid/observation.npz`, task `Pour the contents of the yellow cup into the
pink bowl`, and one persistent WebSocket connection. The baseline uses 5 discarded warm-up
requests and 30 measured requests; the final deployment was rechecked with 10 warm-ups and 300
measured requests. Each response contains 32 actions at 15 Hz (2.133 seconds of control trajectory).

## Current single-card Base FP8r path (4 NFE)

These direct-process measurements use 10 warmups and 30 steady-state requests.
They retain the released Base FP8r checkpoint and four solver evaluations.

| Kernel path | Wall mean | Wall p50 | DiT mean |
|---|---:|---:|---:|
| Quack FP8 GEMMs + generated epilogues | 171.29 ms | 170.41 ms | 155.48 ms |
| + single-pass SwiGLU/rowwise-FP8 | 159.66 ms | 158.68 ms | 144.01 ms |
| + single-load Q/K RMSNorm+RoPE | 155.29 ms | 154.12 ms | 138.90 ms |
| + packed joint LayerNorm/modulation/FP8 producer | **144.92 ms** | **144.48 ms** | **128.95 ms** |

The Q/K request-trace path falls from 15.667 ms (RMSNorm/RoPE plus its
slice/transpose companion) to 5.174 ms over 112 launches. The Triton backend is
deterministic but not bit-exact with the generated Inductor fallback: the
same-seed final action difference is 0.04722 max and 0.00910 mean absolute.
Set `FLUX_ACTION_QK_EPILOGUE=torch` when exact fallback replay is required.

The final joint producer packs all five streams' shift/scale pairs once per NFE.
Its packing kernel runs four times per request (7.84 us total), while the fused
LayerNorm/modulation/row-quant kernel takes 3.012 ms over 112 blocks. A
same-node 30-request A/B measured 154.66 -> 144.92 ms wall and 137.51 ->
128.95 ms DiT. A separate node recorded 140.64 ms wall / 124.25 ms DiT before
the final packing change; this is retained as the best observed run, not used
as the controlled speedup claim. The difference is ordinary GB200 node/DVFS
variation in common GEMM and attention kernels.

The fused active backend is deterministic but not bit-exact with Inductor's
LayerNorm reduction. On the same seed, final actions differ by 0.01006 mean and
0.06815 max absolute (0.240% and 1.627% of the reference action range).
Set `FLUX_ACTION_ACTIVE_EPILOGUE=torch` for exact fallback-backend replay.

## Single-card Base NVFP4 path (4 NFE)

The SM100 NVFP4 tier keeps the released Base FP8r checkpoint and the same four
solver evaluations. At model load, the released row-FP8 GEMM weights are
restored to BF16 and packed once into E2M1 weights with E4M3 block scales.
Runtime activations use the same block-scaled format and Quack's native SM100
block-scaled GEMM. This is a quantized execution backend, not a new checkpoint.

| Backend | Wall mean | Wall p50 | Wall p95 | Sampler mean | DiT mean |
|---|---:|---:|---:|---:|---:|
| FP8r packed epilogues | 144.92 ms | 144.48 ms | 148.71 ms | 137.04 ms | 128.95 ms |
| NVFP4, 10 warmups + 30 requests | **117.97 ms** | **116.21 ms** | **121.92 ms** | **109.98 ms** | **102.74 ms** |

NVFP4 reduces mean request wall time by 18.60%, median wall time by 19.57%,
and mean DiT time by 20.33%. The 120 ms target is met for wall mean and p50;
p95 is 1.92 ms above the target. One 154.97 ms host-side outlier is included
in the reported mean and maximum. All numbers are single-card GB200, Base
FP8r, four NFE, batch-CFG, and the same recorded DROID observation.

The custom activation quantizer is byte-exact with Quack's canonical packed
E2M1 data and E4M3 scale layout on the supported fixed shapes. The joint
LayerNorm/modulation producer uses two kernels: one statistics pass followed
by a tiled modulation-and-quantization pass, avoiding a full BF16 active
tensor. The SwiGLU producer directly writes an NVFP4 operand. Same-seed final
actions are deterministic; relative to the FP8r backend their mean and maximum
absolute differences are 0.04537 and 0.15111 (1.081% and 3.601% of the FP8r
action span). Closed-loop simulation results and qualification limits are in
`profile/flux3-action-robolab-nvfp4-r1-20260930/REPORT.md`.

| Deployment | Server mean | Server p50 | Server p95 | RTT mean | Chunks/s |
|---|---:|---:|---:|---:|---:|
| BF16 eager, Torch SDPA | 527.93 ms | 527.93 ms | 534.29 ms | 532.04 ms | 1.89 |
| BF16 compiled, hybrid cuDNN SDPA | 227.24 ms | 227.27 ms | 230.86 ms | 230.83 ms | 4.40 |
| BF16 compiled, hybrid cuDNN SDPA, batched CFG | 213.61 ms | 213.22 ms | 216.44 ms | 217.44 ms | 4.68 |
| BF16, 2×TP2 parallel CFG, fused UniPC | **83.73 ms** | **83.51 ms** | **85.72 ms** | **87.22 ms** | **11.94** |

The final four-GPU deployment is 6.31x faster in server inference and 6.10x faster end to end than
the original BF16 service. Its RTT mean is 87.22 ms and RTT p95 is 89.27 ms, both below the 90 ms
target. It generates the 2.133-second action chunk 25.47x faster than real time. Relative to the
previous batched-CFG deployment, server latency falls another 60.8%.

Direct-process variant measurements (same recorded input, 30 requests):

| Variant | Mean | p50 | Chunks/s | Semantics |
|---|---:|---:|---:|---|
| BF16 eager + hybrid cuDNN | 454.79 ms | 455.04 ms | 2.20 | Same base model |
| BF16 compiled + hybrid cuDNN | 228.22 ms | 228.06 ms | 4.38 | Same base model |
| BF16 compiled + hybrid cuDNN + batched CFG | 200.05 ms | 200.09 ms | 5.00 | Same base model and sampling |
| BF16 2×TP2 parallel CFG + fused UniPC | 81.77 ms | 81.70 ms | 12.23 | Same BF16 model, CFG, 4 NFE and schedule |
| FP8r compiled + hybrid cuDNN | 210.68 ms | 210.41 ms | 4.75 | Official rowwise-FP8 package |
| Guidance-distilled FP8r compiled | 109.29 ms | 109.18 ms | 9.15 | Official no-CFG model variant |
| Step-distilled FP8r compiled | 34.95 ms | 34.81 ms | 28.61 | Official one-step model variant |

The distilled results are model/sampling changes, not kernel-only speedups, and are therefore not
the default deployed endpoint. FP8r is also not promoted by default: it gives only another 7.7% over
compiled BF16 here while its same-seed action difference from BF16 eager is 0.0914 max absolute.
The guidance-distilled and step-distilled comparisons differ by 0.2396 and 0.1574 max absolute,
respectively; these figures are only numerical differences on one input, not task-success metrics.

The batched-CFG eager path was compared to the original serial-CFG path using the same seed and
produced exactly identical actions (max and mean absolute difference both zero). The compiled
path differs from eager BF16 by 0.0212 max absolute and 0.00438 mean absolute, consistent with
compiled floating-point reassociation rather than a model or sampler change.
The final distributed/fused path differs from the original single-GPU eager path by 0.01569 max
absolute and 0.00360 mean absolute on the same seed. A two-GPU batched-CFG TP block measured
0.959 ms and was not promoted because its projected end-to-end latency did not satisfy the
required 90 ms two-GPU limit.

Raw service JSON and Nsight reports are intentionally excluded from the repository because they
contain cluster hostnames and large profiler payloads. The table above records the validated
aggregate measurements.
