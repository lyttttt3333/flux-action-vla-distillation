# Action-only post-training

This path turns a BF16 FLUX.3 Action WAM checkpoint into an action-only VLA student. It is an
experimental post-training recipe, not a released BFL checkpoint or a validated replacement for the
joint video/action policy.

## Simulation outcome

Two student variants were trained and evaluated against the unmodified step-distilled WAM.
The full action-only student reached 5/30 successes across three tested RoboLab task groups,
versus 21/30 for the paired WAM baseline. The tested groups were BananaInBowl
(student 3/10, baseline 10/10), RubiksCubeAndBanana (2/10 versus 9/10), and
MarkerInMug (0/10 versus 2/10). This variant was rejected.

The later variant in `configs/droid/train_late_video_prune.json` retained future-video
tokens and shared video noise with the teacher for the first 8 of 28 joint blocks,
then removed those tokens before the remaining 20 blocks. Its first simulation gate,
BananaInBowl, reached 1/10 successes versus 10/10 for the baseline. Testing stopped
at that gate; this variant was also rejected. The local evaluation artifacts are
excluded from this Git repository.

These results show that training loss alone did not recover the action policy after
removing the video stream. In the first variant, the teacher's action target depends
on future-video noise absent from the student input. In the second, action-video
interaction continues in the joint blocks after the pruning point.

## What is removed

The student keeps text, the observed-frame latent (`video_cond`), current state (`action_cond`) and 32
noisy action tokens. It removes the future-video input/output heads, future-video mode blocks, 2,720
future-video tokens, video noise and video flow-matching loss. For the fixed DROID geometry this changes
the valid token budget from 3,173 to 453:

| Stream | WAM | Action-only student |
| --- | ---: | ---: |
| Text | 80 | 80 |
| Current image latent | 340 | 340 |
| Future video latent | 2,720 | 0 |
| Current state | 1 | 1 |
| Action chunk | 32 | 32 |
| Total | 3,173 | 453 |

The later fixed-shape inference backend will pad 453 tokens to 512. The generic BF16 path implemented
here uses the actual active streams and does not create placeholder future-video tokens.

## Initialization and training

Use the BF16 step-distilled package at `outputs/droid/variants/sd`; native FP8r/NVFP4 packages are
inference formats and are intentionally rejected as post-training initializers. The loader builds the
smaller model on the meta device, copies every matching shared/action tensor, and never materializes the
removed video-stream tensors in the student.

```sh
uv run torchrun --nproc_per_node 8 -m flux_action.cli train \
  --config configs/droid/train_action_only.json
```

The recovery objective combines two terms on every update:

1. supervised action flow matching against the real DROID action chunk at sampled training timesteps;
2. frozen-teacher action-velocity matching at the exact one-step deployment input: pure action/video
   noise with model timestep 0.999. Teacher and student share the action noise, current image, state and
   caption; only the full WAM teacher receives the 2,720 future-video noise tokens.

The second term transfers the full WAM's action distribution rather than treating structural pruning as
lossless. The first term anchors the student to real actions and prevents teacher MSE alone from learning
only an average over the hidden future-video noise. `teacher_action_mse`,
`teacher_endpoint_action_mae` and `action_mse` are logged separately; the endpoint metric applies the
released one-step solver sigma so it corresponds directly to the paired 32x8 action-chunk error.

The example's 5,000-step schedule and equal action/teacher weights are a starting point for an ablation,
not a converged or selected recipe. Compare checkpoints in simulation against the unmodified
step-distilled WAM using the same task,
environment seed, policy seed and action execution horizon. At minimum report success rate, task score,
end-effector speed/SPARC and paired action error. Do not advance to quantization on loss alone.

`init_policy` applies only when the output directory has no resumable checkpoint. Subsequent launches use
the ordinary optimizer/EMA resume path. An action-only export records
`"predict_future_video": false` and `"content_streams": ["video_cond"]`, so it reloads without an override.

## Current boundary

This first stage is the trainable generic BF16 student. The existing fixed 3,200-token FP8r/NVFP4
backend still implements the WAM geometry and is not selected for this model. The next optimization stage
needs a separate fixed 512-token prepared backend, then NVFP4 weight/activation packing and a new
simulation qualification run. Keeping those changes separate makes any quality loss attributable to
post-training versus quantization/kernel changes.

An initial structural smoke test on one GB200 used the official BF16 step-distilled weights, removed the
future-video branch at load time, and ran 10 discarded plus 30 measured requests through the generic eager
BF16 path. Server inference was 122.10 ms mean / 117.47 ms p50 / 163.84 ms p95; loopback RTT was 123.50 ms
mean / 118.81 ms p50. The result is in
`outputs/action-only-benchmark/7589975/result.json`. This model has not received action-only post-training,
so the measurement validates execution and its latency only—not action quality. It also does not use the
planned fixed-512 or NVFP4 backend.
