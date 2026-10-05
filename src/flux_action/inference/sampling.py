# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Joint video + action denoising for policy inference.

Two samplers over a dict of noised streams (``x_video`` and ``x_<modality>``):

* ``cosmos_unipc_order2`` — the literal Cosmos UniPC (order 2, bh2, predict-x0)
  our DROID policy was evaluated with: 4 steps, shift 5, CFG 3. One denoiser
  call per step (the corrector does not re-evaluate the model), integer
  scheduler ticks passed to the model as ``tick / 1000``.
* ``euler`` — plain rectified-flow Euler on the shifted schedule, for reference.

``cfg_two_pass`` runs the unconditional (empty caption) and conditional forward
separately, exactly like the video pipeline, and combines per stream so the
video and action streams can use different guidance scales.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

Samples = dict[str, Tensor]


def cfg_two_pass(
    model,
    flow: Samples,
    fixed: dict[str, Tensor],
    timesteps: dict[str, Tensor],
    ctx_uc: tuple[Tensor, Tensor],
    ctx_c: tuple[Tensor, Tensor],
    cfg_by_key: dict[str, float],
) -> Samples:
    """pred = uc + cfg * (c - uc) per flow stream. ``fixed`` = ids, cond streams, vector."""

    def forward(ctx: Tensor, ctx_ids: Tensor) -> dict[str, Tensor]:
        return model(
            **flow,
            **fixed,
            **timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=torch.zeros(ctx.shape[:2], device=ctx.device, dtype=ctx.dtype),
        )

    pred_uc = forward(*ctx_uc)
    pred_c = forward(*ctx_c)
    return {k: pred_uc[k] + cfg_by_key[k] * (pred_c[k] - pred_uc[k]) for k in flow}


def timeshift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def euler(
    samples: Samples,
    predict_velocity: Callable[[Samples, float], Samples],
    *,
    n_steps: int,
    alpha: float,
) -> Samples:
    timesteps = timeshift(alpha, torch.linspace(1.0, 0.0, n_steps + 1)).tolist()
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=True):
        pred = predict_velocity(samples, t_curr)
        samples = {
            k: (samples[k].float() + (t_prev - t_curr) * pred[k].float()).to(samples[k].dtype)
            for k in samples
        }
    return samples


def cosmos_unipc_schedule(
    n_steps: int, shift: float, num_train_timesteps: int = 1000
) -> tuple[Tensor, Tensor]:
    """Cosmos' FlowUniPCMultistepScheduler grid: float sigmas for the solver, integer ticks for the model."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    sigma_max = np.float32(1.0 - 1.0 / num_train_timesteps).item()
    sigmas_np = np.linspace(sigma_max, 0.0, n_steps + 1).copy()[:-1]
    sigmas_np = shift * sigmas_np / (1.0 + (shift - 1.0) * sigmas_np)
    model_timesteps = torch.from_numpy((sigmas_np * num_train_timesteps).astype(np.int64))
    sigmas = torch.from_numpy(np.concatenate([sigmas_np, [0.0]]).astype(np.float32))
    return sigmas, model_timesteps


def _bh_coefficients(h: Tensor, rks: list[Tensor], order: int, *, device, dtype, corrector: bool):
    """bh2 predictor / corrector coefficients (predict_x0=True), as in Cosmos."""
    hh = -h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1.0
    b_h = torch.expm1(hh)
    rks_t = torch.stack([rk.to(device=device) for rk in rks] + [torch.ones_like(h, device=device)])
    rows, rhs = [], []
    factorial_i = 1
    for i in range(1, order + 1):
        rows.append(torch.pow(rks_t, i - 1))
        rhs.append((h_phi_k * factorial_i / b_h).to(device=device))
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i
    matrix, rhs_t = torch.stack(rows), torch.stack(rhs)
    if corrector:
        rhos = (
            torch.tensor([0.5], dtype=dtype, device=device)
            if order == 1
            else torch.linalg.solve(matrix, rhs_t).to(dtype)
        )
    elif order == 2:
        rhos = torch.tensor([0.5], dtype=dtype, device=device)  # Cosmos' simplified UniP order-2 coefficient
    else:
        rhos = None
    return h_phi_1, b_h, rhos


def cosmos_unipc_order2(
    samples: Samples,
    predict_velocity: Callable[[Samples, Tensor], Samples],
    *,
    n_steps: int,
    shift: float,
    num_train_timesteps: int = 1000,
) -> Samples:
    """Cosmos UniPC (order 2, bh2, predict-x0) over a dict of streams sharing one grid."""
    keys = list(samples)
    sigmas, model_timesteps = cosmos_unipc_schedule(n_steps, shift, num_train_timesteps)
    solver_order = 2
    model_outputs: list[Samples | None] = [None] * solver_order
    lower_order_nums = 0
    last_sample: Samples | None = None
    this_order = 1

    def alpha_sigma(s: Tensor) -> tuple[Tensor, Tensor]:
        return 1.0 - s, s

    def lam(s: Tensor) -> Tensor:
        a, sg = alpha_sigma(s)
        return torch.log(a) - torch.log(sg)

    for step, tick in enumerate(model_timesteps):
        velocity = predict_velocity(samples, tick)
        sigma_cur = sigmas[step]
        x0 = {k: samples[k].float() - sigma_cur * velocity[k].float() for k in keys}

        if step > 0 and last_sample is not None:  # UniC corrector, using the previous step's order
            order_c = this_order
            sigma_t, sigma_s0 = sigmas[step], sigmas[step - 1]
            alpha_t, sig_t = alpha_sigma(sigma_t)
            _, sig_s0 = alpha_sigma(sigma_s0)
            h = lam(sigma_t) - lam(sigma_s0)
            rks, hist = [], []
            prev_x0 = model_outputs[-1]
            for i in range(1, order_c):
                older = model_outputs[-(i + 1)]
                rk = (lam(sigmas[step - (i + 1)]) - lam(sigma_s0)) / h
                rks.append(rk)
                hist.append({k: (older[k] - prev_x0[k]) / rk for k in keys})
            ex = samples[keys[0]]
            h_phi_1, b_h, rhos_c = _bh_coefficients(
                h, rks, order_c, device=ex.device, dtype=torch.float32, corrector=True
            )
            corrected = {}
            for k in keys:
                base = sig_t / sig_s0 * last_sample[k].float() - alpha_t * h_phi_1 * prev_x0[k]
                residual = sum(rhos_c[i] * d[k] for i, d in enumerate(hist)) if hist else 0.0
                corrected[k] = (base - alpha_t * b_h * (residual + rhos_c[-1] * (x0[k] - prev_x0[k]))).to(
                    samples[k].dtype
                )
            samples = corrected

        model_outputs[0] = model_outputs[1]
        model_outputs[1] = x0  # history holds the pre-correction x0, as in Cosmos

        remaining = len(model_timesteps) - step
        this_order = min(min(solver_order, remaining), lower_order_nums + 1)
        last_sample = samples

        sigma_t, sigma_s0 = sigmas[step + 1], sigmas[step]  # UniP predictor
        alpha_t, sig_t = alpha_sigma(sigma_t)
        _, sig_s0 = alpha_sigma(sigma_s0)
        h = lam(sigma_t) - lam(sigma_s0)
        rks, hist = [], []
        latest = model_outputs[-1]
        for i in range(1, this_order):
            older = model_outputs[-(i + 1)]
            rk = (lam(sigmas[step - i]) - lam(sigma_s0)) / h
            rks.append(rk)
            hist.append({k: (older[k] - latest[k]) / rk for k in keys})
        ex = samples[keys[0]]
        h_phi_1, b_h, rhos_p = _bh_coefficients(
            h, rks, this_order, device=ex.device, dtype=torch.float32, corrector=False
        )
        predicted = {}
        for k in keys:
            base = sig_t / sig_s0 * samples[k].float() - alpha_t * h_phi_1 * latest[k]
            if hist:
                base = base - alpha_t * b_h * sum(rhos_p[i] * d[k] for i, d in enumerate(hist))
            predicted[k] = base.to(samples[k].dtype)
        samples = predicted
        if lower_order_nums < solver_order:
            lower_order_nums += 1
    return samples


@torch.compile(dynamic=False, fullgraph=True, mode="reduce-overhead")
def _unipc_step0(xv, xa, vv, va, sigma0: float, sample0: float, x0_weight: float):
    x0v = xv - sigma0 * vv
    x0a = xa - sigma0 * va
    return (
        sample0 * xv + x0_weight * x0v,
        sample0 * xa + x0_weight * x0a,
        x0v,
        x0a,
    )


@torch.compile(dynamic=False, fullgraph=True, mode="reduce-overhead")
def _unipc_step1(
    xv,
    xa,
    vv,
    va,
    initial_v,
    initial_a,
    x00v,
    x00a,
    sigma1: float,
    correct_last: float,
    correct_prev: float,
    correct_delta: float,
    predict_sample: float,
    predict_x0: float,
    predict_history: float,
    history_scale: float,
):
    x01v = xv - sigma1 * vv
    x01a = xa - sigma1 * va
    corrected_v = (
        correct_last * initial_v
        + correct_prev * x00v
        + correct_delta * (x01v - x00v)
    )
    corrected_a = (
        correct_last * initial_a
        + correct_prev * x00a
        + correct_delta * (x01a - x00a)
    )
    next_v = (
        predict_sample * corrected_v
        + predict_x0 * x01v
        + predict_history * ((x00v - x01v) * history_scale)
    )
    next_a = (
        predict_sample * corrected_a
        + predict_x0 * x01a
        + predict_history * ((x00a - x01a) * history_scale)
    )
    return next_v, next_a, corrected_v, corrected_a, x01v, x01a


@torch.compile(dynamic=False, fullgraph=True, mode="reduce-overhead")
def _unipc_step2(
    xv,
    xa,
    vv,
    va,
    last_v,
    last_a,
    older_x0v,
    older_x0a,
    previous_x0v,
    previous_x0a,
    sigma: float,
    correct_last: float,
    correct_prev: float,
    correct_history: float,
    correct_delta: float,
    correct_history_scale: float,
    predict_sample: float,
    predict_x0: float,
    predict_history: float,
    predict_history_scale: float,
):
    current_x0v = xv - sigma * vv
    current_x0a = xa - sigma * va
    corrected_v = (
        correct_last * last_v
        + correct_prev * previous_x0v
        + correct_history
        * ((older_x0v - previous_x0v) * correct_history_scale)
        + correct_delta * (current_x0v - previous_x0v)
    )
    corrected_a = (
        correct_last * last_a
        + correct_prev * previous_x0a
        + correct_history
        * ((older_x0a - previous_x0a) * correct_history_scale)
        + correct_delta * (current_x0a - previous_x0a)
    )
    next_v = (
        predict_sample * corrected_v
        + predict_x0 * current_x0v
        + predict_history
        * ((previous_x0v - current_x0v) * predict_history_scale)
    )
    next_a = (
        predict_sample * corrected_a
        + predict_x0 * current_x0a
        + predict_history
        * ((previous_x0a - current_x0a) * predict_history_scale)
    )
    return next_v, next_a, corrected_v, corrected_a, current_x0v, current_x0a


@torch.compile(dynamic=False, fullgraph=True, mode="reduce-overhead")
def _unipc_final(xv, xa, vv, va, sigma3: float):
    return xv - sigma3 * vv, xa - sigma3 * va


def _order2_coefficients(sigmas: Tensor, step: int) -> tuple[float, ...]:
    sigma_t, sigma_s0 = sigmas[step], sigmas[step - 1]
    alpha_t = 1.0 - sigma_t
    h = torch.log1p(-sigma_t) - torch.log(sigma_t) - (
        torch.log1p(-sigma_s0) - torch.log(sigma_s0)
    )
    rk = (
        torch.log1p(-sigmas[step - 2])
        - torch.log(sigmas[step - 2])
        - (torch.log1p(-sigma_s0) - torch.log(sigma_s0))
    ) / h
    h_phi_1, b_h, rhos = _bh_coefficients(
        h, [rk], 2, device=torch.device("cpu"), dtype=torch.float32, corrector=True
    )
    return tuple(
        float(value)
        for value in (
            sigma_t / sigma_s0,
            -alpha_t * h_phi_1,
            -alpha_t * b_h * rhos[0],
            -alpha_t * b_h * rhos[1],
            1.0 / rk,
        )
    )


def cosmos_unipc_order2_fused4(
    samples: Samples,
    predict_velocity: Callable[[Samples, Tensor], Samples],
    *,
    shift: float,
    num_train_timesteps: int = 1000,
) -> Samples:
    """Four-step UniPC with the unchanged denoiser calls and fused FP32 solver arithmetic."""
    keys = list(samples)
    if len(keys) != 2:
        raise ValueError("fused four-step sampler requires exactly two streams")
    video_key, action_key = keys
    sigmas, ticks = cosmos_unipc_schedule(4, shift, num_train_timesteps)
    s = tuple(float(value) for value in sigmas)

    initial_v, initial_a = samples[video_key], samples[action_key]
    velocity = predict_velocity(samples, ticks[0])
    h01 = torch.log1p(-sigmas[1]) - torch.log(sigmas[1]) - (
        torch.log1p(-sigmas[0]) - torch.log(sigmas[0])
    )
    hphi01 = float(torch.expm1(-h01))
    xv, xa, x00v, x00a = _unipc_step0(
        initial_v,
        initial_a,
        velocity[video_key],
        velocity[action_key],
        s[0],
        s[1] / s[0],
        -(1.0 - s[1]) * hphi01,
    )

    velocity = predict_velocity({video_key: xv, action_key: xa}, ticks[1])
    h12 = torch.log1p(-sigmas[2]) - torch.log(sigmas[2]) - (
        torch.log1p(-sigmas[1]) - torch.log(sigmas[1])
    )
    hphi12 = float(torch.expm1(-h12))
    rk12 = float(-h01 / h12)
    correct_delta1 = -(1.0 - s[1]) * hphi01 * 0.5
    xv, xa, last_v, last_a, x01v, x01a = _unipc_step1(
        xv,
        xa,
        velocity[video_key],
        velocity[action_key],
        initial_v,
        initial_a,
        x00v,
        x00a,
        s[1],
        s[1] / s[0],
        -(1.0 - s[1]) * hphi01,
        correct_delta1,
        s[2] / s[1],
        -(1.0 - s[2]) * hphi12,
        -(1.0 - s[2]) * hphi12 * 0.5,
        1.0 / rk12,
    )

    velocity = predict_velocity({video_key: xv, action_key: xa}, ticks[2])
    h23 = torch.log1p(-sigmas[3]) - torch.log(sigmas[3]) - (
        torch.log1p(-sigmas[2]) - torch.log(sigmas[2])
    )
    hphi23 = float(torch.expm1(-h23))
    c2_last, c2_prev, c2_hist, c2_delta, c2_scale = _order2_coefficients(sigmas, 2)
    rk23 = float(-h12 / h23)
    xv, xa, last_v, last_a, x02v, x02a = _unipc_step2(
        xv,
        xa,
        velocity[video_key],
        velocity[action_key],
        last_v,
        last_a,
        x00v,
        x00a,
        x01v,
        x01a,
        s[2],
        c2_last,
        c2_prev,
        c2_hist,
        c2_delta,
        c2_scale,
        s[3] / s[2],
        -(1.0 - s[3]) * hphi23,
        -(1.0 - s[3]) * hphi23 * 0.5,
        1.0 / rk23,
    )

    velocity = predict_velocity({video_key: xv, action_key: xa}, ticks[3])
    xv, xa = _unipc_final(xv, xa, velocity[video_key], velocity[action_key], s[3])
    return {video_key: xv, action_key: xa}
