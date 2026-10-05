"""Fused fixed-DROID producers for native SM100 NVFP4 operands."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_BATCH = 2
_TOKENS = 3200
_VALID = 3173
_HIDDEN = 3072
_MLP = 9216
_PACKED = 3 * _HIDDEN + 2 * _MLP
_HIDDEN_RK = (_HIDDEN // 16) // 4
_MLP_RK = (_MLP // 16) // 4


@triton.jit
def _pack_e2m1_pair(low_value, high_value, pair_scale):
    """Blackwell conversion with Quack's RNE midpoint semantics."""
    low_scaled = low_value / pair_scale
    high_scaled = high_value / pair_scale
    low_abs = tl.abs(low_scaled)
    high_abs = tl.abs(high_scaled)
    low_tie_up = (low_abs == 0.75) | (low_abs == 1.75) | (low_abs == 3.5)
    high_tie_up = (high_abs == 0.75) | (high_abs == 1.75) | (high_abs == 3.5)
    away = 1.0000001192092896
    low_scaled = tl.where(low_tie_up, low_scaled * away, low_scaled)
    high_scaled = tl.where(high_tie_up, high_scaled * away, high_scaled)
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 fp4;
          cvt.rn.satfinite.e2m1x2.f32 fp4, $2, $1;
          cvt.u32.u8 $0, fp4;
        }
        """,
        constraints="=r,f,f",
        args=[low_scaled, high_scaled],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e4m3_positive_to_fp32(value):
    bits = value.to(tl.uint8, bitcast=True)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    return tl.where(exponent == 0, mantissa.to(tl.float32) * (1.0 / 512.0), normal)


@triton.jit
def _joint_stats_kernel(sequence_ptr, stats_ptr, block_k: tl.constexpr):
    """Compute joint LayerNorm statistics once without materializing active."""
    row = tl.program_id(0)
    token = row % 3200
    column = tl.arange(0, block_k)
    valid = (column < 3072) & (token < 3173)
    value = tl.load(
        sequence_ptr + row * 3072 + column,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(value, axis=0) * (1.0 / 3072.0)
    centered = value - mean
    variance = tl.sum(centered * centered, axis=0) * (1.0 / 3072.0)
    tl.store(stats_ptr + row * 2, mean)
    tl.store(stats_ptr + row * 2 + 1, tl.rsqrt(variance + 1.0e-6))


@triton.jit
def _joint_apply_quant_kernel(
    sequence_ptr,
    mods_ptr,
    stats_ptr,
    qdata_ptr,
    blocked_scale_ptr,
    tile_k: tl.constexpr,
):
    """Apply modulation and write exact-RNE E2M1 plus blocked E4M3 scales."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    token = row % 3200
    batch = row // 3200
    local = tl.arange(0, tile_k)
    column = tile * tile_k + local
    valid = (column < 3072) & (token < 3173)
    value = tl.load(
        sequence_ptr + row * 3072 + column,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(stats_ptr + row * 2)
    rstd = tl.load(stats_ptr + row * 2 + 1)
    normalized = (value - mean) * rstd
    stream = tl.where(
        token < 80,
        0,
        tl.where(token < 2800, 1, tl.where(token < 3140, 2, tl.where(token < 3172, 3, 4))),
    )
    modulation_offset = stream * 12288 + batch * 3072 + column
    shift = tl.load(mods_ptr + modulation_offset, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(
        mods_ptr + modulation_offset + 6144,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    active = ((1.0 + scale) * normalized + shift).to(tl.bfloat16).to(tl.float32)
    active = tl.where(valid, active, 0.0)
    blocks = active.reshape((tile_k // 16, 16))
    block_amax = tl.max(tl.abs(blocks), axis=1)
    block_scale = tl.minimum(tl.maximum(block_amax * (1.0 / 6.0), 0.015625), 448.0)
    block_scale_fp8 = block_scale.to(tl.float8e4nv)
    rounded_block_scale = _e4m3_positive_to_fp32(block_scale_fp8)
    pair_columns = tl.arange(0, tile_k // 2)
    low_value = tl.gather(active, 2 * pair_columns, axis=0)
    high_value = tl.gather(active, 2 * pair_columns + 1, axis=0)
    pair_scale = (
        rounded_block_scale[:, None] + tl.zeros((tile_k // 16, 8), dtype=tl.float32)
    ).reshape((tile_k // 2,))
    packed = _pack_e2m1_pair(low_value, high_value, pair_scale)
    tl.store(
        qdata_ptr + row * (3072 // 2) + tile * (tile_k // 2) + pair_columns,
        packed,
        mask=tile * (tile_k // 2) + pair_columns < 3072 // 2,
    )
    local_scale = tl.arange(0, tile_k // 16)
    scale_block = tile * (tile_k // 16) + local_scale
    scale_offset = (
        (((row // 128) * 48 + scale_block // 4) * 32 + row % 32) * 4
        + (row // 32) % 4
    ) * 4 + scale_block % 4
    tl.store(blocked_scale_ptr + scale_offset, block_scale_fp8, mask=scale_block < 3072 // 16)


@triton.jit
def _swiglu_nvfp4_kernel(
    packed_ptr,
    qdata_ptr,
    blocked_scale_ptr,
    tile_k: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    local = tl.arange(0, tile_k)
    column = tile * tile_k + local
    valid = column < 9216
    base = row * 27648
    gate = tl.load(packed_ptr + base + 9216 + column, mask=valid, other=0.0).to(tl.float32)
    up = tl.load(packed_ptr + base + 18432 + column, mask=valid, other=0.0).to(tl.float32)
    active = gate * tl.sigmoid(gate) * up
    blocks = active.reshape((tile_k // 16, 16))
    block_amax = tl.max(tl.abs(blocks), axis=1)
    block_scale = tl.minimum(tl.maximum(block_amax * (1.0 / 6.0), 0.015625), 448.0)
    block_scale_fp8 = block_scale.to(tl.float8e4nv)
    rounded_block_scale = _e4m3_positive_to_fp32(block_scale_fp8)

    pair_columns = tl.arange(0, tile_k // 2)
    low_value = tl.gather(active, 2 * pair_columns, axis=0)
    high_value = tl.gather(active, 2 * pair_columns + 1, axis=0)
    pair_scale = (
        rounded_block_scale[:, None] + tl.zeros((tile_k // 16, 8), dtype=tl.float32)
    ).reshape((tile_k // 2,))
    packed = _pack_e2m1_pair(low_value, high_value, pair_scale)
    tl.store(
        qdata_ptr + row * (9216 // 2) + tile * (tile_k // 2) + pair_columns,
        packed,
        mask=tile * (tile_k // 2) + pair_columns < 9216 // 2,
    )

    local_scale = tl.arange(0, tile_k // 16)
    scale_block = tile * (tile_k // 16) + local_scale
    scale_offset = (
        (((row // 128) * 144 + scale_block // 4) * 32 + row % 32) * 4
        + (row // 32) % 4
    ) * 4 + scale_block % 4
    tl.store(blocked_scale_ptr + scale_offset, block_scale_fp8, mask=scale_block < 9216 // 16)


@triton_op("flux_action::joint_active_nvfp4_parts", mutates_args={})
def joint_active_nvfp4_parts(sequence: Tensor, packed_modulations: Tensor) -> tuple[Tensor, Tensor]:
    qdata = torch.empty((_BATCH * _TOKENS, _HIDDEN // 2), dtype=torch.uint8, device=sequence.device)
    scale = torch.empty((50, _HIDDEN_RK, 32, 4, 4), dtype=torch.float8_e4m3fn, device=sequence.device)
    stats = torch.empty((_BATCH * _TOKENS, 2), dtype=torch.float32, device=sequence.device)
    wrap_triton(_joint_stats_kernel)[(_BATCH * _TOKENS,)](
        sequence,
        stats,
        block_k=4096,
        num_warps=4,
        maxnreg=160,
    )
    tile_k = 512
    wrap_triton(_joint_apply_quant_kernel)[
        (_BATCH * _TOKENS, triton.cdiv(_HIDDEN, tile_k))
    ](
        sequence,
        packed_modulations,
        stats,
        qdata,
        scale,
        tile_k=tile_k,
        num_warps=4,
    )
    return qdata, scale


@triton_op("flux_action::swiglu_nvfp4_parts", mutates_args={})
def swiglu_nvfp4_parts(packed: Tensor) -> tuple[Tensor, Tensor]:
    rows = packed.shape[0]
    qdata = torch.empty((rows, _MLP // 2), dtype=torch.uint8, device=packed.device)
    scale = torch.empty(
        (triton.cdiv(rows, 128), _MLP_RK, 32, 4, 4),
        dtype=torch.float8_e4m3fn,
        device=packed.device,
    )
    tile_k = 1024
    wrap_triton(_swiglu_nvfp4_kernel)[(rows, triton.cdiv(_MLP, tile_k))](
        packed,
        qdata,
        scale,
        tile_k=tile_k,
        num_warps=4,
    )
    return qdata, scale


def joint_active_nvfp4(sequence: Tensor, packed_modulations: Tensor) -> tuple[Tensor, Tensor]:
    qdata, scale = joint_active_nvfp4_parts(sequence, packed_modulations)
    return qdata.view(torch.float4_e2m1fn_x2), scale


def swiglu_nvfp4(packed: Tensor) -> tuple[Tensor, Tensor]:
    qdata, scale = swiglu_nvfp4_parts(packed)
    return qdata.view(torch.float4_e2m1fn_x2), scale


__all__ = ["joint_active_nvfp4", "swiglu_nvfp4"]
