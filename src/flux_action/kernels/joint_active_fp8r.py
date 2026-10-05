"""Fused joint LayerNorm, stream modulation, and rowwise FP8 quantization."""

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
_BLOCK_K = 4096


@triton.jit
def _pack_joint_modulations_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    in4_ptr,
    in5_ptr,
    in6_ptr,
    in7_ptr,
    in8_ptr,
    in9_ptr,
    out_ptr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < 61440
    source = index // 6144
    offset = index % 6144
    value = (
        tl.load(in0_ptr + offset, mask=mask & (source == 0), other=0.0)
        + tl.load(in1_ptr + offset, mask=mask & (source == 1), other=0.0)
        + tl.load(in2_ptr + offset, mask=mask & (source == 2), other=0.0)
        + tl.load(in3_ptr + offset, mask=mask & (source == 3), other=0.0)
        + tl.load(in4_ptr + offset, mask=mask & (source == 4), other=0.0)
        + tl.load(in5_ptr + offset, mask=mask & (source == 5), other=0.0)
        + tl.load(in6_ptr + offset, mask=mask & (source == 6), other=0.0)
        + tl.load(in7_ptr + offset, mask=mask & (source == 7), other=0.0)
        + tl.load(in8_ptr + offset, mask=mask & (source == 8), other=0.0)
        + tl.load(in9_ptr + offset, mask=mask & (source == 9), other=0.0)
    )
    tl.store(out_ptr + index, value, mask=mask)


@triton.jit
def _joint_active_fp8r_kernel(
    sequence_ptr,
    mods_ptr,
    quantized_ptr,
    row_scale_ptr,
    block_k: tl.constexpr,
):
    """Keep each joint row live across LayerNorm, modulation, and amax."""
    row = tl.program_id(0)
    column = tl.arange(0, block_k)
    token = row % 3200
    batch = row // 3200
    column_mask = column < 3072
    valid_mask = column_mask & (token < 3173)
    value = tl.load(
        sequence_ptr + row * 3072 + column,
        mask=valid_mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(value, axis=0) * (1.0 / 3072.0)
    centered = value - mean
    variance = tl.sum(centered * centered, axis=0) * (1.0 / 3072.0)
    normalized = centered * tl.rsqrt(variance + 1.0e-6)

    stream = tl.where(
        token < 80,
        0,
        tl.where(
            token < 2800,
            1,
            tl.where(token < 3140, 2, tl.where(token < 3172, 3, 4)),
        ),
    )
    modulation_offset = stream * 12288 + batch * 3072 + column
    shift = tl.load(mods_ptr + modulation_offset, mask=valid_mask).to(tl.float32)
    scale = tl.load(mods_ptr + modulation_offset + 6144, mask=valid_mask).to(tl.float32)

    # The Torch path materializes active as BF16 before FP8 quantization.
    active = ((1.0 + scale) * normalized + shift).to(tl.bfloat16).to(tl.float32)
    active = tl.where(valid_mask, active, 0.0)
    row_amax = tl.max(tl.abs(active), axis=0)
    row_scale = tl.maximum(row_amax * (1.0 / 448.0), 1.0e-12)
    quantized = tl.minimum(tl.maximum(active / row_scale, -448.0), 448.0)
    tl.store(
        quantized_ptr + row * 3072 + column,
        quantized.to(tl.float8e4nv),
        mask=column_mask,
    )
    tl.store(row_scale_ptr + row, row_scale)


@triton_op("flux_action::joint_active_fp8r", mutates_args={})
def joint_active_fp8r(
    sequence: Tensor,
    packed_modulations: Tensor,
) -> tuple[Tensor, Tensor]:
    """Produce fixed-geometry joint FP8 inputs without a BF16 active tensor."""
    quantized = torch.empty(
        (_BATCH * _TOKENS, _HIDDEN),
        dtype=torch.float8_e4m3fn,
        device=sequence.device,
    )
    row_scale = torch.empty((_BATCH * _TOKENS,), dtype=torch.float32, device=sequence.device)
    wrap_triton(_joint_active_fp8r_kernel)[(_BATCH * _TOKENS,)](
        sequence,
        packed_modulations,
        quantized,
        row_scale,
        block_k=_BLOCK_K,
        num_warps=4,
        maxnreg=144,
    )
    return quantized, row_scale


@triton_op("flux_action::pack_joint_modulations", mutates_args={})
def pack_joint_modulations(
    value0: Tensor,
    value1: Tensor,
    value2: Tensor,
    value3: Tensor,
    value4: Tensor,
    value5: Tensor,
    value6: Tensor,
    value7: Tensor,
    value8: Tensor,
    value9: Tensor,
) -> Tensor:
    """Materialize the five joint shift/scale pairs once per NFE."""
    packed = torch.empty((5, 2, _BATCH, _HIDDEN), dtype=torch.bfloat16, device=value0.device)
    wrap_triton(_pack_joint_modulations_kernel)[(triton.cdiv(packed.numel(), 256),)](
        value0,
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
        value7,
        value8,
        value9,
        packed,
        block=256,
        num_warps=4,
    )
    return packed


def validate_joint_active_fp8r_inputs(
    sequence: Tensor,
    packed_modulations: Tensor,
) -> None:
    """Fail before launch if the fixed DROID joint geometry does not match."""
    assert sequence.is_cuda and sequence.dtype == torch.bfloat16
    assert sequence.shape == (_BATCH, _TOKENS, _HIDDEN) and sequence.is_contiguous()
    assert packed_modulations.is_cuda and packed_modulations.device == sequence.device
    assert packed_modulations.dtype == torch.bfloat16
    assert packed_modulations.shape == (5, 2, _BATCH, _HIDDEN)
    assert packed_modulations.is_contiguous()


__all__ = [
    "joint_active_fp8r",
    "pack_joint_modulations",
    "validate_joint_active_fp8r_inputs",
]
