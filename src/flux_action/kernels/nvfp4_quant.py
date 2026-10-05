"""Fixed-geometry single-pass NVFP4 activation quantization for SM100."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.jit
def _pack_e2m1_pair(low_value, high_value, pair_scale):
    low_scaled = low_value / pair_scale
    high_scaled = high_value / pair_scale
    # Quack's canonical NVFP4 quantizer resolves exact E2M1 midpoints toward
    # the larger magnitude, while Blackwell's direct RN convert resolves some
    # midpoint cases toward the smaller magnitude. BF16 values divided by an
    # E4M3 block scale hit those ties frequently. Move by two FP32 ulps at the
    # largest finite E2M1 value so the hardware conversion preserves Quack's
    # public quantization semantics without changing ordinary non-tie values.
    low_abs = tl.abs(low_scaled)
    high_abs = tl.abs(high_scaled)
    low_code = (
        (low_abs > 0.25).to(tl.uint8)
        + (low_abs >= 0.75).to(tl.uint8)
        + (low_abs > 1.25).to(tl.uint8)
        + (low_abs >= 1.75).to(tl.uint8)
        + (low_abs > 2.5).to(tl.uint8)
        + (low_abs >= 3.5).to(tl.uint8)
        + (low_abs > 5.0).to(tl.uint8)
    )
    high_code = (
        (high_abs > 0.25).to(tl.uint8)
        + (high_abs >= 0.75).to(tl.uint8)
        + (high_abs > 1.25).to(tl.uint8)
        + (high_abs >= 1.75).to(tl.uint8)
        + (high_abs > 2.5).to(tl.uint8)
        + (high_abs >= 3.5).to(tl.uint8)
        + (high_abs > 5.0).to(tl.uint8)
    )
    low_code |= tl.where(low_scaled < 0.0, 8, 0).to(tl.uint8)
    high_code |= tl.where(high_scaled < 0.0, 8, 0).to(tl.uint8)
    return low_code | (high_code << 4)


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
def _nvfp4_quant_kernel(
    value_ptr,
    qdata_ptr,
    blocked_scale_ptr,
    rows: tl.constexpr,
    k: tl.constexpr,
    sf_k: tl.constexpr,
    rk: tl.constexpr,
    block_k: tl.constexpr,
):
    """Quantize one row and write scales directly in the SM100 blocked layout."""
    row = tl.program_id(0)
    columns = tl.arange(0, block_k)
    valid = columns < k
    values = tl.load(value_ptr + row * k + columns, mask=valid, other=0.0).to(tl.float32)

    # block_k is a power of two, so Triton can keep the per-16 reduction and
    # pair packing in registers without a temporary scale tensor.
    blocks = values.reshape((block_k // 16, 16))
    block_amax = tl.max(tl.abs(blocks), axis=1)
    block_scale = tl.minimum(tl.maximum(block_amax * (1.0 / 6.0), 0.015625), 448.0)
    block_scale_fp8 = block_scale.to(tl.float8e4nv)
    rounded_block_scale = _e4m3_positive_to_fp32(block_scale_fp8)

    # Load pairs after the reduction so each pair can gather its rounded block
    # scale directly. This also preserves the low/even, high/odd nibble order.
    pair_columns = tl.arange(0, block_k // 2)
    pair_valid = 2 * pair_columns < k
    low_value = tl.load(
        value_ptr + row * k + 2 * pair_columns,
        mask=pair_valid,
        other=0.0,
    ).to(tl.float32)
    high_value = tl.load(
        value_ptr + row * k + 2 * pair_columns + 1,
        mask=pair_valid,
        other=0.0,
    ).to(tl.float32)
    pair_scale = (
        rounded_block_scale[:, None]
        + tl.zeros((block_k // 16, 8), dtype=tl.float32)
    ).reshape((block_k // 2,))
    packed = _pack_e2m1_pair(low_value, high_value, pair_scale)
    tl.store(
        qdata_ptr + row * (k // 2) + pair_columns,
        packed,
        mask=pair_columns < k // 2,
    )

    scale_block = tl.arange(0, block_k // 16)
    # (row // 128, scale_block // 4, row % 32,
    #  (row // 32) % 4, scale_block % 4), contiguous inner strides (16,4,1).
    scale_offset = (
        (((row // 128) * rk + scale_block // 4) * 32 + row % 32) * 4
        + (row // 32) % 4
    ) * 4 + scale_block % 4
    tl.store(
        blocked_scale_ptr + scale_offset,
        block_scale_fp8,
        mask=scale_block < sf_k,
    )


@triton.jit
def _nvfp4_quant_tiled_kernel(
    value_ptr,
    qdata_ptr,
    blocked_scale_ptr,
    k: tl.constexpr,
    rk: tl.constexpr,
    tile_k: tl.constexpr,
):
    """Lower-register variant: one row by 256 columns per program."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    tile_start = tile * tile_k
    columns = tile_start + tl.arange(0, tile_k)
    values = tl.load(value_ptr + row * k + columns).to(tl.float32)
    blocks = values.reshape((tile_k // 16, 16))
    block_amax = tl.max(tl.abs(blocks), axis=1)
    block_scale = tl.minimum(tl.maximum(block_amax * (1.0 / 6.0), 0.015625), 448.0)
    block_scale_fp8 = block_scale.to(tl.float8e4nv)
    rounded_block_scale = _e4m3_positive_to_fp32(block_scale_fp8)

    pair_columns = tl.arange(0, tile_k // 2)
    low_value = tl.load(value_ptr + row * k + tile_start + 2 * pair_columns).to(tl.float32)
    high_value = tl.load(value_ptr + row * k + tile_start + 2 * pair_columns + 1).to(tl.float32)
    pair_scale = (
        rounded_block_scale[:, None] + tl.zeros((tile_k // 16, 8), dtype=tl.float32)
    ).reshape((tile_k // 2,))
    packed = _pack_e2m1_pair(low_value, high_value, pair_scale)
    tl.store(
        qdata_ptr + row * (k // 2) + tile_start // 2 + pair_columns,
        packed,
    )

    local_scale = tl.arange(0, tile_k // 16)
    scale_block = tile * (tile_k // 16) + local_scale
    scale_offset = (
        (((row // 128) * rk + scale_block // 4) * 32 + row % 32) * 4
        + (row // 32) % 4
    ) * 4 + scale_block % 4
    tl.store(blocked_scale_ptr + scale_offset, block_scale_fp8)


@triton_op("flux_action::nvfp4_quant_parts", mutates_args={})
def nvfp4_quant_parts(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return packed E2M1 bytes and canonical blocked E4M3 scale factors."""
    rows, k = value.shape
    assert rows in (128, 2816, 6400) and k in (3072, 9216)
    sf_k = k // 16
    rk = triton.cdiv(sf_k, 4)
    tile_k = 256
    qdata = torch.empty((rows, k // 2), dtype=torch.uint8, device=value.device)
    scale = torch.empty(
        (triton.cdiv(rows, 128), rk, 32, 4, 4),
        dtype=torch.float8_e4m3fn,
        device=value.device,
    )
    wrap_triton(_nvfp4_quant_tiled_kernel)[(rows, triton.cdiv(k, tile_k))](
        value,
        qdata,
        scale,
        k=k,
        rk=rk,
        tile_k=tile_k,
        num_warps=4,
    )
    return qdata, scale


def nvfp4_quant(value: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a supported BF16 matrix and expose torch's packed FP4 dtype."""
    validate_nvfp4_quant_input(value)
    qdata, scale = nvfp4_quant_parts(value)
    return qdata.view(torch.float4_e2m1fn_x2), scale


def validate_nvfp4_quant_input(value: Tensor) -> None:
    assert value.is_cuda and value.dtype == torch.bfloat16 and value.ndim == 2
    assert value.shape[0] in (128, 2816, 6400)
    assert value.shape[1] in (3072, 9216)
    assert value.is_contiguous()


__all__ = ["nvfp4_quant", "validate_nvfp4_quant_input"]
