"""Single-pass producer/rowwise-FP8 fusion for the fixed DROID geometry."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_HIDDEN = 3072
_MLP = 9216
_PACKED = 3 * _HIDDEN + 2 * _MLP


@triton.jit
def _swiglu_fp8r_kernel(
    packed_ptr,
    quantized_ptr,
    scale_ptr,
    BLOCK_K: tl.constexpr,
):
    """Keep one produced SwiGLU row live across its whole-row amax."""
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    mask = columns < 9216
    base = row * 27648
    gate = tl.load(packed_ptr + base + 9216 + columns, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(packed_ptr + base + 18432 + columns, mask=mask, other=0.0).to(tl.float32)
    active = gate * tl.sigmoid(gate) * up
    row_amax = tl.max(tl.abs(active), axis=0)
    row_scale = tl.maximum(row_amax * (1.0 / 448.0), 1.0e-12)
    quantized = tl.minimum(tl.maximum(active / row_scale, -448.0), 448.0)
    tl.store(
        quantized_ptr + row * 9216 + columns,
        quantized.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(scale_ptr + row, row_scale)


@triton_op("flux_action::swiglu_fp8r", mutates_args={})
def swiglu_fp8r(packed: Tensor) -> tuple[Tensor, Tensor]:
    """Apply SwiGLU and exact per-row E4M3 quantization without a BF16 intermediate."""
    rows = packed.shape[0]
    quantized = torch.empty((rows, _MLP), dtype=torch.float8_e4m3fn, device=packed.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=packed.device)
    wrap_triton(_swiglu_fp8r_kernel)[(rows,)](
        packed,
        quantized,
        scale,
        BLOCK_K=16384,
        num_warps=4,
    )
    return quantized, scale


def validate_swiglu_fp8r_input(packed: Tensor) -> None:
    """Fail early when a non-DROID tensor reaches the shape-specialized kernel."""
    assert packed.is_cuda and packed.dtype == torch.bfloat16, "expected CUDA BF16 W13 output"
    assert packed.ndim == 2 and packed.shape[1] == _PACKED, "invalid packed W13 geometry"
    assert packed.is_contiguous(), "packed W13 output must be contiguous"


__all__ = ["swiglu_fp8r", "validate_swiglu_fp8r_input"]
