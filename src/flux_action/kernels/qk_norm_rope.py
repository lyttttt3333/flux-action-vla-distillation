"""Single-load packed Q/K RMSNorm and interleaved RoPE for DROID joint blocks."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_HEADS = 24
_HEAD_DIM = 128
_HIDDEN = _HEADS * _HEAD_DIM
_PACKED = 27648
_JOINT_ROWS = 6400
_JOINT_BATCH = 2
_JOINT_TOKENS = 3200
_JOINT_VALID = 3173
_BLOCK_M = 4


@triton.jit
def _qk_norm_rope_packed_kernel(
    packed_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_ptr,
    q_out_ptr,
    k_out_ptr,
    total_vectors: tl.constexpr,
    tokens: tl.constexpr,
    valid: tl.constexpr,
    block_m: tl.constexpr,
):
    """Keep four packed Q/K head rows live across RMS reduction and RoPE."""
    vector = tl.program_id(0) * block_m + tl.arange(0, block_m)[:, None]
    column = tl.arange(0, 128)[None, :]
    vector_mask = vector < total_vectors
    batch_index = vector // (valid * 24)
    token_head = vector % (valid * 24)
    token_index = token_head // 24
    head_index = token_head % 24
    packed_row = batch_index * tokens + token_index
    packed_base = packed_row * 27648 + head_index * 128

    q = tl.load(packed_ptr + packed_base + column, mask=vector_mask).to(tl.float32)
    k = tl.load(packed_ptr + packed_base + 3072 + column, mask=vector_mask).to(tl.float32)
    q_inv_rms = tl.rsqrt(tl.sum(q * q, axis=1)[:, None] * (1.0 / 128.0) + 1.0e-6)
    k_inv_rms = tl.rsqrt(tl.sum(k * k, axis=1)[:, None] * (1.0 / 128.0) + 1.0e-6)
    q_weight = tl.load(q_weight_ptr + column).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + column).to(tl.float32)
    q_normalized = q * q_inv_rms * q_weight
    k_normalized = k * k_inv_rms * k_weight
    q_pairs = tl.reshape(q_normalized, (block_m, 64, 2))
    k_pairs = tl.reshape(k_normalized, (block_m, 64, 2))
    q_pair_even, q_pair_odd = tl.split(q_pairs)
    k_pair_even, k_pair_odd = tl.split(k_pairs)

    pair = tl.arange(0, 64)[None, :]
    rope_base = (batch_index * valid + token_index) * 128
    cosine = tl.load(rope_ptr + rope_base + 2 * pair, mask=vector_mask).to(tl.float32)
    sine = tl.load(rope_ptr + rope_base + 2 * pair + 1, mask=vector_mask).to(tl.float32)
    q_even = cosine * q_pair_even - sine * q_pair_odd
    q_odd = sine * q_pair_even + cosine * q_pair_odd
    k_even = cosine * k_pair_even - sine * k_pair_odd
    k_odd = sine * k_pair_even + cosine * k_pair_odd
    output_offset = vector * 128 + column
    tl.store(q_out_ptr + output_offset, tl.interleave(q_even, q_odd), mask=vector_mask)
    tl.store(k_out_ptr + output_offset, tl.interleave(k_even, k_odd), mask=vector_mask)


@triton_op("flux_action::qk_norm_rope_packed", mutates_args={})
def qk_norm_rope_packed(
    packed: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    rope: Tensor,
) -> tuple[Tensor, Tensor]:
    """Normalize and rotate Q/K directly from the fixed joint packed projection."""
    shape = (_JOINT_BATCH, _JOINT_VALID, _HEADS, _HEAD_DIM)
    q_out = torch.empty(shape, dtype=torch.bfloat16, device=packed.device)
    k_out = torch.empty_like(q_out)
    total_vectors = _JOINT_BATCH * _JOINT_VALID * _HEADS
    wrap_triton(_qk_norm_rope_packed_kernel)[(triton.cdiv(total_vectors, _BLOCK_M),)](
        packed,
        q_weight,
        k_weight,
        rope,
        q_out,
        k_out,
        total_vectors=total_vectors,
        tokens=_JOINT_TOKENS,
        valid=_JOINT_VALID,
        block_m=_BLOCK_M,
        num_warps=4,
    )
    return q_out, k_out


def validate_qk_norm_rope_inputs(
    packed: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    rope: Tensor,
) -> None:
    """Fail before launch if the fixed joint geometry or layout does not match."""
    assert packed.is_cuda and packed.dtype == torch.bfloat16
    assert packed.shape == (_JOINT_ROWS, _PACKED) and packed.is_contiguous()
    assert q_weight.is_cuda and q_weight.device == packed.device
    assert q_weight.shape == (_HEAD_DIM,) and q_weight.dtype == torch.bfloat16
    assert q_weight.is_contiguous()
    assert k_weight.is_cuda and k_weight.device == packed.device
    assert k_weight.shape == (_HEAD_DIM,) and k_weight.dtype == torch.bfloat16
    assert k_weight.is_contiguous()
    assert rope.is_cuda and rope.device == packed.device
    assert rope.shape == (_JOINT_BATCH, _JOINT_VALID, _HEAD_DIM // 2, 2)
    assert rope.dtype == torch.float32 and rope.is_contiguous()


__all__ = ["qk_norm_rope_packed", "validate_qk_norm_rope_inputs"]
