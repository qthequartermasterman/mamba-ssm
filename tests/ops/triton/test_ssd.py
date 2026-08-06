import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state, chunk_state_ref
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd, _chunk_state_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state_varlen
from mamba_ssm.ops.triton.ssd_state_passing import state_passing, state_passing_ref
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import chunk_scan, chunk_scan_ref
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_chunk_scan, ssd_chunk_scan_combined_ref, ssd_selective_scan
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined, mamba_split_conv1d_scan_ref


def detach_clone(*args):
    return tuple([arg.detach().clone().requires_grad_() if arg is not None else None for arg in args])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('ngroups', [1, 2, 8, "max"])
# @pytest.mark.parametrize('ngroups', [1])
@pytest.mark.parametrize('chunk_size', [64, 128])
# @pytest.mark.parametrize('chunk_size', [128])
def test_chunk_state_varlen(chunk_size, ngroups, dtype):
    device = 'cuda'
    rtol, atol = (1e-2, 3e-3) if dtype != torch.bfloat16 else (1e-2, 6e-3)
    # set seed
    torch.random.manual_seed(chunk_size + (ngroups if ngroups != "max" else 64))
    batch = 300
    seqlens = torch.randint(1, 200, (batch,), device=device)
    # batch = 3
    # seqlens = torch.tensor([201, 56, 5], device=device)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0))
    total_seqlen = seqlens.sum().item()
    seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=device) for i, s in enumerate(seqlens)], dim=0).unsqueeze(0)
    dim = 4096
    # dim = 64
    headdim = 64
    # dim = 32
    dstate = 32
    assert dim % headdim == 0
    nheads = dim // headdim
    if ngroups == "max":
        ngroups = nheads
    assert nheads % ngroups == 0
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    x = torch.randn(total_seqlen, nheads, headdim, dtype=dtype, device=device)
    A = -0.1 * (torch.rand(nheads, device=device))
    dt = F.softplus(torch.randn(total_seqlen, nheads, device=device, dtype=torch.float32) - 4)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum, seq_idx=seq_idx)
    chunk_states, _ = _state_passing_fwd(rearrange(chunk_states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                         seq_idx=seq_idx, chunk_size=chunk_size)
    chunk_states = rearrange(chunk_states, "... (p n) -> ... p n", n=dstate)
    chunk_states = chunk_states.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)
    dt_rounded = dt_rounded.squeeze(0)
    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    out_ref = []
    for b in range(batch):
        x_s = x[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        B_s = B[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        dt_s = dt[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        dA_cumsum_s, dt_rounded_s = _chunk_cumsum_fwd(dt_s, A, chunk_size)
        states = chunk_state(B_s, x_s, dt_rounded_s, dA_cumsum_s)
        _, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum_s[:, :, :, -1],
                                             chunk_size=chunk_size)
        final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)
        out_ref.append(final_states)
    out_ref = torch.cat(out_ref, dim=0)
    print(f"Max diff = {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


def test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_no_overflow() -> None:
    # Regression test for a 32-bit pointer-arithmetic overflow shared by
    # _chunk_cumsum_fwd_kernel and _chunk_cumsum_bwd_kernel:
    # `pid_c * chunk_size * stride_dt_seqlen` was computed in 32-bit and
    # silently wrapped once it exceeded 2**31 - 1, corrupting the pointer
    # offset into `dt` and causing a CUDA illegal memory access. This only
    # shows up when `dt` is a non-contiguous view into a much wider parent
    # tensor (large stride(1)), not for an ordinarily contiguous `dt` -- see
    # https://github.com/triton-lang/triton/issues/1058. Both kernels have
    # the identical uncast pattern and received the identical fix, so both
    # are exercised here.
    device = 'cuda'
    torch.manual_seed(0)
    seqlen = 115_866
    nheads = 128
    chunk_size = 128
    parent_width = 18_560  # wide enough that (nchunks - 1) * chunk_size * stride(1) overflows int32

    A = -torch.exp(torch.randn(nheads, dtype=torch.float32, device=device))
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device)

    parent = torch.zeros((1, seqlen, parent_width), dtype=torch.bfloat16, device=device)
    dt = parent[:, :, -nheads:]
    assert not dt.is_contiguous()

    with torch.no_grad():
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
        ddA = torch.randn_like(dA_cumsum)
        ddt_out = torch.randn_like(dt_out)
        ddt, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    nchunks = math.ceil(seqlen / chunk_size)
    assert dA_cumsum.shape == (1, nheads, nchunks, chunk_size)
    assert dt_out.shape == (1, nheads, nchunks, chunk_size)
    assert torch.isfinite(dA_cumsum).all()
    assert torch.isfinite(dt_out).all()

    assert ddt.shape == dt.shape
    assert dA.shape == (nheads,)
    assert torch.isfinite(ddt).all()
    assert torch.isfinite(dA).all()
    assert torch.isfinite(ddt_bias).all()


def test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_batch_axis_no_overflow() -> None:
    # Regression test for a distinct 32-bit pointer-arithmetic overflow in
    # the same two kernels as test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_no_overflow
    # above: `pid_b * stride_dt_batch` is a separate pointer-offset
    # multiplication from `pid_c * chunk_size * stride_dt_seqlen`, so casting
    # pid_c alone does not protect it. The test above always uses batch=1,
    # so pid_b is always 0 and this term is never exercised regardless of
    # how large stride_dt_batch is -- this test uses batch>1 to cover it.
    # stride_dt_batch is large exactly when dt is a non-contiguous slice of a
    # wide fused projection (e.g. transformers' NemotronHMamba2Mixer
    # splitting a fused in_proj output across batch elements); the original
    # reported crash was batch=4, seqlen=40_960, parent_width=35_072, which
    # requires allocating a ~10.7 GiB parent tensor and can OOM on GPUs with
    # 8-12 GiB. The overflow trigger is (batch - 1) * seqlen * parent_width,
    # so a larger batch needs a smaller seqlen/parent_width product to clear
    # the same threshold -- batch=8 with a much narrower/shorter parent
    # tensor still overflows with margin, at roughly half the memory.
    #
    # isfinite alone cannot reliably catch this (see the comparable
    # discussion for test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow):
    # a wrapped offset can land on another in-bounds address and produce
    # finite but silently wrong values. Compare against the same kernels run
    # on a contiguous copy of the identical values instead.
    device = 'cuda'
    total_memory = torch.cuda.get_device_properties(device).total_memory
    required_memory = 6 * 1024**3  # ~5 GiB parent tensor plus headroom for allocator overhead
    if total_memory < required_memory:
        pytest.skip(f"GPU has {total_memory / 1024**3:.1f} GiB, need >= {required_memory / 1024**3:.0f} GiB")

    torch.manual_seed(0)
    batch = 8
    seqlen = 8_192
    nheads = 128
    chunk_size = 128
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin
    assert parent_width >= nheads
    assert (batch - 1) * seqlen * parent_width > 2**31 - 1, "test parameters too small to overflow the batch-axis term"

    parent = torch.randn((batch, seqlen, parent_width), dtype=torch.bfloat16, device=device)
    dt = parent[:, :, -nheads:]
    assert not dt.is_contiguous()
    assert dt.stride(0) * (batch - 1) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"
    dt_c = dt.contiguous()

    A = -torch.exp(torch.randn(nheads, dtype=torch.float32, device=device))
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device)

    with torch.no_grad():
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
        dA_cumsum_c, dt_out_c = _chunk_cumsum_fwd(dt_c, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    nchunks = math.ceil(seqlen / chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert torch.isfinite(dA_cumsum).all()
    torch.testing.assert_close(dA_cumsum, dA_cumsum_c, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dt_out, dt_out_c, rtol=1e-4, atol=1e-4)

    with torch.no_grad():
        ddA = torch.randn_like(dA_cumsum)
        ddt_out = torch.randn_like(dt_out)
        ddt, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=dt_bias, dt_softplus=True)
        ddt_c, dA_c, ddt_bias_c = _chunk_cumsum_bwd(ddA, ddt_out, dt_c, A, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    assert ddt.shape == dt.shape
    assert torch.isfinite(ddt).all()
    torch.testing.assert_close(ddt, ddt_c, rtol=1e-4, atol=1e-4)
    # Slightly looser: dA is summed over all chunks, and summing in a
    # different order (contiguous vs. wide-sliced memory layout) causes tiny
    # float roundoff differences at the ~1e-3 level even for a correct kernel.
    torch.testing.assert_close(dA, dA_c, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ddt_bias, ddt_bias_c, rtol=1e-3, atol=1e-3)
