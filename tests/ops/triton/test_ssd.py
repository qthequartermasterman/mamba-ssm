import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.ssd_chunk_state import (
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
    chunk_state,
    chunk_state_varlen,
)
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    _mamba_chunk_scan_combined_fwd,
    _mamba_chunk_scan_combined_bwd,
)
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd, _state_passing_bwd

from overflow_test_utils import skip_if_insufficient_gpu_memory, wide_noncontiguous_slices


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
    # int32 overflow in _chunk_cumsum_fwd_kernel/_chunk_cumsum_bwd_kernel,
    # see TODO: LinkToFutureIssueInMamba. Needs dt as a non-contiguous view
    # into a much wider parent tensor (large stride(1)) to trigger.
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
    # Distinct overflow term in the same two kernels: `pid_b * stride_dt_batch`,
    # separate from the pid_c term above and only exercised at batch > 1 (the
    # test above uses batch=1, so pid_b is always 0). See TODO: LinkToFutureIssueInMamba.
    # batch=8 here (vs. the batch=4 real-world crash) trades a larger batch
    # for a narrower/shorter parent tensor at the same overflow margin, to
    # keep memory down -- see (batch - 1) * seqlen * parent_width below.
    #
    # isfinite() can't reliably catch this: a wrapped offset can land on
    # another in-bounds address and read finite-but-wrong values. Compare
    # against the same kernel run on a contiguous copy of the same values
    # instead.
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=6)

    torch.manual_seed(0)
    batch = 8
    seqlen = 8_192
    nheads = 128
    chunk_size = 128
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin
    assert parent_width >= nheads

    dt, = wide_noncontiguous_slices(device, seqlen, parent_width, [(nheads,)], batch=batch)
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


def test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow() -> None:
    # Same overflow class hit through the full mamba_chunk_scan_combined
    # path (see TODO: LinkToFutureIssueInMamba), covering ssd_chunk_scan.py,
    # ssd_combined.py, and ssd_bmm.py's kernels too. x, B, C, and z are all
    # sliced non-contiguously out of a shared wide parent tensor (not just
    # x). Backward is exercised too, since several of the fixed kernels
    # only run there, not under a forward-only no_grad call.
    #
    # Compares the same kernel run on wide-sliced vs. an ordinary
    # .contiguous() copy of identical values, rather than
    # ssd_chunk_scan_combined_ref (unusable here: NaNs at this chunk count
    # even for a correct kernel, per its own docstring, and OOMs at
    # realistic head counts). isfinite() alone isn't sufficient either --
    # a wrapped offset can land in-bounds and produce finite, wrong values.
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=6)

    torch.manual_seed(0)
    batch = 1
    # Small nheads: holding x/z/B/C, their contiguous copies, and both
    # autograd graphs at once OOMs at realistic head counts; the overflow
    # depends only on seqlen/chunk_size/stride, not head count.
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    seqlen = nchunks * chunk_size
    parent_width = 18_560  # same width that overflowed in the ssd_chunk_state.py bug

    # x, z, B, and C are ALL sliced non-contiguously out of the SAME wide
    # parent tensor here (disjoint column ranges, via wide_noncontiguous_slices),
    # rather than one wide parent per tensor -- one ~4 GiB allocation instead
    # of four.
    x, z, B, C = wide_noncontiguous_slices(
        device, seqlen, parent_width,
        [(nheads, headdim), (nheads, headdim), (ngroups, dstate), (ngroups, dstate)],
        batch=batch,
    )
    assert not x.is_contiguous()
    assert not z.is_contiguous()
    assert not B.is_contiguous()
    assert not C.is_contiguous()
    # (nchunks - 1) * chunk_size * stride(1) overflow check for B/C
    assert (nchunks - 1) * chunk_size * B.stride(1) > 2**31 - 1

    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 0.01
    D = torch.randn(nheads, headdim, dtype=torch.float32, device=device)

    # x/z/B/C: requires_grad_() must be called in place, not via .clone()
    # first -- .clone() silently makes a non-contiguous tensor contiguous,
    # which would defeat this test entirely.
    x_c, z_c, B_c, C_c = [t.contiguous() for t in (x, z, B, C)]
    dt_c, A_c, D_c = [t.clone() for t in (dt, A, D)]
    for t in (x, z, B, C, dt, A, D, x_c, z_c, B_c, C_c, dt_c, A_c, D_c):
        t.requires_grad_()
    assert not x.is_contiguous() and x_c.is_contiguous()
    assert not B.is_contiguous() and B_c.is_contiguous()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z)
    out_c = mamba_chunk_scan_combined(x_c, dt_c, A_c, B_c, C_c, chunk_size, D=D_c, z=z_c)
    torch.cuda.synchronize(device)

    assert out.shape == (batch, seqlen, nheads, headdim)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_c.float(), rtol=1e-4, atol=1e-4)

    out.sum().backward()
    out_c.sum().backward()
    torch.cuda.synchronize(device)
    grad_pairs = [("x", x, x_c), ("z", z, z_c), ("B", B, B_c), ("C", C, C_c),
                  ("dt", dt, dt_c), ("A", A, A_c), ("D", D, D_c)]
    for name, t, t_c in grad_pairs:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
        # Slightly looser than the forward check: A.grad and D.grad are
        # summed over all ~900 chunks, and summing in a different order
        # (contiguous vs. wide-sliced memory layout) causes tiny float
        # roundoff differences at the ~1e-3 level even for a correct kernel.
        torch.testing.assert_close(t.grad.float(), t_c.grad.float(), rtol=1e-3, atol=1e-3,
                                   msg=f"{name}.grad mismatch vs contiguous-equivalent")


def test_mamba_chunk_scan_combined_bwd_noncontiguous_dout_no_overflow() -> None:
    # int64 to avoid int32 overflow, see TODO: LinkToFutureIssueInMamba.
    # dout is the incoming gradient from autograd, so unlike x/dt/B/C (all
    # produced internally by our own ops), it can arrive non-contiguous
    # with an arbitrary, caller-controlled large stride -- e.g. sliced out
    # of a wider fused gradient tensor upstream.
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=6)

    torch.manual_seed(0)
    batch = 1
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906
    seqlen = nchunks * chunk_size
    parent_width = 18_560

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32, device=device)
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(nheads, headdim, dtype=torch.float32, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 0.01

    out, *_ = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D)

    dout_parent = torch.randn(batch, seqlen, parent_width, dtype=torch.float32, device=device)
    dout = dout_parent[:, :, -nheads * headdim:].view(batch, seqlen, nheads, headdim)
    assert not dout.is_contiguous()
    assert (nchunks - 1) * chunk_size * dout.stride(1) > 2**31 - 1
    dout_c = dout.contiguous()

    def run(dout_):
        return _mamba_chunk_scan_combined_bwd(dout_, x, dt, A, B, C, out, chunk_size, D=D)

    dx, ddt, dA, dB, dC, dD, *_ = run(dout)
    dx_ref, ddt_ref, dA_ref, dB_ref, dC_ref, dD_ref, *_ = run(dout_c)
    torch.cuda.synchronize(device)
    # dA/dD are summed over all ~900 chunks; summing in a different order
    # (contiguous vs. wide-sliced memory layout) causes tiny float roundoff
    # differences at the ~1e-3 level even for a correct kernel -- same as
    # the combined test above.
    tols = {"dA": (1e-3, 1e-3), "dD": (1e-3, 1e-3)}
    for name, g, g_ref in [("dx", dx, dx_ref), ("ddt", ddt, ddt_ref), ("dA", dA, dA_ref),
                           ("dB", dB, dB_ref), ("dC", dC, dC_ref), ("dD", dD, dD_ref)]:
        assert torch.isfinite(g).all(), f"{name} has non-finite values"
        rtol, atol = tols.get(name, (None, None))
        torch.testing.assert_close(g, g_ref, rtol=rtol, atol=atol, msg=f"{name} mismatch vs contiguous dout")


def test_state_passing_fwd_bwd_noncontiguous_dA_chunk_cumsum_no_overflow() -> None:
    # int64 to avoid int32 overflow, see TODO: LinkToFutureIssueInMamba.
    # dA_chunk_cumsum is *always* non-contiguous in production
    # (ssd_combined.py slices dA_cumsum[:, :, :, -1]), but only reaches
    # overflow-relevant scale once nheads * nchunks is large. Reproduced
    # directly here via a padded parent tensor -- states/dout stay small
    # and cheap; only the padding dim (not nheads/nchunks/dim) needs to be
    # huge to hit the threshold.
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=12)

    torch.manual_seed(0)
    batch = 1
    nheads = 128
    nchunks = 64
    dim = 64
    width = 300_000  # padding only, not a real chunk_size

    parent = torch.randn(batch, nheads, nchunks, width, dtype=torch.float32, device=device)
    dA_chunk_cumsum = parent[:, :, :, -1]
    assert not dA_chunk_cumsum.is_contiguous()
    assert (nheads - 1) * dA_chunk_cumsum.stride(1) > 2**31 - 1
    dA_chunk_cumsum_c = dA_chunk_cumsum.contiguous()

    states = torch.randn(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)
    dout = torch.randn(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    out_ref, final_states_ref = _state_passing_fwd(states, dA_chunk_cumsum_c)
    torch.cuda.synchronize(device)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(final_states, final_states_ref)

    dstates, ddA, _ = _state_passing_bwd(states, dA_chunk_cumsum, dout, has_initial_states=False)
    dstates_ref, ddA_ref, _ = _state_passing_bwd(states, dA_chunk_cumsum_c, dout, has_initial_states=False)
    torch.cuda.synchronize(device)
    assert torch.isfinite(dstates).all()
    torch.testing.assert_close(dstates, dstates_ref)
    torch.testing.assert_close(ddA, ddA_ref)


def test_chunk_state_varlen_noncontiguous_wide_view_no_overflow() -> None:
    # Same overflow class in _chunk_state_varlen_kernel, see
    # TODO: LinkToFutureIssueInMamba. pid_c here comes from a loaded
    # cu_seqlens value rather than tl.program_id() directly.
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=6)

    torch.manual_seed(0)
    nheads = 8
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    total_seqlen = nchunks * chunk_size
    # dtype=torch.int32 matters: a plain torch.tensor([...]) defaults to
    # int64, which would make end_idx already 64-bit regardless of the
    # kernel's own cast, silently defeating this test.
    cu_seqlens = torch.tensor([0, total_seqlen], device=device, dtype=torch.int32)
    parent_width = 18_560

    # x and B share ONE wide parent tensor (disjoint column ranges) instead
    # of one each -- see wide_noncontiguous_slices' docstring.
    x, B = wide_noncontiguous_slices(
        device, total_seqlen, parent_width, [(nheads, headdim), (ngroups, dstate)],
    )
    assert not x.is_contiguous()
    assert not B.is_contiguous()
    assert (nchunks - 1) * chunk_size * x.stride(0) > 2**31 - 1

    x_c, B_c = x.contiguous(), B.contiguous()

    dt = F.softplus(torch.randn(total_seqlen, nheads, device=device, dtype=torch.float32) - 4)
    A = -0.1 * torch.rand(nheads, device=device)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum)
    chunk_states_c = _chunk_state_fwd(B_c.unsqueeze(0), x_c.unsqueeze(0), dt_rounded, dA_cumsum)
    dA_cumsum, dt_rounded = dA_cumsum.squeeze(0), dt_rounded.squeeze(0)
    chunk_states, chunk_states_c = chunk_states.squeeze(0), chunk_states_c.squeeze(0)

    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    out_c = chunk_state_varlen(B_c, x_c, dt_rounded, dA_cumsum, cu_seqlens, chunk_states_c)
    torch.cuda.synchronize(device)

    assert out.shape == (1, nheads, headdim, dstate)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_c.float(), rtol=1e-4, atol=1e-4)
