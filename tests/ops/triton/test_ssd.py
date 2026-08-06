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


def skip_if_insufficient_gpu_memory(device, required_gib):
    # Shared by the int32-overflow regression tests below, which all need a
    # wide (large stride(1)) tensor to trigger the bug -- skip rather than
    # OOM on GPUs that are otherwise perfectly capable of running the suite.
    total_memory = torch.cuda.get_device_properties(device).total_memory
    required_memory = required_gib * 1024**3
    if total_memory < required_memory:
        pytest.skip(f"GPU has {total_memory / 1024**3:.1f} GiB, need >= {required_gib} GiB")


def wide_noncontiguous_slices(device, seqlen, parent_width, shapes, batch=None):
    """Allocate ONE (batch, seqlen, parent_width) parent tensor -- or
    (seqlen, parent_width) if batch is None -- and return disjoint,
    non-contiguous slices of it, one per entry in `shapes` (each a tuple of
    trailing dims whose product is that slice's width). Every slice shares
    the same large stride(1) == parent_width regardless of which columns it
    takes, since slicing the last dim of a contiguous tensor doesn't change
    the stride of an earlier dim -- so this is used by every int32-overflow
    regression test below needing multiple wide, non-contiguous tensors, at
    the memory cost of ONE parent instead of one per tensor.
    """
    widths = [math.prod(shape) for shape in shapes]
    assert parent_width >= sum(widths)
    parent_shape = (seqlen, parent_width) if batch is None else (batch, seqlen, parent_width)
    parent = torch.randn(parent_shape, dtype=torch.bfloat16, device=device)
    slices = []
    offset = 0
    for width, shape in zip(widths, shapes):
        leading = () if batch is None else (batch,)
        slices.append(parent[..., offset:offset + width].view(*leading, seqlen, *shape))
        offset += width
    return slices


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
    skip_if_insufficient_gpu_memory(device, required_gib=6)  # ~5 GiB parent tensor plus allocator headroom

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
    # Regression test for the same 32-bit pointer-arithmetic overflow class
    # as test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_no_overflow above,
    # but hit through the full mamba_chunk_scan_combined path instead of
    # _chunk_cumsum_fwd/_chunk_cumsum_bwd directly.
    #
    # x, B, C, and z are ALL sliced non-contiguously out of a much wider
    # parent tensor here (not just x), exactly as e.g. transformers'
    # NemotronHMamba2Mixer slices x/B/C/dt/z out of a wide fused in_proj
    # output -- see https://github.com/triton-lang/triton/issues/1058. This
    # matters: mamba_chunk_scan_combined only forces a tensor .contiguous()
    # if *neither* its last dim nor its dim-1 has stride 1 (see the
    # `x.stride(-1) != 1 and x.stride(1) != 1` checks in
    # _mamba_chunk_scan_combined_fwd/_bwd), so a tensor sliced along its last
    # dim out of a wider parent keeps its large seqlen-stride all the way
    # into the kernels. A first version of this test only made `x`
    # non-contiguous and left B/C as small ordinary allocations -- that
    # version passed even with the ssd_bmm.py/ssd_chunk_scan.py/
    # ssd_combined.py fixes reverted, because B/C's own strides never got
    # large enough to overflow in the first place; it wasn't actually
    # exercising the bug in those kernels at all.
    #
    # Backward is also exercised (not just forward under no_grad), since
    # several of the fixed kernels -- _chunk_scan_chunk_state_bwd_dx_kernel
    # (ssd_combined.py), _chunk_scan_bwd_dz_kernel/_dstates_kernel/
    # _dc_kernel/_dcb_kernel/_ddAcs_stable_kernel (ssd_chunk_scan.py),
    # _bmm_chunk_bwd_kernel (ssd_bmm.py), _chunk_state_bwd_db_kernel
    # (ssd_chunk_state.py) -- are backward-only and never run under a
    # forward-only, no_grad call.
    #
    # isfinite alone is not a strong enough correctness check: an overflowed
    # pointer offset that wraps to another in-bounds address can produce
    # finite but silently *wrong* values rather than a crash or a NaN --
    # confirmed by manually reverting the _chunk_state_fwd_kernel cast and
    # observing this test keep passing under an isfinite-only check.
    #
    # ssd_chunk_scan_combined_ref (the pure PyTorch reference) is not usable
    # for a strong comparison at this scale: at ~900 chunks it produces NaN
    # gradients even for the fixed/correct kernel (it's noted in its own
    # docstring as "much less numerically stable"), and its
    # O(nchunks * nheads * chunk_size^2) intermediate OOMs at realistic head
    # counts. Instead, compare the *same* production kernel path against
    # itself: run it once on the non-contiguous wide-sliced tensors (which
    # can overflow) and once on an ordinary .contiguous() copy of the exact
    # same values (which cannot overflow). Since it's identical arithmetic
    # over identical values -- only the memory layout differs -- a correct
    # kernel should produce (near-)identical output and gradients; any real
    # divergence is the bug, not numerical-algorithm mismatch.
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=6)  # ~4 GiB shared parent tensor plus allocator headroom

    torch.manual_seed(0)
    batch = 1
    # Kept small: holding x/z/B/C plus their contiguous-equivalent copies and
    # both branches' autograd graphs at once (needed for the comparison
    # below) OOMs at realistic head counts, and the overflow this guards
    # against depends only on seqlen/chunk_size/stride, not head count.
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

    # IMPORTANT: .contiguous() must run BEFORE requires_grad_() so it makes
    # an independent, ordinarily-strided copy of the same values rather than
    # becoming part of x/z/B/C's autograd graph -- and requires_grad_() must
    # be called in place (not preceded by .clone()) on x/z/B/C themselves,
    # since .clone() silently returns a *contiguous* copy of a non-contiguous
    # input (confirmed empirically), which would defeat the whole point of
    # this test by never actually exercising a non-contiguous, large-stride
    # tensor in the kernel at all. dt/A/D need independent leaves too (not
    # the literal same tensor reused for both branches), since two separate
    # .backward() calls on the same leaf would accumulate rather than give
    # independently comparable gradients.
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


def test_chunk_state_varlen_noncontiguous_wide_view_no_overflow() -> None:
    # Regression test for the same 32-bit pointer-arithmetic overflow class
    # in _chunk_state_varlen_kernel: `pid_c * chunk_size * stride_x_seqlen`
    # (and the identical pattern for stride_b_seqlen), where
    # `pid_c = (end_idx - 1) // chunk_size` is derived from a *loaded*
    # cu_seqlens value rather than directly from tl.program_id(), but is
    # otherwise the same uncast-leading-operand chain as every other kernel
    # fixed in this PR. chunk_state_varlen is reachable in production via
    # mamba_chunk_scan_combined(..., cu_seqlens=..., return_varlen_states=True)
    # (varlen/packed-sequence inference), not exercised by
    # test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow
    # above since that test doesn't use cu_seqlens.
    #
    # As with the combined test above, compare against the same function
    # called on contiguous-equivalent copies of the same values rather than
    # a separately-implemented reference (see that test's comment for why).
    device = 'cuda'
    skip_if_insufficient_gpu_memory(device, required_gib=6)  # ~4 GiB shared parent tensor plus allocator headroom

    torch.manual_seed(0)
    nheads = 8
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    total_seqlen = nchunks * chunk_size
    # dtype=torch.int32 matters: a plain torch.tensor([...]) defaults to
    # int64, which makes `end_idx` (loaded from it) already 64-bit
    # regardless of the .to(tl.int64) cast on pid_c -- silently defeating
    # this exact test (confirmed empirically: reverting the cast produced
    # zero output difference with an int64 cu_seqlens, but crashed outright
    # once cu_seqlens was int32, which is what packed-sequence callers
    # commonly use, e.g. flash-attention's cu_seqlens convention).
    cu_seqlens = torch.tensor([0, total_seqlen], device=device, dtype=torch.int32)  # one big sequence -> big pid_c
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
