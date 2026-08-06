import math

import torch

from mamba_ssm.ops.triton.layer_norm import rms_norm_fn


def test_rms_norm_large_row_count_no_overflow() -> None:
    # Regression test for a 32-bit pointer-arithmetic overflow in
    # _layer_norm_fwd_1pass_kernel and _layer_norm_bwd_kernel in
    # layer_norm.py: `row * stride_x_row` (fwd) / `row_start * stride_x_row`
    # (bwd) was computed in 32-bit and silently wrapped once it exceeded
    # 2**31 - 1, corrupting the pointer offset into `x` and causing a CUDA
    # illegal memory access. Like the layernorm_gated.py overflow, this
    # doesn't need a non-contiguous view -- an ordinarily contiguous (M, N)
    # input is enough once M * N is large, since `row` ranges over all M
    # flattened rows and stride_x_row == N. See
    # https://github.com/triton-lang/triton/issues/1058.
    #
    # layer_norm.py has no other test file exercising it at all (unlike
    # layernorm_gated.py, which is a related but distinct module with its
    # own kernels and its own cast fix).
    #
    # The forward kernel's `row` ranges up to M - 1 directly, so M * N just
    # over 2**31 - 1 is enough there. The backward kernel instead launches
    # only sm_count programs, each handling `rows_per_program =
    # ceil(M / sm_count)` rows via `row_start = row_block_id *
    # rows_per_program` -- the largest row_start is only
    # (sm_count - 1) * rows_per_program, which is *less* than M - 1 and,
    # depending on the GPU's SM count, can fall short of the int32 boundary
    # even when M * N alone comfortably exceeds it. A first version of this
    # test picked M/N with only the forward case's margin in mind and
    # missed the backward kernel's cast entirely as a result. Size N with
    # enough headroom that the backward case overflows too, and assert that
    # explicitly using the actual device's SM count rather than assuming it.
    device = 'cuda'
    torch.manual_seed(0)
    N = 32_768  # hard cap: layer_norm.py raises if group_size/N exceeds 64KB / dtype_size
    M = 80_000  # (sm_count - 1) * ceil(M / sm_count) * N still clears 2**31 - 1 with margin (asserted below)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    rows_per_program = math.ceil(M / sm_count)
    bwd_row_start_max = (sm_count - 1) * rows_per_program
    assert bwd_row_start_max * N > 2**31 - 1, (
        f"test parameters too small to overflow the backward kernel on this GPU "
        f"(sm_count={sm_count}): increase N"
    )

    x = torch.randn(M, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    weight = torch.randn(N, dtype=torch.float32, device=device, requires_grad=True)

    out = rms_norm_fn(x, weight, bias=None)
    torch.cuda.synchronize(device)
    assert out.shape == (M, N)
    assert torch.isfinite(out).all()

    out.sum().backward()
    torch.cuda.synchronize(device)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
