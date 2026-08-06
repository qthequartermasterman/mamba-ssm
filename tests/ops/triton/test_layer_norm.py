import math

import torch

from mamba_ssm.ops.triton.layer_norm import rms_norm_fn


def test_rms_norm_large_row_count_no_overflow() -> None:
    # int64 to avoid int32 overflow, see TODO: LinkToFutureIssueInMamba.
    # Backward launches only sm_count programs (not one per row), so its
    # max row_start is well under M - 1 -- N needs enough headroom that
    # the backward case overflows too, not just M * N for forward.
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
