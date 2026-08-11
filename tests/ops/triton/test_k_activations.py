import torch

from mamba_ssm.ops.triton.k_activations import swiglu

from overflow_test_utils import gpu_memory_skipif


@gpu_memory_skipif(24)
def test_swiglu_large_row_count_no_overflow() -> None:
    # int64 to avoid int32 overflow, see TODO: LinkToFutureIssueInMamba.
    device = 'cuda'
    torch.manual_seed(0)
    M = 115_866
    N = 9_280  # xy is (M, 2*N); (M - 1) * stride_x_row = (M - 1) * 2*N > 2**31 - 1

    xy = torch.randn(M, 2 * N, dtype=torch.bfloat16, device=device, requires_grad=True)

    out = swiglu(xy)
    torch.cuda.synchronize(device)
    assert out.shape == (M, N)
    assert torch.isfinite(out).all()

    out.sum().backward()
    torch.cuda.synchronize(device)
    assert xy.grad is not None and torch.isfinite(xy.grad).all()
