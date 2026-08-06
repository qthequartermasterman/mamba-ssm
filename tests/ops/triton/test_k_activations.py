import torch

from mamba_ssm.ops.triton.k_activations import swiglu


def test_swiglu_large_row_count_no_overflow() -> None:
    # Regression test for a 32-bit pointer-arithmetic overflow in
    # _swiglu_fwd_kernel and _swiglu_bwd_kernel: `row * stride_x_row` was
    # computed in 32-bit and silently wrapped once it exceeded 2**31 - 1,
    # corrupting the pointer offset and causing a CUDA illegal memory
    # access, the same overflow class as layer_norm.py/layernorm_gated.py --
    # an ordinarily contiguous (M, N) input is enough once M * N is large,
    # since `row` ranges over all M flattened rows. See
    # https://github.com/triton-lang/triton/issues/1058.
    #
    # k_activations.py had no test coverage at all before this -- not even
    # indirectly through another module's test.
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
