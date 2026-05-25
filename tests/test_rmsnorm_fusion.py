import torch
from torch import nn

from kimi26_rmsnorm_fusion import FusedRMSNormLinear, SimpleRMSNorm


def test_fused_rmsnorm_linear_matches_baseline_float32():
    torch.manual_seed(1)
    norm = SimpleRMSNorm(64, eps=1e-5)
    linear = nn.Linear(64, 128, bias=False)
    fused = FusedRMSNormLinear(norm, linear)
    x = torch.randn(3, 5, 64)

    baseline = linear(norm(x))
    actual = fused(x)

    assert torch.allclose(baseline, actual, rtol=1e-5, atol=1e-5)


def test_fused_rmsnorm_linear_rejects_wrong_shape():
    norm = SimpleRMSNorm(32)
    linear = nn.Linear(16, 64, bias=False)

    try:
        FusedRMSNormLinear(norm, linear)
    except ValueError as exc:
        assert "input dimension" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible dimensions.")
