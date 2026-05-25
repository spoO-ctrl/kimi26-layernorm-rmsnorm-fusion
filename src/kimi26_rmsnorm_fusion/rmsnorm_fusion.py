from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _get_rmsnorm_eps(norm: nn.Module, default: float = 1e-5) -> float:
    for name in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(norm, name, None)
        if value is not None:
            return float(value)
    return default


def _get_rmsnorm_weight(norm: nn.Module) -> torch.Tensor:
    weight = getattr(norm, "weight", None)
    if weight is None:
        raise TypeError("RMSNorm module must expose a `weight` tensor.")
    return weight


def fuse_rmsnorm_linear(
    norm: nn.Module,
    linear: nn.Linear,
    *,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """Fold RMSNorm gamma into a following Linear layer.

    Kimi K2.6 uses RMSNorm. Unlike LayerNorm, RMSNorm has no mean-centering
    term, so the static transform is simply W_fused = W * gamma.
    Runtime still needs the per-token RMS denominator.
    """
    norm_weight = _get_rmsnorm_weight(norm)
    linear_weight = linear.weight
    if linear_weight.shape[-1] != norm_weight.numel():
        raise ValueError(
            "Linear input dimension must match RMSNorm weight length: "
            f"{linear_weight.shape[-1]} != {norm_weight.numel()}"
        )

    if detach:
        norm_weight = norm_weight.detach()
        linear_weight = linear_weight.detach()

    fused_weight = linear_weight * norm_weight.to(linear_weight.dtype).unsqueeze(0)
    fused_bias = linear.bias.detach() if detach and linear.bias is not None else linear.bias
    return fused_weight.contiguous(), fused_bias, _get_rmsnorm_eps(norm)


class SimpleRMSNorm(nn.Module):
    """Small RMSNorm compatible with Kimi/DeepSeek-style math for local tests."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon).to(x.dtype)
        return x * self.weight.to(x.dtype)


@dataclass(frozen=True)
class FusionReport:
    max_diff: float
    mean_diff: float
    allclose: bool


class FusedRMSNormLinear(nn.Module):
    """Drop-in inference module for Linear(RMSNorm(x)).

    This is a correctness-first PyTorch prototype. For production speedups,
    port this math to Triton, CUDA, vLLM, SGLang, or TensorRT-LLM.
    """

    def __init__(
        self,
        norm: nn.Module,
        linear: nn.Linear,
        *,
        detach: bool = True,
        rms_compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        fused_weight, fused_bias, eps = fuse_rmsnorm_linear(norm, linear, detach=detach)
        self.weight = nn.Parameter(fused_weight, requires_grad=not detach)
        if fused_bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(fused_bias.clone(), requires_grad=not detach)
        self.eps = eps
        self.rms_compute_dtype = rms_compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(self.rms_compute_dtype).pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps).to(x.dtype)
        out = (x * inv_rms) @ self.weight.to(x.dtype).transpose(-1, -2)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def compare(self, norm: nn.Module, linear: nn.Linear, x: torch.Tensor) -> FusionReport:
        baseline = linear(norm(x))
        fused = self(x)
        diff = (baseline - fused).abs()
        return FusionReport(
            max_diff=float(diff.max().item()),
            mean_diff=float(diff.mean().item()),
            allclose=bool(torch.allclose(baseline, fused, rtol=1e-3, atol=1e-3)),
        )


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())
