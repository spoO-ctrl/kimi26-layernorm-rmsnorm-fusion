from __future__ import annotations

import argparse

import torch
from torch import nn

from kimi26_rmsnorm_fusion import FusedRMSNormLinear, SimpleRMSNorm, max_abs_diff


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RMSNorm+Linear fusion equivalence.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--out", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Use `auto` on Mac; CUDA only works on NVIDIA GPU machines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device_name = default_device() if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this PyTorch install. On Mac, run with "
            "`--device auto`, `--device mps`, or `--device cpu`."
        )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch install. Use `--device cpu`.")
    device = torch.device(device_name)
    if device.type in {"cpu", "mps"} and args.dtype in {"bfloat16", "float16"}:
        print(f"Using float32 instead of {args.dtype} for reliable Mac/CPU verification.")
        dtype = torch.float32

    torch.manual_seed(1234)
    norm = SimpleRMSNorm(args.hidden, eps=1e-5).to(device=device, dtype=dtype)
    linear = nn.Linear(args.hidden, args.out, bias=False).to(device=device, dtype=dtype)
    fused = FusedRMSNormLinear(norm, linear).to(device=device, dtype=dtype)
    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)

    with torch.no_grad():
        baseline = linear(norm(x))
        fused_out = fused(x)
        diff = max_abs_diff(baseline.float(), fused_out.float())
        mean_diff = float((baseline.float() - fused_out.float()).abs().mean().item())

    print(f"device={device} dtype={dtype} shape=({args.batch}, {args.seq}, {args.hidden}) -> {args.out}")
    print(f"max_abs_diff={diff:.6e}")
    print(f"mean_abs_diff={mean_diff:.6e}")
    print("PASS" if diff < 2e-2 else "CHECK_DIFF")


if __name__ == "__main__":
    main()
