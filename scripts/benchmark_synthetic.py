from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from kimi26_rmsnorm_fusion import FusedRMSNormLinear, SimpleRMSNorm, max_abs_diff


KIMI_LIKE_CONFIGS = {
    "small": (1, 128, 1024, 4096),
    "kimi_attn_hidden": (1, 512, 7168, 7168),
    "kimi_mlp_dense": (1, 512, 7168, 18432),
    "kimi_expert": (1, 4096, 7168, 2048),
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark synthetic Kimi K2.6 RMSNorm+Linear fusion.")
    parser.add_argument("--config", choices=KIMI_LIKE_CONFIGS.keys(), default="small")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--out", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for both paths.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Use `auto` on Mac; CUDA only works on NVIDIA GPU machines.",
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, x: torch.Tensor, warmup: int, iters: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn(x)
        sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    args = parse_args()
    batch, seq, hidden, out = KIMI_LIKE_CONFIGS[args.config]
    batch = args.batch or batch
    seq = args.seq or seq
    hidden = args.hidden or hidden
    out = args.out or out

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
        print(f"Using float32 instead of {args.dtype} for reliable Mac/CPU benchmarking.")
        dtype = torch.float32
    torch.manual_seed(1234)

    norm = SimpleRMSNorm(hidden, eps=1e-5).to(device=device, dtype=dtype)
    linear = nn.Linear(hidden, out, bias=False).to(device=device, dtype=dtype)
    fused = FusedRMSNormLinear(norm, linear).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)

    baseline_fn = lambda t: linear(norm(t))
    fused_fn = fused
    if args.compile:
        baseline_fn = torch.compile(baseline_fn)
        fused_fn = torch.compile(fused_fn)

    with torch.no_grad():
        diff = max_abs_diff(baseline_fn(x).float(), fused_fn(x).float())

    baseline_ms = time_fn(baseline_fn, x, args.warmup, args.iters, device)
    fused_ms = time_fn(fused_fn, x, args.warmup, args.iters, device)

    print(f"config={args.config} compile={args.compile}")
    print(f"device={device} dtype={dtype} shape=({batch}, {seq}, {hidden}) -> {out}")
    print(f"max_abs_diff={diff:.6e}")
    print(f"baseline_ms={baseline_ms:.4f}")
    print(f"fused_ms={fused_ms:.4f}")
    print(f"speedup={baseline_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
