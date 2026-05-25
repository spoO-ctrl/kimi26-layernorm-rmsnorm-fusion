# Kimi K2.6 RMSNorm + Linear Fusion

This repo is a clean Kimi K2.6-oriented version of the LayerNorm fusion sample.
The important difference is that Kimi K2.6 uses RMSNorm, not classic LayerNorm.

## What This Implements

For a normal model block:

```text
y = Linear(RMSNorm(x))
```

RMSNorm is:

```text
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
```

Because RMSNorm has no mean subtraction and usually no beta, the static weight
folding is simple:

```text
W_fused = W * gamma
```

At runtime we still compute the per-token RMS denominator:

```text
y = Linear(x * inv_rms, W_fused)
```

This repo provides:

- `FusedRMSNormLinear`: correctness-first PyTorch module.
- `SimpleRMSNorm`: small local RMSNorm for testing.
- synthetic Kimi-like benchmarks.
- module-pair inspection helper for Transformers models.
- tests that prove the fused path matches the baseline.

## Why This Is Different From The LLaMA Sample

The LLaMA sample fuses classic LayerNorm:

```text
LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta
```

That requires a centering correction in the weights and a beta correction in the
bias. Kimi K2.6 uses `rms_norm_eps` and RMSNorm in its DeepSeek-style text stack,
so that LayerNorm formula should not be copied directly.

## Kimi K2.6 Shapes To Care About

The public Kimi K2.6 config uses:

```text
hidden_size = 7168
intermediate_size = 18432
moe_intermediate_size = 2048
num_hidden_layers = 61
num_routed_experts = 384
num_experts_per_tok = 8
rms_norm_eps = 1e-5
dtype = bfloat16
```

Start with synthetic tests before touching the full model.

## Setup

Run these commands from this repo:

```bash
cd /Users/spoo/Documents/Codex/2026-05-25/files-mentioned-by-the-user-l4/kimi26-rmsnorm-fusion
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you are on a CUDA machine, install the CUDA-enabled PyTorch wheel that matches
your driver before running benchmarks.

## Run Correctness Test

From:

```bash
cd /Users/spoo/Documents/Codex/2026-05-25/files-mentioned-by-the-user-l4/kimi26-rmsnorm-fusion
```

Run:

```bash
source .venv/bin/activate
pytest -q
python scripts/verify_equivalence.py
```

Mac/CPU fallback:

```bash
python scripts/verify_equivalence.py --device cpu --dtype float32 --hidden 1024 --out 4096
```

Apple Silicon MPS, if your PyTorch build supports it:

```bash
python scripts/verify_equivalence.py --device mps --dtype float32 --hidden 1024 --out 4096
```

## Run Benchmarks

On a Mac, use CPU or MPS. CUDA is only for NVIDIA GPU machines.

Mac CPU small benchmark:

```bash
python scripts/benchmark_synthetic.py --config small --device cpu --dtype float32
```

Apple Silicon MPS small benchmark, if available:

```bash
python scripts/benchmark_synthetic.py --config small --device mps --dtype float32
```

The full Kimi-like shapes can be too slow or memory-heavy on a MacBook Air. Use
smaller overrides locally:

```bash
python scripts/benchmark_synthetic.py --device cpu --dtype float32 --batch 1 --seq 64 --hidden 1024 --out 4096
```

On an NVIDIA CUDA machine, run:

```bash
python scripts/benchmark_synthetic.py --config small --device cuda
```

Kimi-like attention hidden projection:

```bash
python scripts/benchmark_synthetic.py --config kimi_attn_hidden --device cuda
```

Kimi-like dense MLP projection:

```bash
python scripts/benchmark_synthetic.py --config kimi_mlp_dense --device cuda
```

Kimi-like expert projection:

```bash
python scripts/benchmark_synthetic.py --config kimi_expert --device cuda
```

Try PyTorch compilation:

```bash
python scripts/benchmark_synthetic.py --config kimi_attn_hidden --device cuda --compile
```

## Inspect Real Model Module Pairs

Start with the tiny compatible model so you do not accidentally download the
full Kimi K2.6 checkpoint:

```bash
python scripts/find_norm_linear_pairs.py --model tiny-random/kimi-k2.6
```

For the real model, only run this on a machine prepared for Kimi K2.6:

```bash
python scripts/find_norm_linear_pairs.py --model moonshotai/Kimi-K2.6
```

## How To Apply This To Real Kimi K2.6

1. Load the Kimi K2.6 model with `trust_remote_code=True`.
2. Identify a pair shaped like:

```text
some_rmsnorm -> following_linear
```

Likely first targets:

```text
q_a_layernorm -> q_b_proj
kv_a_layernorm -> kv_b_proj
post_attention_layernorm -> first MLP/MoE projections
```

3. Replace the pair with `FusedRMSNormLinear` only for inference.
4. Verify output difference on the same input.
5. Benchmark with realistic token counts.
6. If PyTorch is not faster, move the same math into Triton/CUDA or the serving
   engine.

## Production Note

This repo proves the math and gives you a benchmark harness. Real inference
speedups usually require a real fused kernel in vLLM, SGLang, TensorRT-LLM, or
Triton. Plain PyTorch may not be faster because it still launches separate
operations internally.
