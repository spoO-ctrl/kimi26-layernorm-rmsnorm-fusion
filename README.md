# Kimi K2.6 RMSNorm + Linear Fusion

This repo implements and benchmarks normalization fusion for **actual Kimi K2.6**.
Although Kimi's module names include strings such as `q_a_layernorm`,
`kv_a_layernorm`, `input_layernorm`, and `post_attention_layernorm`, the
underlying class in the downloaded model code is:

```text
DeepseekV3RMSNorm
```

So the correct Kimi K2.6 implementation is **RMSNorm + Linear fusion**, not
classic LayerNorm + Linear fusion.

## Verified Kimi K2.6 Facts

The following was verified from `moonshotai/Kimi-K2.6` config and model code:

| Item | Value |
| --- | --- |
| Architecture | `KimiK25ForConditionalGeneration` |
| Outer model type | `kimi_k25` |
| Text model type | `kimi_k2` |
| Norm class | `DeepseekV3RMSNorm` |
| `rms_norm_eps` | `1e-5` |
| Hidden size | `7168` |
| Dense intermediate size | `18432` |
| MoE intermediate size | `2048` |
| Hidden layers | `61` |
| Routed experts | `384` |
| Experts per token | `8` |
| Main dtype | `bfloat16` |

Relevant model-code locations found in `modeling_deepseek.py`:

| Line | Evidence |
| --- | --- |
| `94` | `class DeepseekV3RMSNorm(nn.Module)` |
| `668` | `self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)` |
| `678` | `self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)` |
| `770` | `q = self.q_b_proj(self.q_a_layernorm(...))` |
| `779` | `kv = self.kv_b_proj(self.kv_a_layernorm(...))` |
| `1148` | `self.input_layernorm = DeepseekV3RMSNorm(...)` |
| `1150` | `self.post_attention_layernorm = DeepseekV3RMSNorm(...)` |
| `1351` | `self.norm = DeepseekV3RMSNorm(...)` |

## Fusion Formula

Classic LayerNorm is:

```text
LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta
```

Kimi K2.6 uses RMSNorm:

```text
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
```

For a following linear layer:

```text
y = Linear(RMSNorm(x))
```

we fold the RMSNorm scale `gamma` into the linear weight:

```text
W_fused = W * gamma
```

Runtime still computes the per-token RMS denominator:

```text
inv_rms = rsqrt(mean(x^2) + eps)
y = Linear(x * inv_rms, W_fused)
```

This is simpler than the LLaMA LayerNorm sample because RMSNorm has no
mean-subtraction correction and no beta-bias correction.

## What This Repo Contains

| File | Purpose |
| --- | --- |
| `src/kimi26_rmsnorm_fusion/rmsnorm_fusion.py` | `FusedRMSNormLinear`, `SimpleRMSNorm`, and weight-folding logic |
| `tests/test_rmsnorm_fusion.py` | Unit tests proving fused output matches baseline |
| `scripts/verify_equivalence.py` | CUDA/CPU correctness check with Kimi-like dimensions |
| `scripts/benchmark_synthetic.py` | Synthetic Kimi-shaped benchmark harness |
| `scripts/find_norm_linear_pairs.py` | Helper for finding norm-to-linear pairs in a Transformers model |

## Benchmark Results

Hardware:

```text
RunPod
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.8.0+cu128
CUDA: available
dtype: bfloat16
```

Stable benchmark command pattern:

```bash
python scripts/benchmark_synthetic.py --config <config> --device cuda --iters 300
```

| Config | Shape | Baseline | Fused | Speedup | Max Diff |
| --- | --- | ---: | ---: | ---: | ---: |
| `small` | `(1, 128, 1024) -> 4096` | `0.0439 ms` | `0.0382 ms` | `1.149x` | `0.000000e+00` |
| `kimi_attn_hidden` | `(1, 512, 7168) -> 7168` | `0.4016 ms` | `0.3890 ms` | `1.032x` | `0.000000e+00` |
| `kimi_expert` | `(1, 4096, 7168) -> 2048` | `1.4923 ms` | `1.3658 ms` | `1.093x` | `0.000000e+00` |
| `kimi_mlp_dense` | `(1, 512, 7168) -> 18432` | `0.9179 ms` | `0.9027 ms` | `1.017x` | `0.000000e+00` |

Correctness check:

```text
device=cuda dtype=torch.bfloat16 shape=(2, 16, 7168) -> 4096
max_abs_diff=0.000000e+00
mean_abs_diff=0.000000e+00
PASS
```

## Setup

On Mac or Linux:

```bash
cd /path/to/kimi26-layernorm-rmsnorm-fusion
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On RunPod's official PyTorch image, the global Python may already include CUDA
PyTorch. If a fresh `.venv` cannot import `torch`, either install CUDA PyTorch
inside the venv or use the global environment:

```bash
deactivate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

## Run Correctness Tests

```bash
pytest -q
python scripts/verify_equivalence.py --device cuda
```

Mac CPU fallback:

```bash
python scripts/verify_equivalence.py --device cpu --dtype float32 --hidden 1024 --out 4096
```

Apple Silicon MPS, if supported:

```bash
python scripts/verify_equivalence.py --device mps --dtype float32 --hidden 1024 --out 4096
```

## Run Benchmarks

CUDA:

```bash
mkdir -p results

python scripts/benchmark_synthetic.py --config small --device cuda --iters 300 | tee results/small.txt
python scripts/benchmark_synthetic.py --config kimi_attn_hidden --device cuda --iters 300 | tee results/kimi_attn_hidden.txt
python scripts/benchmark_synthetic.py --config kimi_expert --device cuda --iters 300 | tee results/kimi_expert.txt
python scripts/benchmark_synthetic.py --config kimi_mlp_dense --device cuda --iters 300 | tee results/kimi_mlp_dense.txt
```

Mac/CPU quick check:

```bash
python scripts/benchmark_synthetic.py --device cpu --dtype float32 --batch 1 --seq 64 --hidden 1024 --out 4096
```

## Verify Actual Kimi K2.6 Code

Download only config and Python files, not the full weights:

```bash
mkdir -p /workspace/kimi26_code

hf download moonshotai/Kimi-K2.6 \
  --local-dir /workspace/kimi26_code \
  --include config.json \
  --include "*.py"
```

Inspect config:

```bash
python - <<'PY'
import json
cfg = json.load(open("/workspace/kimi26_code/config.json"))
tc = cfg["text_config"]
print("architecture:", cfg["architectures"])
print("outer model_type:", cfg["model_type"])
print("text model_type:", tc["model_type"])
print("rms_norm_eps:", tc.get("rms_norm_eps"))
print("hidden_size:", tc["hidden_size"])
print("intermediate_size:", tc["intermediate_size"])
print("moe_intermediate_size:", tc["moe_intermediate_size"])
print("num_hidden_layers:", tc["num_hidden_layers"])
print("n_routed_experts:", tc["n_routed_experts"])
print("num_experts_per_tok:", tc["num_experts_per_tok"])
PY
```

Inspect norm usage:

```bash
grep -n "class .*Norm\|RMSNorm\|input_layernorm\|post_attention_layernorm\|q_a_layernorm\|kv_a_layernorm" \
  /workspace/kimi26_code/modeling_deepseek.py
```

## Recommended Actual Kimi K2.6 Fusion Targets

Start with direct RMSNorm-to-Linear pairs:

| Target | Why |
| --- | --- |
| `q_a_layernorm -> q_b_proj` | Direct pair in attention path |
| `kv_a_layernorm -> kv_b_proj` | Direct pair in attention path |
| `post_attention_layernorm -> mlp / MoE gate/up projections` | High-value feed-forward path |
| `final norm -> lm_head` | Optional output projection fusion |

Avoid starting with `input_layernorm -> attention`; that path fans into attention
projection, RoPE, cache, and kernel-specific logic. The direct `q_a_layernorm`
and `kv_a_layernorm` pairs are safer first integration points.

## Conclusion

For actual Kimi K2.6, the correct "LayerNorm fusion" deliverable is:

```text
Kimi K2.6 RMSNorm + Linear fusion
```

The naming is important: the model code uses variables with `layernorm` in the
name, but the implementation class is `DeepseekV3RMSNorm`. Applying the LLaMA
classic LayerNorm formula would be mathematically wrong for Kimi K2.6.

This repo proves the fusion math, validates exact output equivalence, and shows
small CUDA speedups on RTX 4090. Production speedups should be implemented in a
real serving path such as Triton, vLLM, SGLang, TensorRT-LLM, or a custom CUDA
kernel.
