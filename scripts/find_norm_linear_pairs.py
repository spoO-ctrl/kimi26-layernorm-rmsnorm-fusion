from __future__ import annotations

import argparse

import torch
from torch import nn
from transformers import AutoModel


NORM_NAME_HINTS = ("norm", "layernorm", "rmsnorm", "layer_norm")
LINEAR_NAME_HINTS = ("proj", "linear", "gate", "up", "down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print likely RMSNorm -> Linear pairs in a Transformers model."
    )
    parser.add_argument(
        "--model",
        default="tiny-random/kimi-k2.6",
        help="Use a tiny compatible model first; full Kimi K2.6 is very large.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--max-pairs", type=int, default=80)
    return parser.parse_args()


def looks_like_norm(name: str, module: nn.Module) -> bool:
    lower_name = name.lower()
    lower_type = type(module).__name__.lower()
    return any(hint in lower_name or hint in lower_type for hint in NORM_NAME_HINTS)


def looks_like_linear(name: str, module: nn.Module) -> bool:
    lower_name = name.lower()
    return isinstance(module, nn.Linear) or any(hint in lower_name for hint in LINEAR_NAME_HINTS)


def main() -> None:
    args = parse_args()
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float32,
    )

    printed = 0
    for block_name, block in model.named_modules():
        children = list(block.named_children())
        for idx, (child_name, child) in enumerate(children):
            if not looks_like_norm(child_name, child):
                continue
            following = children[idx + 1 : idx + 8]
            for next_name, next_module in following:
                if looks_like_linear(next_name, next_module):
                    print(f"{block_name}.{child_name} -> {block_name}.{next_name}")
                    printed += 1
                    break
            if printed >= args.max_pairs:
                return


if __name__ == "__main__":
    main()
