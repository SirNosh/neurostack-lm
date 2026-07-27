from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import nn


def expert_flop_audit(
    *,
    batch_size: int,
    tokens: int,
    cycles: int,
    hidden_size: int = 896,
    bottleneck: int = 128,
    locations: int = 4,
    computed_experts: int = 4,
    selected_experts: int = 2,
) -> dict[str, int | str]:
    """Count both matmuls for every expert the current dense code executes."""
    flops_per_expert = 4 * batch_size * tokens * hidden_size * bottleneck
    actual = cycles * locations * computed_experts * flops_per_expert
    selected_only = cycles * locations * selected_experts * flops_per_expert
    return {
        "execution": "dense-experts-after-sparse-routing",
        "computed_experts": computed_experts,
        "selected_experts": selected_experts,
        "actual_expert_flops": actual,
        "hypothetical_selected_only_flops": selected_only,
    }


def hash_backbone_snapshot(model_path: Path) -> str:
    files = sorted(model_path.glob("*.safetensors"))
    config = model_path / "config.json"
    if config.exists():
        files.append(config)
    if not files:
        raise ValueError(f"no backbone weights found in {model_path}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def hash_module_parameters(module: nn.Module) -> str:
    """Hash names, dtypes, shapes, and bytes without retaining a second model copy."""
    digest = hashlib.sha256()
    for name, parameter in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        raw = parameter.detach().cpu().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def qwen_backbone_flops(
    *,
    sequence_lengths: list[int],
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    attention_heads: int,
    key_value_heads: int,
    vocabulary_size: int,
) -> int:
    """Standardized multiply-add estimate for full causal Qwen passes."""
    head_dim = hidden_size // attention_heads
    key_value_dim = key_value_heads * head_dim
    projection_weights = (
        2 * hidden_size * hidden_size
        + 2 * hidden_size * key_value_dim
        + 3 * hidden_size * intermediate_size
    )
    total = 0
    for tokens in sequence_lengths:
        linear = 2 * batch_size * tokens * projection_weights
        attention = (
            4
            * batch_size
            * attention_heads
            * tokens
            * tokens
            * head_dim
        )
        lm_head = 2 * batch_size * hidden_size * vocabulary_size
        total += layers * (linear + attention) + lm_head
    return total


def tensor_payload_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if hasattr(value, "__dataclass_fields__"):
        return sum(
            tensor_payload_bytes(getattr(value, field))
            for field in value.__dataclass_fields__
        )
    if isinstance(value, (list, tuple)):
        return sum(tensor_payload_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_payload_bytes(item) for item in value.values())
    return 0
