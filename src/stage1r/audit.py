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
