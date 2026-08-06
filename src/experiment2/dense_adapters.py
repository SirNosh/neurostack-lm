from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


BRANCHES = ("relational", "planning", "memory", "verification")


class DenseAdapterBranch(nn.Module):
    def __init__(self, hidden_size: int = 896, bottleneck: int = 128) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(hidden))))


class DenseAdapterBank(nn.Module):
    """Four always-executed branches with fixed mean aggregation."""

    def __init__(self, hidden_size: int = 896, bottleneck: int = 128) -> None:
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                name: DenseAdapterBranch(hidden_size, bottleneck)
                for name in BRANCHES
            }
        )
        self.last_branch_outputs: dict[str, torch.Tensor] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.last_branch_outputs = {
            name: branch(hidden) for name, branch in self.branches.items()
        }
        update = torch.stack(tuple(self.last_branch_outputs.values()), dim=0).mean(0)
        return hidden + update

    def train_only(self, branch_name: str | None) -> None:
        if branch_name is not None and branch_name not in self.branches:
            raise ValueError(f"unknown branch {branch_name}")
        for name, branch in self.branches.items():
            branch.requires_grad_(branch_name is None or name == branch_name)
