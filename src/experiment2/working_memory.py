from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class WorkingMemoryBatch:
    keys: torch.Tensor
    values: torch.Tensor
    occupied: torch.Tensor


class BootstrapWorkingMemory(nn.Module):
    """Eight-slot differentiable A1 memory with distinct support addresses."""

    def __init__(
        self,
        cognitive_dim: int = 256,
        key_dim: int = 64,
        slots: int = 8,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.temperature = temperature
        self.key = nn.Linear(cognitive_dim, key_dim)
        self.query = nn.Linear(cognitive_dim, key_dim)
        self.value = nn.Linear(cognitive_dim, cognitive_dim)
        self.operation = nn.Linear(cognitive_dim, 5)
        self.address = nn.Linear(cognitive_dim, slots)

    def targets(self, support_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        operation = torch.zeros_like(support_mask, dtype=torch.long)
        address = torch.full_like(support_mask, -100, dtype=torch.long)
        for row, mask in enumerate(support_mask.bool()):
            for rank, fact in enumerate(mask.nonzero(as_tuple=False).flatten().tolist()):
                operation[row, fact] = 1  # REPLACE
                address[row, fact] = rank % self.slots
        return operation, address

    def write(
        self, facts: torch.Tensor, support_mask: torch.Tensor
    ) -> tuple[WorkingMemoryBatch, torch.Tensor, torch.Tensor]:
        batch, _, dim = facts.shape
        operation_logits = self.operation(facts)
        address_logits = self.address(facts)
        keys = facts.new_zeros((batch, self.slots, self.key.out_features))
        values = facts.new_zeros((batch, self.slots, dim))
        occupied = torch.zeros((batch, self.slots), device=facts.device, dtype=torch.bool)
        _, address_targets = self.targets(support_mask)
        encoded_keys, encoded_values = self.key(facts), self.value(facts)
        for row in range(batch):
            for fact in support_mask[row].bool().nonzero(as_tuple=False).flatten().tolist():
                slot = int(address_targets[row, fact])
                keys[row, slot] = encoded_keys[row, fact]
                values[row, slot] = encoded_values[row, fact]
                occupied[row, slot] = True
        return WorkingMemoryBatch(keys, values, occupied), operation_logits, address_logits

    def read(
        self, memory: WorkingMemoryBatch, question: torch.Tensor, *, lesion: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lesion:
            return question.new_zeros(question.shape), question.new_zeros(
                (question.shape[0], self.slots)
            )
        query = F.normalize(self.query(question), dim=-1)
        keys = F.normalize(memory.keys, dim=-1)
        logits = torch.einsum("bd,bsd->bs", query, keys) / self.temperature
        logits = logits.masked_fill(~memory.occupied, -1e4)
        weights = logits.softmax(-1)
        weights = torch.where(
            memory.occupied.any(-1, keepdim=True), weights, torch.zeros_like(weights)
        )
        return torch.einsum("bs,bsd->bd", weights, memory.values), weights


def working_memory_use_loss(
    with_memory_nll: torch.Tensor, lesioned_nll: torch.Tensor
) -> torch.Tensor:
    return F.softplus(0.1 + with_memory_nll - lesioned_nll).mean()
