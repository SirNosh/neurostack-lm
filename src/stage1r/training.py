from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .mechanisms import RoutingResult, SparseRouter


PREFERRED_EXPERT_BY_FAMILY = {
    "babi": 0,
    "babilong": 0,
    "clutrr": 0,
    "multisession_chat": 2,
    "epbench": 2,
    "reasoning": 1,
    "prm800k": 3,
}


def preferred_expert(family: str) -> int:
    try:
        return PREFERRED_EXPERT_BY_FAMILY[family]
    except KeyError as error:
        raise ValueError(f"no preregistered router bootstrap label for {family}") from error


def router_qualification_loss(
    result: RoutingResult,
    families: list[str],
    *,
    step: int,
    anneal_steps: int,
    bootstrap_weight: float = 0.2,
    balance_weight: float = 0.01,
    z_weight: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batch-level router objective with annealed functional bootstrap labels."""
    targets = torch.tensor(
        [preferred_expert(family) for family in families],
        device=result.logits.device,
    )
    fraction = max(0.0, 1.0 - step / max(1, anneal_steps))
    bootstrap = SparseRouter.bootstrap_loss(result, targets)
    balance = SparseRouter.load_balance_loss(result)
    z_loss = SparseRouter.z_loss(result)
    total = (
        bootstrap_weight * fraction * bootstrap
        + balance_weight * balance
        + z_weight * z_loss
    )
    return total, {
        "router_bootstrap": bootstrap.detach(),
        "router_balance": balance.detach(),
        "router_z": z_loss.detach(),
        "router_bootstrap_weight": torch.tensor(
            bootstrap_weight * fraction, device=result.logits.device
        ),
    }


@dataclass
class EWCState:
    reference: dict[str, torch.Tensor]
    fisher: dict[str, torch.Tensor]

    @classmethod
    def from_model_gradients(cls, model: nn.Module) -> "EWCState":
        reference = {}
        fisher = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            reference[name] = parameter.detach().clone()
            fisher[name] = (
                parameter.grad.detach().square().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
        return cls(reference, fisher)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        parameters = dict(model.named_parameters())
        if not self.reference:
            return next(model.parameters()).new_zeros(())
        return sum(
            (
                self.fisher[name]
                * (parameters[name] - reference).square()
            ).sum()
            for name, reference in self.reference.items()
        )


def consolidation_loss(
    slow_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    retention_loss: torch.Tensor,
    ewc_penalty: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full-memory teacher to memory-disabled slow-path distillation objective."""
    task = F.cross_entropy(slow_logits, targets)
    distillation = F.kl_div(
        F.log_softmax(slow_logits, dim=-1),
        F.softmax(teacher_logits.detach(), dim=-1),
        reduction="batchmean",
    )
    total = task + 0.5 * distillation + 0.2 * retention_loss + 1e-4 * ewc_penalty
    return total, {
        "sleep_task": task.detach(),
        "sleep_full_to_slow_kl": distillation.detach(),
        "sleep_retention": retention_loss.detach(),
        "sleep_ewc": ewc_penalty.detach(),
    }


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def changed_parameters(
    model: nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach().cpu(), snapshot[name])
    }

