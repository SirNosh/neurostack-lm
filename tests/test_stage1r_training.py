import torch
from torch import nn

from src.stage1r.mechanisms import SparseRouter
from src.stage1r.training import (
    EWCState,
    consolidation_loss,
    router_qualification_loss,
)


def test_router_bootstrap_labels_anneal_to_zero():
    router = SparseRouter(8)
    result = router(torch.randn(4, 8))
    early, early_parts = router_qualification_loss(
        result,
        ["babi", "multisession_chat", "reasoning", "prm800k"],
        step=0,
        anneal_steps=100,
    )
    late, late_parts = router_qualification_loss(
        result,
        ["babi", "multisession_chat", "reasoning", "prm800k"],
        step=100,
        anneal_steps=100,
    )
    assert early_parts["router_bootstrap_weight"] > 0
    assert late_parts["router_bootstrap_weight"] == 0
    assert torch.isfinite(early) and torch.isfinite(late)


def test_ewc_penalty_is_zero_at_reference_and_positive_after_change():
    model = nn.Linear(3, 2)
    model(torch.ones(1, 3)).sum().backward()
    ewc = EWCState.from_model_gradients(model)
    assert ewc.penalty(model) == 0
    with torch.no_grad():
        model.weight.add_(1)
    assert ewc.penalty(model) > 0


def test_consolidation_loss_contains_all_preregistered_terms():
    slow = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[3.0, -1.0], [-1.0, 3.0]])
    total, parts = consolidation_loss(
        slow,
        teacher,
        torch.tensor([0, 1]),
        retention_loss=torch.tensor(0.4),
        ewc_penalty=torch.tensor(2.0),
    )
    reconstructed = (
        parts["sleep_task"]
        + 0.5 * parts["sleep_full_to_slow_kl"]
        + 0.2 * parts["sleep_retention"]
        + 1e-4 * parts["sleep_ewc"]
    )
    torch.testing.assert_close(total.detach(), reconstructed)

