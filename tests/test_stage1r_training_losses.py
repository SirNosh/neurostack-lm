import torch

from src.stage1r.training import (
    memory_write_loss,
    working_memory_loss,
    workspace_admission_loss,
)


def test_working_memory_loss_reaches_operation_and_slot_logits():
    operation_logits = torch.randn(3, 5, requires_grad=True)
    slot_logits = torch.randn(3, 8, requires_grad=True)
    loss, terms = working_memory_loss(
        operation_logits,
        slot_logits,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 3, 7]),
        occupied=torch.ones(3, 8, dtype=torch.bool),
        protection=torch.zeros(3, 8),
    )
    loss.backward()
    assert operation_logits.grad.abs().sum() > 0
    assert slot_logits.grad.abs().sum() > 0
    assert set(terms) == {
        "working_operation",
        "working_slot",
        "working_write_sparsity",
        "working_false_overwrite",
    }


def test_workspace_and_memory_write_losses_are_trainable():
    logits = torch.randn(2, 4, requires_grad=True)
    workspace_loss = workspace_admission_loss(
        logits,
        torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]),
        torch.tensor([[True, True, False, True], [True, True, True, True]]),
    )
    write_logits = torch.randn(2, requires_grad=True)
    write_probability = torch.sigmoid(write_logits)
    write_loss = memory_write_loss(write_probability, torch.tensor([True, False]))
    (workspace_loss + write_loss).backward()
    assert logits.grad.abs().sum() > 0
    assert write_logits.grad.abs().sum() > 0
