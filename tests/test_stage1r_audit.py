import torch

from src.stage1r.audit import (
    expert_flop_audit,
    qwen_backbone_flops,
    tensor_payload_bytes,
)


def test_dense_expert_execution_counts_all_four_experts():
    audit = expert_flop_audit(batch_size=2, tokens=16, cycles=2)
    assert audit["computed_experts"] == 4
    assert audit["selected_experts"] == 2
    assert audit["actual_expert_flops"] == (
        2 * audit["hypothetical_selected_only_flops"]
    )


def test_qwen_flop_audit_increases_with_feedback_tokens():
    common = {
        "batch_size": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "layers": 2,
        "attention_heads": 4,
        "key_value_heads": 2,
        "vocabulary_size": 100,
    }
    plain = qwen_backbone_flops(sequence_lengths=[16, 16], **common)
    feedback = qwen_backbone_flops(sequence_lengths=[16, 20], **common)
    assert feedback > plain


def test_tensor_payload_bytes_counts_nested_dataclass_tensors():
    from dataclasses import dataclass

    @dataclass
    class State:
        first: torch.Tensor
        second: tuple[torch.Tensor]

    state = State(torch.zeros(3, dtype=torch.float32), (torch.zeros(2),))
    assert tensor_payload_bytes(state) == 20
