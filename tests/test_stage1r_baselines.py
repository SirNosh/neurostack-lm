import pytest
import torch
import torch.nn.functional as F

from src.stage1r.baselines import R0ParameterMatchedAdapter
from src.stage1r.model import Stage1RNeuroStack
from stage1r_helpers import ToyCausalLM, toy_inputs


def make_models():
    r5 = Stage1RNeuroStack(ToyCausalLM(), hidden_size=32)
    target = sum(
        parameter.numel() for parameter in r5.parameters() if parameter.requires_grad
    )
    r0 = R0ParameterMatchedAdapter(
        ToyCausalLM(),
        target_trainable_parameters=target,
        hidden_size=32,
    )
    return r0, target


def test_r0_matches_r5_trainable_parameters_within_two_percent():
    r0, target = make_models()
    assert abs(r0.trainable_parameter_count - target) / target <= 0.02
    assert not any(
        parameter.requires_grad for parameter in r0.backbone.parameters()
    )
    assert len(r0.adapters) == 4
    assert r0.adapter_layer_indices == (5, 11, 17, 23)


def test_r0_uses_real_token_path_and_up_to_three_passes():
    r0, _ = make_models()
    r0.eval()
    ids, mask = toy_inputs()
    output = r0(ids, mask, passes=3)
    assert output.backbone_passes == 3
    assert len(output.pass_logits) == 3
    assert len(output.feedback_tokens) == 2
    assert not torch.equal(output.pass_summaries[0], output.pass_summaries[1])
    assert output.token_logits.shape == (2, 64)
    with pytest.raises(ValueError):
        r0(ids, mask, passes=4)


def test_r0_sleep_updates_only_generic_adapters():
    r0, _ = make_models()
    r0.set_sleep_mode()
    before = {
        name: parameter.detach().clone()
        for name, parameter in r0.named_parameters()
    }
    ids, mask = toy_inputs()
    loss = F.cross_entropy(r0(ids, mask).token_logits, torch.tensor([1, 2]))
    loss.backward()
    optimizer = torch.optim.SGD(
        [parameter for parameter in r0.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.step()
    changed = {
        name
        for name, parameter in r0.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert changed
    assert all(name.startswith("adapters.") for name in changed)


def test_r0_flops_count_all_parameter_matched_adapters():
    r0, _ = make_models()
    expected = (
        4
        * 2
        * (16 * 3 + 4 * 2)
        * 32
        * sum(adapter.bottleneck for adapter in r0.adapters)
    )
    assert r0.adapter_matmul_flops(batch_size=2, tokens=16, passes=3) == expected


def test_r0_feedback_changes_iterative_result():
    r0, _ = make_models()
    r0.eval()
    ids, mask = toy_inputs()
    iterative = r0(ids, mask, passes=3, feedback_enabled=True)
    zero_feedback = r0(ids, mask, passes=3, feedback_enabled=False)
    assert not torch.equal(
        iterative.pass_summaries[-1], zero_feedback.pass_summaries[-1]
    )
    assert all(token.abs().sum() > 0 for token in iterative.feedback_tokens)
    assert all(token.abs().sum() == 0 for token in zero_feedback.feedback_tokens)
