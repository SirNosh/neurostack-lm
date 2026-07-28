import copy

import torch
import torch.nn.functional as F

from src.stage1r.mechanisms import EpisodicMemory, LesionConfig
from src.stage1r.model import Stage1RNeuroStack
from stage1r_helpers import ToyCausalLM, toy_inputs


def make_model() -> Stage1RNeuroStack:
    torch.manual_seed(7)
    return Stage1RNeuroStack(ToyCausalLM(), hidden_size=32)


def test_workspace_broadcast_changes_next_cycle_representation():
    model = make_model()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    output = model(
        ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=2,
    )
    assert not torch.equal(
        output.cycles[0].hidden_summary, output.cycles[1].hidden_summary
    )


def test_first_cycle_router_sees_current_contextual_input():
    model = make_model()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    first = model(
        ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final
    changed_ids = ids.clone()
    changed_ids[:, -1] = (changed_ids[:, -1] + 1) % model.backbone.lm_head.out_features
    second = model(
        changed_ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final
    assert not torch.equal(first.routing_input[:, :256], second.routing_input[:, :256])


def test_two_cycle_task_loss_reaches_memory_and_workspace_decisions():
    model = make_model()
    model.train()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    output = model(
        ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=2,
    )
    F.cross_entropy(output.final.token_logits, torch.tensor([1, 2])).backward()
    assert model.working_operation.weight.grad is not None
    assert model.working_operation.weight.grad.abs().sum() > 0
    assert model.working_slot.weight.grad is not None
    assert model.working_slot.weight.grad.abs().sum() > 0
    assert model.workspace.scorer[-1].weight.grad is not None
    assert model.workspace.scorer[-1].weight.grad.abs().sum() > 0


def test_token_support_scores_are_exposed_and_trainable():
    model = make_model()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    output = model(
        ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final
    assert output.support_logits.shape == ids.shape
    F.binary_cross_entropy_with_logits(
        output.support_logits, torch.zeros_like(output.support_logits)
    ).backward()
    assert model.support_scorer.weight.grad is not None
    assert model.support_scorer.weight.grad.abs().sum() > 0


def test_state_persists_across_calls_and_reset_isolates_examples():
    model = make_model()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    first = model(
        ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final.state
    second = model(
        ids,
        mask,
        first,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final.state
    assert not torch.equal(first.pfc.slots, second.pfc.slots)
    reset = model.reset_state(second, torch.tensor([True, False]))
    assert reset.pfc.slots[0].count_nonzero() == 0
    torch.testing.assert_close(reset.pfc.slots[1], second.pfc.slots[1])


def test_wake_feedback_changes_fast_state_but_not_slow_parameters():
    model = make_model()
    model.set_wake_mode()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    memory = EpisodicMemory()
    output = model(
        ids,
        mask,
        state,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "y"],
        cycles=1,
    ).final
    updated = model.apply_wake_feedback(
        output,
        outcome=torch.ones(2),
        encode_targets=torch.tensor([True, False]),
        bootstrap_mode=True,
        episodic_memory=memory,
        session_ids=["a", "b"],
        task_contexts=["x", "y"],
        timestamps=[1, 1],
        provenances=["p1", "p2"],
    )
    assert updated.fast_weights.pfc.matrix.abs().sum() > 0
    assert len(memory.events) == 1
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_eval_episodic_write_uses_model_decision_without_annotation():
    model = make_model()
    model.eval()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    memory = EpisodicMemory()
    output = model(
        ids,
        mask,
        state,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final
    output.controls.memory_write[:] = torch.tensor([0.9, 0.1])
    model.apply_wake_feedback(
        output,
        outcome=torch.ones(2),
        episodic_memory=memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        timestamps=[1, 1],
        provenances=["model", "model"],
    )
    assert [event.session_id for event in memory.events] == ["a"]


def test_sleep_step_changes_only_allowed_slow_parameters():
    model = make_model()
    model.set_sleep_mode()
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all(
        name.startswith("adapters.") or name.startswith("retrieval_integration.")
        for name in trainable
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    output = model(
        ids,
        mask,
        state,
        EpisodicMemory(),
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final
    optimizer = torch.optim.SGD(
        [value for value in model.parameters() if value.requires_grad], lr=0.01
    )
    loss = F.cross_entropy(output.token_logits, torch.tensor([1, 2]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = {
        name
        for name, value in model.named_parameters()
        if not torch.equal(value.detach(), before[name])
    }
    assert changed
    assert changed <= trainable


def test_episodic_lesion_returns_no_retrieved_events():
    model = make_model()
    ids, mask = toy_inputs()
    state = model.initialize_state(2, device="cpu", dtype=torch.float32)
    memory = EpisodicMemory()
    seeded = model(
        ids,
        mask,
        state,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
    ).final
    model.apply_wake_feedback(
        seeded,
        outcome=torch.ones(2),
        encode_targets=torch.ones(2, dtype=torch.bool),
        bootstrap_mode=True,
        episodic_memory=memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        timestamps=[1, 1],
        provenances=["a", "b"],
    )
    lesioned = model(
        ids,
        mask,
        seeded.state,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
        lesions=LesionConfig(episodic=False),
    ).final
    assert all(not events for events in lesioned.retrieval.events)


def test_lesion_switches_disable_assigned_paths():
    model = make_model()
    ids, mask = toy_inputs()
    memory = EpisodicMemory()
    initial = model.initialize_state(2, device="cpu", dtype=torch.float32)

    pfc_off = model(
        ids,
        mask,
        initial,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
        lesions=LesionConfig(pfc=False),
    ).final
    torch.testing.assert_close(pfc_off.state.pfc.slots, initial.pfc.slots)

    working_off = model(
        ids,
        mask,
        initial,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
        lesions=LesionConfig(working_memory=False),
    ).final
    torch.testing.assert_close(
        working_off.state.working_memory.values, initial.working_memory.values
    )

    control_off = model(
        ids,
        mask,
        initial,
        memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        cycles=1,
        lesions=LesionConfig(
            workspace=False,
            verifier=False,
            modulators=False,
            routing=False,
        ),
    ).final
    assert control_off.broadcast_tokens is None
    assert control_off.verifier_logits.count_nonzero() == 0
    torch.testing.assert_close(
        control_off.routing.probabilities,
        torch.full_like(control_off.routing.probabilities, 0.25),
    )
    assert control_off.modulators.da.count_nonzero() == 0
    torch.testing.assert_close(
        control_off.modulators.ne, torch.full_like(control_off.modulators.ne, 0.5)
    )

    fast_off_state = model.apply_wake_feedback(
        control_off,
        outcome=torch.ones(2),
        encode_targets=torch.zeros(2, dtype=torch.bool),
        bootstrap_mode=True,
        episodic_memory=memory,
        session_ids=["a", "b"],
        task_contexts=["x", "x"],
        timestamps=[1, 1],
        provenances=["a", "b"],
        lesions=LesionConfig(fast_weights=False),
    )
    assert fast_off_state.fast_weights.pfc.matrix.count_nonzero() == 0


def test_deterministic_end_to_end_smoke_lifetime():
    def lifetime():
        model = make_model()
        ids, mask = toy_inputs()
        memory = EpisodicMemory()
        state = model.initialize_state(2, device="cpu", dtype=torch.float32)
        wake = model(
            ids,
            mask,
            state,
            memory,
            session_ids=["a", "b"],
            task_contexts=["x", "x"],
            cycles=2,
        ).final
        state = model.apply_wake_feedback(
            wake,
            outcome=torch.tensor([1.0, -1.0]),
            encode_targets=torch.tensor([True, True]),
            bootstrap_mode=True,
            episodic_memory=memory,
            session_ids=["a", "b"],
            task_contexts=["x", "x"],
            timestamps=[1, 1],
            provenances=["a", "b"],
        )
        recalled = model(
            ids,
            mask,
            state,
            memory,
            session_ids=["a", "b"],
            task_contexts=["x", "x"],
            cycles=1,
        ).final
        return recalled.token_logits.detach(), len(memory.events)

    first_logits, first_events = lifetime()
    second_logits, second_events = lifetime()
    torch.testing.assert_close(first_logits, second_logits)
    assert first_events == second_events == 2
