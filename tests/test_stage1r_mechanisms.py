from dataclasses import fields

import pytest
import torch

from src.stage1r.mechanisms import (
    EpisodicEvent,
    EpisodicMemory,
    FastWeightAdapter,
    MemoryOperation,
    ModulatorController,
    ModulatorSignals,
    PersistentPFC,
    RoutingResult,
    SparseRouter,
    Verifier,
    WorkingMemory,
    Workspace,
)


def test_pfc_open_gate_changes_state():
    pfc = PersistentPFC(input_dim=12, slot_dim=8)
    state = pfc.initialize(2)
    updated = pfc.update(torch.randn(2, 12), state, torch.ones(2, 4))
    assert not torch.equal(updated.slots, state.slots)


def test_pfc_closed_gate_preserves_state():
    pfc = PersistentPFC(input_dim=12, slot_dim=8)
    state = pfc.initialize(2)
    state.slots.normal_()
    updated = pfc.update(torch.randn(2, 12), state, torch.zeros(2, 4))
    torch.testing.assert_close(updated.slots, state.slots)


def test_pfc_reset_is_row_scoped():
    pfc = PersistentPFC(input_dim=12, slot_dim=8)
    state = pfc.initialize(2)
    state.slots.fill_(1)
    reset = pfc.reset(state, torch.tensor([True, False]))
    assert reset.slots[0].count_nonzero() == 0
    torch.testing.assert_close(reset.slots[1], state.slots[1])


def test_working_memory_replace_writes_selected_slot():
    memory = WorkingMemory(slots=8, key_dim=4, value_dim=6)
    state = memory.initialize(1)
    state = memory.operate(
        state,
        torch.tensor([MemoryOperation.REPLACE]),
        torch.tensor([3]),
        key=torch.ones(1, 4),
        value=torch.ones(1, 6),
        confidence=torch.tensor([0.8]),
    )
    assert state.occupied[0, 3]
    assert state.confidence[0, 3] == pytest.approx(0.8)


def test_working_memory_keep_is_noop():
    memory = WorkingMemory(slots=8, key_dim=4, value_dim=6)
    state = memory.initialize(1)
    state.values[0, 2].fill_(7)
    kept = memory.operate(
        state, torch.tensor([MemoryOperation.KEEP]), torch.tensor([2])
    )
    torch.testing.assert_close(kept.values[0, 2], state.values[0, 2])


def test_working_memory_protection_blocks_clear():
    memory = WorkingMemory(slots=8, key_dim=4, value_dim=6)
    state = memory.initialize(1)
    state.occupied[0, 1] = True
    state.values[0, 1].fill_(3)
    state = memory.operate(
        state, torch.tensor([MemoryOperation.PROTECT]), torch.tensor([1])
    )
    state = memory.operate(
        state, torch.tensor([MemoryOperation.CLEAR]), torch.tensor([1])
    )
    assert state.occupied[0, 1]
    assert state.values[0, 1].sum() == 18


def test_working_memory_merge_combines_values():
    memory = WorkingMemory(slots=8, key_dim=4, value_dim=6)
    state = memory.initialize(1)
    state.occupied[0, 0] = True
    state.confidence[0, 0] = 1
    state.values[0, 0].fill_(2)
    merged = memory.operate(
        state,
        torch.tensor([MemoryOperation.MERGE]),
        torch.tensor([0]),
        key=torch.ones(1, 4),
        value=torch.full((1, 6), 4.0),
        confidence=torch.tensor([1.0]),
    )
    torch.testing.assert_close(merged.values[0, 0], torch.full((6,), 3.0))


def test_working_memory_straight_through_choices_receive_gradients():
    memory = WorkingMemory(slots=8, key_dim=4, value_dim=6)
    state = memory.initialize(2)
    operation_logits = torch.randn(2, 5, requires_grad=True)
    slot_logits = torch.randn(2, 8, requires_grad=True)
    operation_soft = operation_logits.softmax(-1)
    slot_soft = slot_logits.softmax(-1)
    operation_hard = torch.nn.functional.one_hot(
        torch.ones(2, dtype=torch.long), 5
    ).float()
    slot_hard = torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 8).float()
    updated = memory.operate_differentiable(
        state,
        operation_hard + operation_soft - operation_soft.detach(),
        slot_hard + slot_soft - slot_soft.detach(),
        key=torch.randn(2, 4),
        value=torch.randn(2, 6),
        confidence=torch.full((2,), 0.8),
    )
    updated.values.square().sum().backward()
    assert operation_logits.grad is not None
    assert slot_logits.grad is not None
    assert operation_logits.grad.abs().sum() > 0
    assert slot_logits.grad.abs().sum() > 0


def test_workspace_capacity_is_exactly_four():
    workspace = Workspace(value_dim=8, capacity=4, broadcast_dim=12)
    result = workspace.compete([("token", torch.randn(2, 7, 8))])
    assert result.slots.shape == (2, 4, 8)
    assert result.sources.shape == (2, 4)


def test_workspace_broadcast_projects_all_slots():
    workspace = Workspace(value_dim=8, capacity=4, broadcast_dim=12)
    result = workspace.compete([("pfc", torch.randn(2, 4, 8))])
    broadcast = workspace.broadcast(result)
    assert broadcast.shape == (2, 4, 12)
    assert not torch.equal(broadcast, torch.zeros_like(broadcast))


def test_workspace_soft_admission_trains_scorer_and_exposes_candidates():
    workspace = Workspace(value_dim=8, capacity=4, broadcast_dim=12)
    workspace.train()
    result = workspace.compete([("token", torch.randn(2, 7, 8))])
    result.slots.square().sum().backward()
    assert result.all_candidate_logits.shape == (2, 7)
    assert result.selected_indices.shape == (2, 4)
    assert result.candidate_source_ids.shape == (2, 7)
    assert result.candidate_valid_masks.all()
    assert workspace.scorer[-1].weight.grad is not None
    assert workspace.scorer[-1].weight.grad.abs().sum() > 0


def test_router_selects_exactly_top_two():
    router = SparseRouter(input_dim=10)
    result = router(torch.randn(5, 10))
    assert result.indices.shape == (5, 4, 2)
    torch.testing.assert_close(result.weights.sum(-1), torch.ones(5, 4))


def test_router_load_balance_penalizes_collapse():
    uniform_probabilities = torch.full((8, 4, 4), 0.25)
    balanced_indices = torch.arange(4).view(4, 1, 1).expand(4, 4, 2)
    balanced_indices = torch.cat([balanced_indices, balanced_indices], dim=0)
    balanced = RoutingResult(
        torch.zeros_like(uniform_probabilities),
        uniform_probabilities,
        balanced_indices,
        torch.full((8, 4, 2), 0.5),
    )
    collapsed = RoutingResult(
        torch.tensor([4.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(8, 4, 4),
        torch.softmax(
            torch.tensor([4.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(8, 4, 4),
            dim=-1,
        ),
        torch.zeros(8, 4, 2, dtype=torch.long),
        torch.full((8, 4, 2), 0.5),
    )
    assert SparseRouter.load_balance_loss(balanced) < SparseRouter.load_balance_loss(
        collapsed
    )


def _event(session: str, value: float = 1.0) -> EpisodicEvent:
    return EpisodicEvent(
        key=torch.tensor([1.0, 0.0]),
        value=torch.full((4,), value),
        timestamp=1,
        session_id=session,
        task_context="task",
        goal_state=torch.zeros(2),
        workspace_summary=torch.zeros(4),
        outcome=1,
        confidence=0.9,
        provenance="official-example",
    )


def test_episodic_retrieval_is_session_isolated():
    memory = EpisodicMemory(value_dim=4, top_k=1)
    memory.write(_event("a", 2))
    memory.write(_event("b", 9))
    result = memory.retrieve(torch.tensor([[1.0, 0.0]]), session_ids=["a"])
    torch.testing.assert_close(result.values[0, 0], torch.full((4,), 2.0))
    assert result.events[0][0].session_id == "a"


def test_episodic_retrieval_breadth_is_per_example():
    memory = EpisodicMemory(value_dim=4, top_k=4)
    for value in range(1, 5):
        memory.write(_event("a", value))
        memory.write(_event("b", value))
    result = memory.retrieve(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        session_ids=["a", "b"],
        breadths=torch.tensor([1, 3]),
    )
    assert len(result.events[0]) == 1
    assert len(result.events[1]) == 3
    assert (result.scores[0, 1:] < -1e3).all()


def test_episodic_event_has_no_answer_label_field():
    names = {item.name for item in fields(EpisodicEvent)}
    assert "answer" not in names
    assert "answer_id" not in names
    assert "logits" not in names


def test_fast_weights_begin_at_zero():
    adapter = FastWeightAdapter(4, 6, rank=2)
    state = adapter.initialize(3, device="cpu", dtype=torch.float32)
    assert state.matrix.count_nonzero() == 0
    assert adapter.apply(torch.randn(3, 4), state).count_nonzero() == 0


def test_fast_weight_feedback_changes_matrix_and_reset_removes_it():
    adapter = FastWeightAdapter(4, 6, rank=2)
    state = adapter.initialize(1, device="cpu", dtype=torch.float32)
    updated = adapter.update(
        state,
        torch.randn(1, 4),
        torch.randn(1, 6),
        da=torch.ones(1),
        ach=torch.ones(1),
        ne=torch.ones(1),
    )
    assert updated.matrix.abs().sum() > 0
    reset = adapter.initialize(1, device="cpu", dtype=torch.float32)
    assert reset.matrix.count_nonzero() == 0


def test_each_modulator_changes_assigned_control():
    controller = ModulatorController(3)
    low = torch.zeros(1)
    high = torch.ones(1)
    baseline = controller.controls(ModulatorSignals(low, low, low, low, low))
    da = controller.controls(ModulatorSignals(high, low, low, low, low))
    ne = controller.controls(ModulatorSignals(low, high, low, low, low))
    ach = controller.controls(ModulatorSignals(low, low, high, low, low))
    serotonin = controller.controls(ModulatorSignals(low, low, low, high, low))
    overload = controller.controls(ModulatorSignals(low, low, low, low, high))
    assert da.replay_priority > baseline.replay_priority
    assert ne.router_temperature < baseline.router_temperature
    assert ach.encode_weight > baseline.encode_weight
    assert serotonin.continue_bias > baseline.continue_bias
    assert overload.conflict_pressure > baseline.conflict_pressure


def test_generic_controller_is_same_size_and_can_cross_control_channels():
    differentiated = ModulatorController(3, differentiated=True)
    generic = ModulatorController(3, differentiated=False)
    assert sum(p.numel() for p in differentiated.parameters()) == sum(
        p.numel() for p in generic.parameters()
    )
    with torch.no_grad():
        generic.control_map.weight.zero_()
        generic.control_map.bias.zero_()
        generic.control_map.weight[8, 0] = 4
    low = torch.zeros(1)
    high = torch.ones(1)
    baseline = generic.controls(ModulatorSignals(low, low, low, low, low))
    changed = generic.controls(ModulatorSignals(high, low, low, low, low))
    assert changed.verify_threshold > baseline.verify_threshold


def test_verifier_requires_both_classes():
    verifier = Verifier(4)
    logits = verifier(torch.randn(4, 4))
    with pytest.raises(ValueError):
        verifier.loss(logits, torch.ones(4))
    loss = verifier.loss(logits, torch.tensor([0, 1, 0, 1]))
    assert torch.isfinite(loss)
