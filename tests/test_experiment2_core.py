import torch

from src.experiment2.data import Experiment2Example
from src.experiment2.dense_adapters import DenseAdapterBank
from src.experiment2.losses import (
    distinct_support_slots,
    full_sequence_labels,
    full_sequence_loss,
)
from src.experiment2.span_pooling import AttentiveSpanPooler
from src.experiment2.streams import EpisodeStreamLoader
from src.experiment2.runner import StreamRunner
from src.experiment2.model import DenseFrozenBackbone


def example(identifier: str, timestamp: int, reset: bool) -> Experiment2Example:
    return Experiment2Example(
        identifier, "epbench", "prompt", "answer", "book", "book", timestamp,
        [], None, [], identifier if timestamp == 0 else None, [], None, None, None,
        reset, reset, reset, reset,
    )


def test_dense_bank_executes_all_branches_and_starts_as_identity():
    bank = DenseAdapterBank(hidden_size=16, bottleneck=4)
    hidden = torch.randn(2, 5, 16)
    output = bank(hidden)
    assert tuple(bank.last_branch_outputs) == (
        "relational", "planning", "memory", "verification"
    )
    assert torch.equal(output, hidden)


def test_experiment2_has_no_sparse_router():
    bank = DenseAdapterBank(hidden_size=16, bottleneck=4)
    assert not any("router" in name.lower() for name, _ in bank.named_modules())


def test_full_sequence_loss_masks_prompt_and_trains_every_answer_token():
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    labels = full_sequence_labels(ids, torch.tensor([3]))
    assert labels.tolist() == [[-100, -100, -100, 6, 7]]
    logits = torch.randn(1, 5, 10, requires_grad=True)
    full_sequence_loss(logits, ids, torch.tensor([3])).backward()
    assert logits.grad[0, 2:4].abs().sum() > 0
    assert logits.grad[0, :2].abs().sum() == 0


def test_support_slots_are_distinct_not_all_zero():
    targets = distinct_support_slots(torch.tensor([[1, 0, 1, 1]], dtype=torch.float))
    assert targets.tolist() == [[0, -100, 1, 2]]


def test_span_pooling_rejects_bad_spans_and_propagates_gradients():
    pooler = AttentiveSpanPooler(hidden_size=8)
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    pooled = pooler(hidden, [[(1, 4)]])[0]
    pooled.sum().backward()
    assert hidden.grad[0, 1:4].abs().sum() > 0
    assert hidden.grad[0, :1].abs().sum() == 0


def test_episode_loader_keeps_stream_order_and_reset_boundary():
    loader = EpisodeStreamLoader([example("q", 1, False), example("e", 0, True)])
    stream = next(iter(loader))
    assert [item.example_id for item in stream] == ["e", "q"]
    assert stream[0].reset_episodic_memory
    assert not stream[1].reset_episodic_memory


def test_stream_runner_preserves_events_for_later_questions():
    stream = [example("event", 0, True), example("question", 1, False)]
    seen = []
    state = StreamRunner().run(
        stream, lambda item, current: seen.append((item.example_id, tuple(current.event_ids)))
    )
    assert seen == [("event", ("event",)), ("question", ("event",))]
    assert state.steps == 2


def test_stream_runner_rejects_midstream_reset():
    stream = [example("event", 0, True), example("question", 1, True)]
    try:
        StreamRunner().run(stream, lambda _item, _state: None)
    except ValueError as error:
        assert "stream boundary" in str(error)
    else:
        raise AssertionError("midstream reset should fail")


class TinyLayer(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1,)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer() for _ in range(4)])
        self.head = torch.nn.Linear(8, 11, bias=False)

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return self.head(hidden)


def test_dense_backbone_is_identity_and_frozen_at_initialization():
    reference = TinyBackbone()
    wrapped_source = TinyBackbone()
    wrapped_source.load_state_dict(reference.state_dict())
    model = DenseFrozenBackbone(
        wrapped_source, hidden_size=8, adapter_layer_indices=(0, 1, 2, 3)
    )
    hidden = torch.randn(2, 5, 8)
    assert torch.equal(model(hidden), reference(hidden))
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.adapters.parameters())
