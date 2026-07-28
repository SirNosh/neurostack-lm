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
from src.experiment2.babi import parse_babi
from src.experiment2.support import FactScorer
from src.experiment2.working_memory import BootstrapWorkingMemory


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


def test_babi_adapter_exposes_all_facts_and_fact_level_support(tmp_path):
    source = tmp_path / "qa1_test_train.txt"
    source.write_text(
        "1 Mary went to the kitchen.\n"
        "2 John went to the office.\n"
        "3 Where is Mary?\tkitchen\t1\n",
        encoding="utf-8",
    )
    item = parse_babi(source)[0]
    assert len(item.fact_spans) == 2
    assert item.support_fact_indices == [0]
    assert item.input_text[item.fact_spans[1][0]:item.fact_spans[1][1]] == (
        "John went to the office."
    )


def test_support_gradients_reach_only_relational_dense_branch():
    bank = DenseAdapterBank(hidden_size=16, bottleneck=4)
    bank.train_only("relational")
    scorer = FactScorer(cognitive_dim=16, hidden_dim=8)
    hidden = bank(torch.randn(2, 3, 16))
    scorer(hidden[:, :2], hidden[:, 2]).sum().backward()
    assert bank.branches["relational"].up.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, branch in bank.branches.items()
        if name != "relational"
        for parameter in branch.parameters()
    )


def test_working_memory_targets_distinct_slots_and_lesion_changes_answer_logits():
    memory = BootstrapWorkingMemory(cognitive_dim=8, key_dim=4)
    facts = torch.randn(1, 3, 8)
    question = torch.randn(1, 8)
    support = torch.tensor([[1, 0, 1]], dtype=torch.float)
    state, operation_logits, address_logits = memory.write(facts, support)
    operations, addresses = memory.targets(support)
    assert operations.tolist() == [[1, 0, 1]]
    assert addresses.tolist() == [[0, -100, 1]]
    assert operation_logits.shape == (1, 3, 5)
    assert address_logits.shape == (1, 3, 8)
    read, _ = memory.read(state, question)
    lesioned, _ = memory.read(state, question, lesion=True)
    decoder = torch.nn.Linear(8, 11, bias=False)
    assert not torch.equal(decoder(question + read), decoder(question + lesioned))
