from types import SimpleNamespace

import torch

from src.experiment2.a1 import A1SupportWorkingMemoryModel
from src.experiment2.babi import parse_babi
from src.experiment2.model import DenseFrozenBackbone
from src.experiment2.tokenization import collate_tokenized, tokenize_example


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        result = {"input_ids": [ord(character) % 31 for character in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class TinyLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden):
        return (hidden + torch.tanh(self.linear(hidden)),)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, hidden_size=16, vocab=32):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [TinyLayer(hidden_size) for _ in range(4)]
        )
        self.embedding = torch.nn.Embedding(vocab, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
            states.append(hidden)
        return SimpleNamespace(logits=self.head(hidden), hidden_states=tuple(states))


def test_a1_two_cycle_loss_is_finite_and_memory_lesion_changes_logits(tmp_path):
    source = tmp_path / "qa1_test_train.txt"
    source.write_text(
        "1 Mary went home.\n2 John stayed outside.\n"
        "3 Where is Mary?\thome\t1\n",
        encoding="utf-8",
    )
    tokenized = [tokenize_example(parse_babi(source)[0], CharacterTokenizer())]
    batch = collate_tokenized(tokenized, pad_token_id=0)
    dense = DenseFrozenBackbone(
        TinyCausalLM(), hidden_size=16, adapter_layer_indices=(0, 1, 2, 3)
    )
    model = A1SupportWorkingMemoryModel(dense, cognitive_dim=8)
    output = model(batch, tokenized)
    assert torch.isfinite(output.loss)
    assert not torch.equal(output.answer_logits, output.lesioned_answer_logits)
    output.loss.backward()
    assert dense.adapters[0].branches["relational"].up.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for bank in dense.adapters
        for name, branch in bank.branches.items()
        if name != "relational"
        for parameter in branch.parameters()
    )
