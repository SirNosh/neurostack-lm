from types import SimpleNamespace

import torch
from torch import nn


class ToyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor):
        return (torch.tanh(self.linear(hidden)),)


class ToyDecoder(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32, layers: int = 24):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [ToyDecoderLayer(hidden_size) for _ in range(layers)]
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
    ):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32):
        super().__init__()
        self.model = ToyDecoder(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def toy_inputs(batch: int = 2, length: int = 7):
    ids = torch.arange(batch * length).view(batch, length) % 64
    return ids, torch.ones_like(ids)

