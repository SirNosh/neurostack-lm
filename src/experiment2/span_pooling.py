from __future__ import annotations

import torch
from torch import nn


class AttentiveSpanPooler(nn.Module):
    def __init__(self, hidden_size: int = 896) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, std=hidden_size**-0.5)

    def forward(
        self, hidden: torch.Tensor, spans: list[list[tuple[int, int]]]
    ) -> list[torch.Tensor]:
        pooled: list[torch.Tensor] = []
        for row, row_spans in enumerate(spans):
            values = []
            for start, end in row_spans:
                if not 0 <= start < end <= hidden.shape[1]:
                    raise ValueError("span is outside the token sequence")
                tokens = hidden[row, start:end]
                scores = torch.tanh(self.projection(tokens)) @ self.query
                values.append((scores.softmax(0).unsqueeze(-1) * tokens).sum(0))
            pooled.append(torch.stack(values) if values else hidden.new_zeros((0, hidden.shape[-1])))
        return pooled
