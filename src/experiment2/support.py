from __future__ import annotations

import torch
from torch import nn


class FactScorer(nn.Module):
    """Fact/question interaction scorer from the Experiment 2 protocol."""

    def __init__(self, cognitive_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fact = nn.Linear(cognitive_dim, hidden_dim, bias=False)
        self.question = nn.Linear(cognitive_dim, hidden_dim, bias=False)
        self.product = nn.Linear(cognitive_dim, hidden_dim, bias=False)
        self.delta = nn.Linear(cognitive_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, facts: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        if facts.ndim != 3 or question.shape != facts.shape[:1] + facts.shape[2:]:
            raise ValueError("expected facts [batch,count,dim] and question [batch,dim]")
        query = question.unsqueeze(1)
        hidden = torch.tanh(
            self.fact(facts)
            + self.question(query)
            + self.product(facts * query)
            + self.delta((facts - query).abs())
        )
        return self.output(hidden).squeeze(-1)
