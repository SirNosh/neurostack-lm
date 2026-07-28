from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .losses import fact_support_loss, sequence_nll_from_logits
from .model import DenseFrozenBackbone
from .span_pooling import AttentiveSpanPooler
from .support import FactScorer
from .tokenization import TokenizedExperiment2Example
from .working_memory import BootstrapWorkingMemory, working_memory_use_loss


@dataclass
class A1Output:
    loss: torch.Tensor
    sequence_nll: torch.Tensor
    lesioned_sequence_nll: torch.Tensor
    fact_logits: torch.Tensor
    fact_mask: torch.Tensor
    support_mask: torch.Tensor
    operation_logits: torch.Tensor
    address_logits: torch.Tensor
    answer_logits: torch.Tensor
    lesioned_answer_logits: torch.Tensor
    memory_read_weights: torch.Tensor


class A1SupportWorkingMemoryModel(nn.Module):
    def __init__(
        self, dense_backbone: DenseFrozenBackbone, *, cognitive_dim: int = 256
    ) -> None:
        super().__init__()
        self.dense_backbone = dense_backbone
        for bank in self.dense_backbone.adapters:
            bank.train_only("relational")
        hidden_size = self.dense_backbone.hidden_size
        self.fact_pooler = AttentiveSpanPooler(hidden_size)
        self.question_pooler = AttentiveSpanPooler(hidden_size)
        self.fact_projection = nn.Linear(hidden_size, cognitive_dim)
        self.question_projection = nn.Linear(hidden_size, cognitive_dim)
        self.fact_scorer = FactScorer(cognitive_dim)
        self.token_support = nn.Linear(hidden_size, 1)
        self.working_memory = BootstrapWorkingMemory(
            cognitive_dim=cognitive_dim, key_dim=max(4, cognitive_dim // 4)
        )
        self.memory_prefix = nn.Linear(cognitive_dim, hidden_size)

    def _pool(
        self, hidden: torch.Tensor, examples: list[TokenizedExperiment2Example]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fact_rows = self.fact_pooler(
            hidden, [item.fact_token_spans for item in examples]
        )
        question_rows = self.question_pooler(
            hidden, [[item.question_token_span] for item in examples]
        )
        max_facts = max(row.shape[0] for row in fact_rows)
        facts = hidden.new_zeros((len(examples), max_facts, hidden.shape[-1]))
        fact_mask = torch.zeros(
            (len(examples), max_facts), device=hidden.device, dtype=torch.bool
        )
        support_mask = torch.zeros_like(fact_mask)
        for row, (values, item) in enumerate(zip(fact_rows, examples)):
            facts[row, : values.shape[0]] = values
            fact_mask[row, : values.shape[0]] = True
            support_mask[row, item.example.support_fact_indices] = True
        questions = torch.stack([row[0] for row in question_rows])
        return facts, questions, fact_mask, support_mask

    def _cycle(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix: torch.Tensor,
    ):
        embeddings = self.dense_backbone.backbone.get_input_embeddings()(input_ids)
        inputs = torch.cat([prefix, embeddings], dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype
        )
        return self.dense_backbone(
            inputs_embeds=inputs,
            attention_mask=torch.cat([prefix_mask, attention_mask], dim=1),
            output_hidden_states=True,
            use_cache=False,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        examples: list[TokenizedExperiment2Example],
    ) -> A1Output:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lengths = batch["prompt_lengths"]
        first = self.dense_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = first.hidden_states[-1]
        facts, questions, fact_mask, support_mask = self._pool(hidden, examples)
        fact_values = self.fact_projection(facts)
        question_value = self.question_projection(questions)
        fact_logits = self.fact_scorer(fact_values, question_value).masked_fill(
            ~fact_mask, -1e4
        )
        memory, operation_logits, address_logits = self.working_memory.write(
            fact_values, support_mask
        )
        _, read_weights = self.working_memory.read(memory, question_value)
        prefix = self.memory_prefix(memory.values) * memory.occupied.unsqueeze(-1)
        full = self._cycle(input_ids, attention_mask, prefix)
        lesioned = self._cycle(input_ids, attention_mask, torch.zeros_like(prefix))
        prefix_length = prefix.shape[1]
        sequence_nll = sequence_nll_from_logits(
            full.logits, input_ids, prompt_lengths, prefix_length=prefix_length
        )
        lesioned_nll = sequence_nll_from_logits(
            lesioned.logits, input_ids, prompt_lengths, prefix_length=prefix_length
        )

        support_loss = fact_support_loss(
            fact_logits, support_mask.float(), valid_mask=fact_mask
        )
        operation_targets, address_targets = self.working_memory.targets(support_mask)
        operation_loss = F.cross_entropy(
            operation_logits[fact_mask], operation_targets[fact_mask]
        )
        address_loss = F.cross_entropy(
            address_logits.reshape(-1, address_logits.shape[-1]),
            address_targets.reshape(-1),
            ignore_index=-100,
        )
        token_targets = torch.zeros_like(attention_mask, dtype=hidden.dtype)
        for row, item in enumerate(examples):
            for start, end in (
                item.fact_token_spans[index]
                for index in item.example.support_fact_indices
            ):
                token_targets[row, start:end] = 1
        token_loss = F.binary_cross_entropy_with_logits(
            self.token_support(hidden).squeeze(-1)[attention_mask.bool()],
            token_targets[attention_mask.bool()],
        )
        total = (
            sequence_nll.mean()
            + 0.5 * (support_loss + 0.1 * token_loss)
            + 0.2 * operation_loss
            + 0.2 * address_loss
            + 0.2 * working_memory_use_loss(sequence_nll, lesioned_nll)
        )
        return A1Output(
            total,
            sequence_nll,
            lesioned_nll,
            fact_logits,
            fact_mask,
            support_mask,
            operation_logits,
            address_logits,
            full.logits,
            lesioned.logits,
            read_weights,
        )
