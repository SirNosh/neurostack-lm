from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .model import ADAPTER_LAYER_INDICES


class GenericAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.norm = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(0.05)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.dropout(
            self.up(F.silu(self.down(self.norm(hidden))))
        )


@dataclass
class BaselineOutput:
    token_logits: torch.Tensor
    pass_logits: list[torch.Tensor]
    pass_summaries: list[torch.Tensor]
    feedback_tokens: list[torch.Tensor]
    backbone_passes: int
    retrieval_indices: list[list[int]] | None = None


class R0ParameterMatchedAdapter(nn.Module):
    """Parameter-matched iterative baseline with generic feedback tokens."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        target_trainable_parameters: int,
        hidden_size: int = 896,
        adapter_layer_indices: Sequence[int] = ADAPTER_LAYER_INDICES,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.adapter_layer_indices = tuple(adapter_layer_indices)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if len(self.backbone.model.layers) <= max(self.adapter_layer_indices):
            raise ValueError("backbone does not expose the required adapter layers")

        self.feedback_token_count = 4
        self.feedback_projection = nn.Linear(
            hidden_size, self.feedback_token_count * hidden_size
        )
        feedback_parameters = sum(
            parameter.numel() for parameter in self.feedback_projection.parameters()
        )
        fixed_per_adapter = 2 * hidden_size
        per_bottleneck = 2 * hidden_size + 1
        total_units = round(
            (
                target_trainable_parameters
                - feedback_parameters
                - len(self.adapter_layer_indices) * fixed_per_adapter
            )
            / per_bottleneck
        )
        if total_units < len(self.adapter_layer_indices):
            raise ValueError("target parameter count is too small for four adapters")
        base, remainder = divmod(total_units, len(self.adapter_layer_indices))
        widths = [
            base + (index < remainder)
            for index in range(len(self.adapter_layer_indices))
        ]
        self.adapters = nn.ModuleList(
            [GenericAdapter(hidden_size, width) for width in widths]
        )
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        for adapter, layer_index in zip(self.adapters, self.adapter_layer_indices):
            def hook(_module, _inputs, output, adapter=adapter):
                hidden = output[0] if isinstance(output, tuple) else output
                adapted = adapter(hidden)
                if isinstance(output, tuple):
                    return (adapted, *output[1:])
                return adapted

            self._hook_handles.append(
                self.backbone.model.layers[layer_index].register_forward_hook(hook)
            )
        actual = self.trainable_parameter_count
        self.parameter_match_error = abs(actual - target_trainable_parameters) / max(
            1, target_trainable_parameters
        )
        if self.parameter_match_error > 0.02:
            raise ValueError("R0 trainable parameters are outside the +/-2% envelope")

    @classmethod
    def from_qwen(
        cls,
        model_path: Path,
        *,
        target_trainable_parameters: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "R0ParameterMatchedAdapter":
        import transformers

        transformers.utils.is_flash_attn_2_available = lambda: False
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        return cls(
            backbone,
            target_trainable_parameters=target_trainable_parameters,
        ).to(device=device, dtype=dtype)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def total_active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def adapter_matmul_flops(
        self, *, batch_size: int, tokens: int, passes: int
    ) -> int:
        processed_tokens = tokens * passes + self.feedback_token_count * (passes - 1)
        return (
            4
            * batch_size
            * processed_tokens
            * self.hidden_size
            * sum(adapter.bottleneck for adapter in self.adapters)
        )

    def feedback_matmul_flops(self, *, batch_size: int, passes: int) -> int:
        return (
            2
            * batch_size
            * self.hidden_size
            * self.feedback_token_count
            * self.hidden_size
            * max(0, passes - 1)
        )

    def mechanism_matmul_flops(
        self, *, batch_size: int, tokens: int, passes: int
    ) -> int:
        return self.adapter_matmul_flops(
            batch_size=batch_size, tokens=tokens, passes=passes
        ) + self.feedback_matmul_flops(batch_size=batch_size, passes=passes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        passes: int = 1,
        feedback_enabled: bool = True,
    ) -> BaselineOutput:
        if not 1 <= passes <= 3:
            raise ValueError("R0 permits one to three backbone passes")
        outputs: list[torch.Tensor] = []
        summaries: list[torch.Tensor] = []
        feedback_tokens: list[torch.Tensor] = []
        original_embeddings = self.backbone.model.embed_tokens(input_ids)
        current_embeddings = original_embeddings
        current_mask = attention_mask
        for pass_index in range(passes):
            hidden = self.backbone.model(
                inputs_embeds=current_embeddings,
                attention_mask=current_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            outputs.append(self.backbone.lm_head(hidden[:, -1]))
            mask = current_mask.to(hidden.dtype).unsqueeze(-1)
            summary = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            summaries.append(summary)
            if pass_index + 1 < passes:
                projected = self.feedback_projection(summary).view(
                    hidden.shape[0], self.feedback_token_count, self.hidden_size
                )
                feedback = projected if feedback_enabled else torch.zeros_like(projected)
                feedback_tokens.append(feedback)
                current_embeddings = torch.cat(
                    (feedback, original_embeddings), dim=1
                )
                prefix_mask = torch.ones(
                    attention_mask.shape[0],
                    self.feedback_token_count,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                current_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        return BaselineOutput(
            outputs[-1], outputs, summaries, feedback_tokens, passes
        )

    def set_sleep_mode(self) -> None:
        self.train()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapters.parameters():
            parameter.requires_grad_(True)
        for parameter in self.feedback_projection.parameters():
            parameter.requires_grad_(True)

    def set_evaluation_mode(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class R1OrdinaryRAG(R0ParameterMatchedAdapter):
    """Parameter-matched generic adapters plus session-scoped ordinary RAG."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        target_trainable_parameters: int,
        hidden_size: int = 896,
        adapter_layer_indices: Sequence[int] = ADAPTER_LAYER_INDICES,
        capacity: int = 8192,
        top_k: int = 4,
    ) -> None:
        if capacity != 8192 or top_k != 4:
            raise ValueError("R1 is frozen at 8192 entries and top-4 retrieval")
        super().__init__(
            backbone,
            target_trainable_parameters=target_trainable_parameters,
            hidden_size=hidden_size,
            adapter_layer_indices=adapter_layer_indices,
        )
        self.capacity = capacity
        self.top_k = top_k
        self._keys: dict[str, list[torch.Tensor]] = {}
        self._values: dict[str, list[torch.Tensor]] = {}

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._keys.clear()
            self._values.clear()
        else:
            self._keys.pop(session_id, None)
            self._values.pop(session_id, None)

    def _retrieve(
        self, query: torch.Tensor, session_ids: Sequence[str]
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        retrieved = []
        all_indices = []
        for row, session_id in enumerate(session_ids):
            keys = self._keys.get(session_id, [])
            values = self._values.get(session_id, [])
            if not keys:
                retrieved.append(query.new_zeros((0, self.hidden_size)))
                all_indices.append([])
                continue
            key_tensor = torch.stack(keys).to(query)
            scores = F.cosine_similarity(query[row : row + 1], key_tensor, dim=-1)
            indices = scores.topk(min(self.top_k, len(keys))).indices.tolist()
            retrieved.append(torch.stack([values[index] for index in indices]).to(query))
            all_indices.append(indices)
        return retrieved, all_indices

    @torch.no_grad()
    def _write(
        self, summaries: torch.Tensor, session_ids: Sequence[str]
    ) -> None:
        for row, session_id in enumerate(session_ids):
            keys = self._keys.setdefault(session_id, [])
            values = self._values.setdefault(session_id, [])
            value = summaries[row].detach().float().cpu()
            keys.append(value)
            values.append(value)
            if len(keys) > self.capacity:
                del keys[0]
                del values[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        session_ids: Sequence[str],
        passes: int = 1,
        write: bool = True,
    ) -> BaselineOutput:
        if len(session_ids) != input_ids.shape[0]:
            raise ValueError("one R1 session ID is required per batch row")
        if not 1 <= passes <= 3:
            raise ValueError("R1 permits one to three backbone passes")
        original = self.backbone.model.embed_tokens(input_ids)
        query_mask = attention_mask.to(original.dtype).unsqueeze(-1)
        query = (original * query_mask).sum(1) / query_mask.sum(1).clamp_min(1)
        retrieved, retrieval_indices = self._retrieve(query, session_ids)
        max_retrieved = max((len(items) for items in retrieved), default=0)
        if max_retrieved:
            prefix = original.new_zeros(
                input_ids.shape[0], max_retrieved, self.hidden_size
            )
            prefix_mask = attention_mask.new_zeros(
                input_ids.shape[0], max_retrieved
            )
            for row, items in enumerate(retrieved):
                prefix[row, : len(items)] = items
                prefix_mask[row, : len(items)] = 1
            embeddings = torch.cat((prefix, original), dim=1)
            mask = torch.cat((prefix_mask, attention_mask), dim=1)
        else:
            embeddings, mask = original, attention_mask
        logits: list[torch.Tensor] = []
        summaries: list[torch.Tensor] = []
        feedback_tokens: list[torch.Tensor] = []
        for pass_index in range(passes):
            hidden = self.backbone.model(
                inputs_embeds=embeddings,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            logits.append(self.backbone.lm_head(hidden[:, -1]))
            float_mask = mask.to(hidden.dtype).unsqueeze(-1)
            summaries.append(
                (hidden * float_mask).sum(1) / float_mask.sum(1).clamp_min(1)
            )
            if pass_index + 1 < passes:
                feedback = self.feedback_projection(summaries[-1]).view(
                    input_ids.shape[0], self.feedback_token_count, self.hidden_size
                )
                feedback_tokens.append(feedback)
                feedback_mask = attention_mask.new_ones(
                    input_ids.shape[0], self.feedback_token_count
                )
                embeddings = torch.cat((feedback, embeddings), dim=1)
                mask = torch.cat((feedback_mask, mask), dim=1)
        if write:
            self._write(summaries[-1], session_ids)
        return BaselineOutput(
            logits[-1], logits, summaries, feedback_tokens, passes, retrieval_indices
        )


class R2RecurrentMemoryTokens(R0ParameterMatchedAdapter):
    """Parameter-matched recurrent baseline with exactly 16 memory tokens."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        target_trainable_parameters: int,
        hidden_size: int = 896,
        adapter_layer_indices: Sequence[int] = ADAPTER_LAYER_INDICES,
        memory_token_count: int = 16,
    ) -> None:
        if memory_token_count != 16:
            raise ValueError("R2 is frozen at exactly 16 memory tokens")
        memory_parameters = memory_token_count * hidden_size
        super().__init__(
            backbone,
            target_trainable_parameters=target_trainable_parameters - memory_parameters,
            hidden_size=hidden_size,
            adapter_layer_indices=adapter_layer_indices,
        )
        self.memory_token_count = memory_token_count
        self.memory_tokens = nn.Parameter(
            torch.empty(memory_token_count, hidden_size)
        )
        nn.init.normal_(self.memory_tokens, std=0.02)
        self.parameter_match_error = abs(
            self.trainable_parameter_count - target_trainable_parameters
        ) / max(1, target_trainable_parameters)
        if self.parameter_match_error > 0.02:
            raise ValueError("R2 trainable parameters are outside the +/-2% envelope")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        passes: int = 1,
    ) -> BaselineOutput:
        if not 1 <= passes <= 3:
            raise ValueError("R2 permits one to three backbone passes")
        original = self.backbone.model.embed_tokens(input_ids)
        memory = self.memory_tokens.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        prefix_mask = attention_mask.new_ones(
            input_ids.shape[0], self.memory_token_count
        )
        mask = torch.cat((prefix_mask, attention_mask), dim=1)
        logits: list[torch.Tensor] = []
        summaries: list[torch.Tensor] = []
        feedback: list[torch.Tensor] = []
        for pass_index in range(passes):
            embeddings = torch.cat((memory, original), dim=1)
            hidden = self.backbone.model(
                inputs_embeds=embeddings,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            logits.append(self.backbone.lm_head(hidden[:, -1]))
            float_mask = mask.to(hidden.dtype).unsqueeze(-1)
            summaries.append(
                (hidden * float_mask).sum(1) / float_mask.sum(1).clamp_min(1)
            )
            if pass_index + 1 < passes:
                memory = hidden[:, : self.memory_token_count]
                feedback.append(memory)
        return BaselineOutput(logits[-1], logits, summaries, feedback, passes)
