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
