from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .dense_adapters import DenseAdapterBank


QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ADAPTER_LAYER_INDICES = (5, 11, 17, 23)


class DenseFrozenBackbone(nn.Module):
    """Frozen causal LM with independent dense adapter banks at four layers."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int = 896,
        adapter_layer_indices: Sequence[int] = ADAPTER_LAYER_INDICES,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter_layer_indices = tuple(adapter_layer_indices)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.adapters = nn.ModuleList(
            [DenseAdapterBank(hidden_size) for _ in self.adapter_layer_indices]
        )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.backbone.model.layers

    def _install_hooks(self) -> None:
        if len(self.decoder_layers) <= max(self.adapter_layer_indices):
            raise ValueError("backbone does not expose the required adapter layers")
        for location, layer_index in enumerate(self.adapter_layer_indices):
            def hook(_module, _inputs, output, location=location):
                hidden = output[0] if isinstance(output, tuple) else output
                adapted = self.adapters[location](hidden)
                return (adapted, *output[1:]) if isinstance(output, tuple) else adapted

            self._handles.append(
                self.decoder_layers[layer_index].register_forward_hook(hook)
            )

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    @classmethod
    def from_qwen(
        cls,
        model_path: Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DenseFrozenBackbone":
        import transformers

        transformers.utils.is_flash_attn_2_available = lambda: False
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        model = cls(backbone)
        model.adapters.to(device=device, dtype=dtype)
        return model
