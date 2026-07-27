from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .mechanisms import (
    CognitiveState,
    ControlValues,
    EpisodicEvent,
    EpisodicMemory,
    FastWeightBank,
    LesionConfig,
    MemoryOperation,
    ModulatorController,
    ModulatorSignals,
    PersistentPFC,
    RetrievalResult,
    RoutingResult,
    SparseRouter,
    Verifier,
    WorkingMemory,
    Workspace,
    WorkspaceState,
)


QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ADAPTER_LAYER_INDICES = (5, 11, 17, 23)


class AdapterExpert(nn.Module):
    """896 -> 128 -> 896 residual expert from the fixed Stage 1R protocol."""

    def __init__(self, hidden_size: int = 896, bottleneck: int = 128) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(0.05)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.up(F.silu(self.down(self.norm(hidden)))))


class RoutedAdapterBank(nn.Module):
    def __init__(self, hidden_size: int = 896, experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [AdapterExpert(hidden_size) for _ in range(experts)]
        )
        self.summary_projection = nn.Linear(hidden_size, 256)
        self.last_summaries: torch.Tensor | None = None

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=2)
        gather_index = indices[:, None, :, None].expand(
            -1, hidden.shape[1], -1, hidden.shape[2]
        )
        selected = torch.gather(expert_outputs, 2, gather_index)
        mixture = (selected * weights[:, None, :, None]).sum(dim=2)
        self.last_summaries = self.summary_projection(expert_outputs.mean(dim=1))
        return hidden + mixture


@dataclass
class CycleOutput:
    token_logits: torch.Tensor
    hidden_summary: torch.Tensor
    state: CognitiveState
    workspace: WorkspaceState
    broadcast_tokens: torch.Tensor | None
    routing: RoutingResult
    routing_input: torch.Tensor
    retrieval: RetrievalResult
    modulators: ModulatorSignals
    controls: ControlValues
    verifier_logits: torch.Tensor
    action_logits: torch.Tensor
    appraisal: torch.Tensor
    pfc_input: torch.Tensor
    working_query: torch.Tensor
    working_operation_logits: torch.Tensor
    working_slot_logits: torch.Tensor
    working_write_key: torch.Tensor
    working_write_value: torch.Tensor
    retrieval_summary: torch.Tensor


@dataclass
class LifetimeOutput:
    cycles: list[CycleOutput]

    @property
    def final(self) -> CycleOutput:
        return self.cycles[-1]


class Stage1RNeuroStack(nn.Module):
    """Token-level Stage 1R model around a frozen causal-language backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int = 896,
        adapter_layer_indices: Sequence[int] = ADAPTER_LAYER_INDICES,
        differentiated_modulators: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.adapter_layer_indices = tuple(adapter_layer_indices)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.adapters = nn.ModuleList(
            [RoutedAdapterBank(hidden_size) for _ in self.adapter_layer_indices]
        )
        self.token_projection = nn.Linear(hidden_size, 256)
        self.token_key = nn.Linear(256, 64)
        self.episodic_key = nn.Linear(256, 256)
        self.retrieval_integration = nn.Linear(256, 256)
        self.pfc = PersistentPFC()
        self.pfc_gate = nn.Linear(768, 4)
        self.working_memory = WorkingMemory()
        self.working_operation = nn.Linear(768, len(MemoryOperation))
        self.working_slot = nn.Linear(768, 8)
        self.working_key = nn.Linear(768, 64)
        self.working_value = nn.Linear(768, 256)
        self.workspace = Workspace(broadcast_dim=hidden_size)
        self.router = SparseRouter(768)
        self.modulators = ModulatorController(
            768, differentiated=differentiated_modulators
        )
        self.verifier = Verifier(768)
        self.action_head = nn.Linear(768, 3)
        self.appraisal = nn.Sequential(
            nn.Linear(768, 256), nn.SiLU(), nn.Linear(256, 6)
        )
        self.verifier_candidate = nn.Linear(1, 256)
        self.appraisal_candidate = nn.Linear(6, 256)
        self.fast_weights = FastWeightBank(rank=8)
        self._active_routing: RoutingResult | None = None
        self._active_routing_input: torch.Tensor | None = None
        self._routing_state: CognitiveState | None = None
        self._routing_lesions = LesionConfig()
        self._routing_mask: torch.Tensor | None = None
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._install_adapter_hooks()

    @classmethod
    def from_qwen(
        cls,
        model_path: Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        differentiated_modulators: bool = True,
    ) -> "Stage1RNeuroStack":
        import transformers

        transformers.utils.is_flash_attn_2_available = lambda: False
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        model = cls(
            backbone, differentiated_modulators=differentiated_modulators
        ).to(device=device, dtype=dtype)
        return model

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.backbone.model.layers

    def _install_adapter_hooks(self) -> None:
        if len(self.decoder_layers) <= max(self.adapter_layer_indices):
            raise ValueError("backbone does not expose the required adapter layers")
        if self.adapter_layer_indices[0] <= 4:
            raise ValueError("the first adapter must follow the contextual routing prefix")

        def early_router_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self._routing_state is None or self._routing_mask is None:
                raise RuntimeError("routing context was not initialized")
            mask = self._routing_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            current_token = self.token_projection(pooled)
            pfc = self._routing_state.pfc.slots.mean(1)
            occupied = self._routing_state.working_memory.occupied.unsqueeze(-1)
            working = (
                self._routing_state.working_memory.values * occupied
            ).sum(1) / occupied.sum(1).clamp_min(1)
            router_input = torch.cat([current_token, pfc, working], dim=-1)
            signals = self.modulators(router_input, self._routing_state.overload)
            if not self._routing_lesions.modulators:
                neutral = torch.full_like(signals.ne, 0.5)
                signals = ModulatorSignals(
                    da=torch.zeros_like(signals.da),
                    ne=neutral,
                    ach=neutral,
                    serotonin=neutral,
                    overload=torch.zeros_like(signals.overload),
                )
            controls = self.modulators.controls(signals)
            fast_hidden = (
                self.fast_weights.router_query.apply(
                    current_token, self._routing_state.fast_weights.router_query
                )
                if self._routing_lesions.fast_weights
                else None
            )
            routing = self.router(
                router_input,
                temperature=(
                    controls.router_temperature
                    if self._routing_lesions.modulators
                    else 1.0
                ),
                fast_hidden=fast_hidden,
            )
            if not self._routing_lesions.routing:
                uniform = torch.full_like(routing.probabilities, 0.25)
                indices = torch.tensor([0, 1], device=hidden.device).expand(
                    *uniform.shape[:-1], 2
                )
                routing = RoutingResult(
                    torch.zeros_like(routing.logits),
                    uniform,
                    indices,
                    torch.full_like(indices, 0.5, dtype=uniform.dtype),
                )
            self._active_routing = routing
            self._active_routing_input = router_input
            return output

        self._hook_handles.append(
            self.decoder_layers[4].register_forward_hook(early_router_hook)
        )
        for location, layer_index in enumerate(self.adapter_layer_indices):
            def hook(_module, _inputs, output, location=location):
                if self._active_routing is None:
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                adapted = self.adapters[location](
                    hidden,
                    self._active_routing.indices[:, location],
                    self._active_routing.weights[:, location],
                )
                if isinstance(output, tuple):
                    return (adapted, *output[1:])
                return adapted

            self._hook_handles.append(
                self.decoder_layers[layer_index].register_forward_hook(hook)
            )

    def initialize_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CognitiveState:
        return CognitiveState(
            pfc=self.pfc.initialize(batch_size, device=device, dtype=dtype),
            working_memory=self.working_memory.initialize(
                batch_size, device=device, dtype=dtype
            ),
            fast_weights=self.fast_weights.initialize(
                batch_size, device=device, dtype=dtype
            ),
            overload=torch.zeros(batch_size, device=device, dtype=dtype),
        )

    def reset_state(self, state: CognitiveState, mask: torch.Tensor) -> CognitiveState:
        fast = self.fast_weights.initialize(
            state.pfc.slots.shape[0],
            device=state.pfc.slots.device,
            dtype=state.pfc.slots.dtype,
        )
        if not mask.all():
            keep = (~mask.bool()).view(-1, 1, 1)
            fast = type(fast)(
                *[
                    type(current)(
                        torch.where(keep, current.u, fresh.u),
                        torch.where(keep, current.v, fresh.v),
                    )
                    for current, fresh in zip(
                        (
                            state.fast_weights.pfc,
                            state.fast_weights.working_read,
                            state.fast_weights.retrieval,
                            state.fast_weights.router_query,
                        ),
                        (fast.pfc, fast.working_read, fast.retrieval, fast.router_query),
                    )
                ]
            )
        overload = state.overload.clone()
        overload[mask.bool()] = 0
        return CognitiveState(
            self.pfc.reset(state.pfc, mask),
            self.working_memory.reset(state.working_memory, mask),
            fast,
            overload,
            0,
        )

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        broadcast: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if broadcast is None:
            self._routing_mask = attention_mask
            output = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            token_mask = attention_mask
        else:
            token_embeddings = self.backbone.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([broadcast, token_embeddings], dim=1)
            prefix_mask = torch.ones(
                broadcast.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype
            )
            token_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            self._routing_mask = token_mask
            output = self.backbone.model(
                inputs_embeds=inputs_embeds,
                attention_mask=token_mask,
                use_cache=False,
                return_dict=True,
            )
        hidden = output.last_hidden_state
        pooled = (hidden * token_mask.unsqueeze(-1)).sum(1) / token_mask.sum(
            1, keepdim=True
        )
        logits = self.backbone.lm_head(hidden[:, -1])
        return pooled, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: CognitiveState,
        episodic_memory: EpisodicMemory,
        *,
        session_ids: Sequence[str],
        task_contexts: Sequence[str],
        cycles: int = 3,
        lesions: LesionConfig = LesionConfig(),
    ) -> LifetimeOutput:
        if not 1 <= cycles <= 3:
            raise ValueError("Stage 1R permits one to three cognitive cycles")
        outputs: list[CycleOutput] = []
        broadcast = None
        for cycle in range(cycles):
            self._active_routing = None
            self._active_routing_input = None
            self._routing_state = state
            self._routing_lesions = lesions
            pooled, token_logits = self._run_backbone(
                input_ids, attention_mask, broadcast
            )
            if self._active_routing is None or self._active_routing_input is None:
                raise RuntimeError("early-layer routing hook did not run")
            routing = self._active_routing
            router_input = self._active_routing_input
            token = self.token_projection(pooled)
            working_query = self.token_key(token)
            if lesions.working_memory:
                working_read, _ = self.working_memory.read(
                    state.working_memory, working_query
                )
                if lesions.fast_weights:
                    working_read = working_read + self.fast_weights.working_read.apply(
                        working_query, state.fast_weights.working_read
                    )
            else:
                working_read = torch.zeros_like(token)
            pfc_summary = (
                state.pfc.slots.mean(1) if lesions.pfc else torch.zeros_like(token)
            )
            controller_input = torch.cat([token, pfc_summary, working_read], dim=-1)
            signals = self.modulators(controller_input, state.overload)
            if not lesions.modulators:
                neutral = torch.full_like(signals.ne, 0.5)
                signals = ModulatorSignals(
                    da=torch.zeros_like(signals.da),
                    ne=neutral,
                    ach=neutral,
                    serotonin=neutral,
                    overload=torch.zeros_like(signals.overload),
                )
            controls = self.modulators.controls(signals)
            query_key = self.episodic_key(token)
            if lesions.episodic:
                retrieval = episodic_memory.retrieve(
                    query_key,
                    session_ids=session_ids,
                    task_contexts=task_contexts,
                    breadths=controls.retrieval_breadth,
                )
            else:
                empty_memory = EpisodicMemory()
                retrieval = empty_memory.retrieve(
                    query_key,
                    session_ids=session_ids,
                    breadths=controls.retrieval_breadth,
                )
            retrieval_weights = F.softmax(retrieval.scores, dim=-1)
            retrieval_weights = torch.where(
                retrieval.scores > -1e3,
                retrieval_weights,
                torch.zeros_like(retrieval_weights),
            ).to(retrieval.values.dtype)
            retrieval_summary = torch.einsum(
                "bk,bkd->bd", retrieval_weights, retrieval.values
            )
            retrieval_summary = self.retrieval_integration(retrieval_summary)
            if lesions.fast_weights:
                retrieval_summary = retrieval_summary + self.fast_weights.retrieval.apply(
                    retrieval_summary, state.fast_weights.retrieval
                )
            retrieval_summary = retrieval_summary * controls.retrieve_weight.unsqueeze(-1)
            pfc_input = torch.cat(
                [token, pfc_summary, working_read, retrieval_summary, token], dim=-1
            )
            update_gates = self.pfc_gate(controller_input).sigmoid()
            update_gates = update_gates * (
                1 - 0.5 * controls.conflict_pressure.unsqueeze(-1)
            )
            new_pfc = (
                self.pfc.update(pfc_input, state.pfc, update_gates)
                if lesions.pfc
                else state.pfc
            )
            if lesions.pfc and lesions.fast_weights:
                correction = self.fast_weights.pfc.apply(
                    pfc_input, state.fast_weights.pfc
                )
                new_pfc.slots = new_pfc.slots + correction.unsqueeze(1)
            if lesions.pfc:
                new_pfc.slots[:, 2] = new_pfc.slots[:, 2] * (
                    1 - controls.strategy_reset.unsqueeze(-1)
                )

            operation_logits = self.working_operation(controller_input)
            slot_logits = self.working_slot(controller_input)
            if self.training:
                operation_sample = F.gumbel_softmax(
                    operation_logits, tau=1.0, hard=True
                )
                slot_sample = F.gumbel_softmax(slot_logits, tau=1.0, hard=True)
            else:
                operation_sample = F.one_hot(
                    operation_logits.argmax(-1), len(MemoryOperation)
                ).to(operation_logits.dtype)
                slot_sample = F.one_hot(slot_logits.argmax(-1), 8).to(slot_logits.dtype)
            write_key = self.working_key(controller_input)
            write_value = self.working_value(controller_input)
            if lesions.working_memory:
                new_working = self.working_memory.operate_differentiable(
                    state.working_memory,
                    operation_sample,
                    slot_sample,
                    key=write_key,
                    value=write_value,
                    confidence=controls.memory_write,
                )
            else:
                new_working = state.working_memory

            verifier_logits = (
                self.verifier(controller_input)
                if lesions.verifier
                else torch.zeros(input_ids.shape[0], device=token.device, dtype=token.dtype)
            )
            action_logits = self.action_head(controller_input)
            action_logits = action_logits.clone()
            action_logits[:, 0] += controls.continue_bias
            action_logits[:, 2] += (
                controls.abstain_threshold - 0.5
            )
            appraisal = self.appraisal(controller_input).sigmoid()
            specialist_summaries = [
                bank.last_summaries
                if bank.last_summaries is not None
                else torch.zeros(
                    input_ids.shape[0], 4, 256, device=token.device, dtype=token.dtype
                )
                for bank in self.adapters
            ]
            specialist = torch.stack(specialist_summaries).mean(0)
            workspace = self.workspace.compete(
                [
                    ("token", token.unsqueeze(1)),
                    ("pfc", new_pfc.slots),
                    ("working_memory", new_working.values),
                    ("episodic", retrieval.values),
                    ("specialist", specialist),
                    (
                        "verifier",
                        self.verifier_candidate(verifier_logits.unsqueeze(-1)).unsqueeze(1),
                    ),
                    (
                        "appraisal",
                        self.appraisal_candidate(appraisal).unsqueeze(1),
                    ),
                ],
                masks=[
                    torch.ones(input_ids.shape[0], 1, device=token.device, dtype=torch.bool),
                    torch.ones(input_ids.shape[0], 4, device=token.device, dtype=torch.bool),
                    new_working.occupied,
                    retrieval.scores > -1e3,
                    torch.ones(input_ids.shape[0], 4, device=token.device, dtype=torch.bool),
                    torch.ones(input_ids.shape[0], 1, device=token.device, dtype=torch.bool),
                    torch.ones(input_ids.shape[0], 1, device=token.device, dtype=torch.bool),
                ],
            )
            broadcast = (
                self.workspace.broadcast(workspace) if lesions.workspace else None
            )
            state = CognitiveState(
                new_pfc,
                new_working,
                state.fast_weights,
                signals.overload,
                state.cycle + 1,
            )
            outputs.append(
                CycleOutput(
                    token_logits,
                    token,
                    state,
                    workspace,
                    broadcast,
                    routing,
                    router_input,
                    retrieval,
                    signals,
                    controls,
                    verifier_logits,
                    action_logits,
                    appraisal,
                    pfc_input,
                    working_query,
                    operation_logits,
                    slot_logits,
                    write_key,
                    write_value,
                    retrieval_summary,
                )
            )
        self._active_routing = None
        self._active_routing_input = None
        self._routing_state = None
        return LifetimeOutput(outputs)

    @torch.no_grad()
    def apply_wake_feedback(
        self,
        output: CycleOutput,
        *,
        outcome: torch.Tensor,
        episodic_memory: EpisodicMemory,
        session_ids: Sequence[str],
        task_contexts: Sequence[str],
        timestamps: Sequence[int],
        provenances: Sequence[str],
        encode_targets: torch.Tensor | None = None,
        bootstrap_mode: bool = False,
        write_threshold: float = 0.5,
        lesions: LesionConfig = LesionConfig(),
    ) -> CognitiveState:
        state = output.state
        da = (output.modulators.da + outcome.sign()).clamp(-1, 1) / 2
        ach = output.modulators.ach
        ne = output.modulators.ne
        fast = state.fast_weights
        if lesions.fast_weights:
            fast = type(fast)(
                self.fast_weights.pfc.update(
                    fast.pfc,
                    output.pfc_input,
                    output.hidden_summary,
                    da=da,
                    ach=ach,
                    ne=ne,
                ),
                self.fast_weights.working_read.update(
                    fast.working_read,
                    output.working_query,
                    output.hidden_summary,
                    da=da,
                    ach=ach,
                    ne=ne,
                ),
                self.fast_weights.retrieval.update(
                    fast.retrieval,
                    output.retrieval_summary,
                    output.hidden_summary,
                    da=da,
                    ach=ach,
                    ne=ne,
                ),
                self.fast_weights.router_query.update(
                    fast.router_query,
                    output.hidden_summary,
                    torch.cat(
                        [output.hidden_summary, output.hidden_summary], dim=-1
                    ),
                    da=da,
                    ach=ach,
                    ne=ne,
                ),
            )
        if bootstrap_mode:
            if encode_targets is None:
                raise ValueError("bootstrap episodic writing requires encode_targets")
            write_decisions = encode_targets.bool()
        else:
            write_decisions = output.controls.memory_write >= write_threshold
        if lesions.episodic:
            for row, should_encode in enumerate(write_decisions):
                if not should_encode:
                    continue
                episodic_memory.write(
                    EpisodicEvent(
                        key=self.episodic_key(output.hidden_summary[row]),
                        value=output.hidden_summary[row],
                        timestamp=timestamps[row],
                        session_id=session_ids[row],
                        task_context=task_contexts[row],
                        goal_state=state.pfc.slots[row, 0],
                        workspace_summary=output.workspace.slots[row].mean(0),
                        outcome=float(outcome[row]),
                        confidence=float(output.controls.memory_write[row]),
                        provenance=provenances[row],
                    )
                )
        return CognitiveState(
            state.pfc.detach(),
            state.working_memory.detach(),
            fast.detach(),
            state.overload.detach(),
            state.cycle,
        )

    def set_wake_mode(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def set_sleep_mode(self) -> None:
        self.train()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        allowed = (self.adapters, self.retrieval_integration)
        for module in allowed:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def set_development_mode(self) -> None:
        self.train()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                parameter.requires_grad_(True)

    @staticmethod
    def sleep_loss(
        slow_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        retention_loss: torch.Tensor,
        ewc_penalty: torch.Tensor,
    ) -> torch.Tensor:
        task = F.cross_entropy(slow_logits, targets)
        distillation = F.kl_div(
            F.log_softmax(slow_logits, dim=-1),
            F.softmax(teacher_logits.detach(), dim=-1),
            reduction="batchmean",
        )
        return task + 0.5 * distillation + 0.2 * retention_loss + 1e-4 * ewc_penalty
