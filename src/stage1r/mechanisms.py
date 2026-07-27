from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


PFC_SLOTS = ("goal", "rule", "plan", "uncertainty")
EXPERTS = ("relational", "planning", "memory", "verification")


@dataclass
class PFCState:
    slots: torch.Tensor

    def detach(self) -> "PFCState":
        return PFCState(self.slots.detach())


class PersistentPFC(nn.Module):
    """Four gated recurrent slots motivated by the PBWM model.

    O'Reilly & Frank (2006), doi:10.1162/089976606775093909.
    This is a functional abstraction, not a biological reproduction.
    """

    def __init__(self, input_dim: int = 1280, slot_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.cells = nn.ModuleList(
            [nn.GRUCell(input_dim, slot_dim) for _ in PFC_SLOTS]
        )

    def initialize(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PFCState:
        return PFCState(
            torch.zeros(batch_size, len(PFC_SLOTS), self.slot_dim, device=device, dtype=dtype)
        )

    def update(
        self,
        inputs: torch.Tensor,
        state: PFCState,
        update_gates: torch.Tensor,
    ) -> PFCState:
        if inputs.shape != (state.slots.shape[0], self.input_dim):
            raise ValueError(f"expected PFC inputs [batch,{self.input_dim}]")
        if update_gates.shape != state.slots.shape[:2]:
            raise ValueError("update_gates must have shape [batch,4]")
        candidates = torch.stack(
            [
                cell(inputs, state.slots[:, index])
                for index, cell in enumerate(self.cells)
            ],
            dim=1,
        )
        gates = update_gates.clamp(0, 1).unsqueeze(-1)
        return PFCState(gates * candidates + (1 - gates) * state.slots)

    def reset(self, state: PFCState, mask: torch.Tensor) -> PFCState:
        if mask.shape != state.slots.shape[:1]:
            raise ValueError("reset mask must have shape [batch]")
        slots = state.slots.clone()
        slots[mask.bool()] = 0
        return PFCState(slots)


class MemoryOperation(IntEnum):
    KEEP = 0
    REPLACE = 1
    MERGE = 2
    CLEAR = 3
    PROTECT = 4


@dataclass
class WorkingMemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    confidence: torch.Tensor
    age: torch.Tensor
    protection: torch.Tensor
    occupied: torch.Tensor

    def clone(self) -> "WorkingMemoryState":
        return WorkingMemoryState(
            self.keys.clone(),
            self.values.clone(),
            self.confidence.clone(),
            self.age.clone(),
            self.protection.clone(),
            self.occupied.clone(),
        )

    def detach(self) -> "WorkingMemoryState":
        return WorkingMemoryState(
            self.keys.detach(),
            self.values.detach(),
            self.confidence.detach(),
            self.age.detach(),
            self.protection.detach(),
            self.occupied.detach(),
        )


class WorkingMemory(nn.Module):
    """Eight addressable slots with explicit update operations.

    The external-memory interface follows Neural Turing Machine/DNC practice;
    protection and discrete operations implement the Stage 1R contract.
    """

    def __init__(self, slots: int = 8, key_dim: int = 64, value_dim: int = 256) -> None:
        super().__init__()
        self.slots = slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query = nn.Linear(key_dim, key_dim, bias=False)

    def initialize(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> WorkingMemoryState:
        shape = (batch_size, self.slots)
        return WorkingMemoryState(
            keys=torch.zeros(*shape, self.key_dim, device=device, dtype=dtype),
            values=torch.zeros(*shape, self.value_dim, device=device, dtype=dtype),
            confidence=torch.zeros(*shape, device=device, dtype=dtype),
            age=torch.zeros(*shape, device=device, dtype=dtype),
            protection=torch.zeros(*shape, device=device, dtype=dtype),
            occupied=torch.zeros(*shape, device=device, dtype=torch.bool),
        )

    def read(
        self, state: WorkingMemoryState, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_query = F.normalize(self.query(query), dim=-1)
        normalized_keys = F.normalize(state.keys, dim=-1)
        scores = torch.einsum("bd,bsd->bs", normalized_query, normalized_keys)
        scores = scores.masked_fill(~state.occupied, -1e4)
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(
            state.occupied.any(-1, keepdim=True), weights, torch.zeros_like(weights)
        )
        return torch.einsum("bs,bsd->bd", weights, state.values), weights

    def operate(
        self,
        state: WorkingMemoryState,
        operation: torch.Tensor,
        slot: torch.Tensor,
        *,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> WorkingMemoryState:
        batch = state.keys.shape[0]
        if operation.shape != (batch,) or slot.shape != (batch,):
            raise ValueError("operation and slot must have shape [batch]")
        result = state.clone()
        result.age = result.age + result.occupied.to(result.age.dtype)
        for row in range(batch):
            index = int(slot[row])
            op = MemoryOperation(int(operation[row]))
            if op == MemoryOperation.KEEP:
                continue
            if op == MemoryOperation.PROTECT:
                result.protection[row, index] = 1
                continue
            if op == MemoryOperation.CLEAR:
                if result.protection[row, index] < 0.5:
                    result.keys[row, index].zero_()
                    result.values[row, index].zero_()
                    result.confidence[row, index] = 0
                    result.age[row, index] = 0
                    result.occupied[row, index] = False
                continue
            if key is None or value is None:
                raise ValueError("replace and merge require key and value")
            incoming_confidence = (
                confidence[row] if confidence is not None else torch.ones((), device=value.device)
            )
            if op == MemoryOperation.REPLACE:
                if result.protection[row, index] < 0.5 or not result.occupied[row, index]:
                    result.keys[row, index] = key[row]
                    result.values[row, index] = value[row]
                    result.confidence[row, index] = incoming_confidence
                    result.age[row, index] = 0
                    result.protection[row, index] = 0
                    result.occupied[row, index] = True
            elif op == MemoryOperation.MERGE:
                old_weight = result.confidence[row, index].clamp_min(0)
                denominator = (old_weight + incoming_confidence).clamp_min(1e-6)
                result.keys[row, index] = (
                    old_weight * result.keys[row, index] + incoming_confidence * key[row]
                ) / denominator
                result.values[row, index] = (
                    old_weight * result.values[row, index] + incoming_confidence * value[row]
                ) / denominator
                result.confidence[row, index] = denominator.clamp_max(1)
                result.age[row, index] = 0
                result.occupied[row, index] = True
        return result

    def reset(self, state: WorkingMemoryState, mask: torch.Tensor) -> WorkingMemoryState:
        fresh = self.initialize(
            state.keys.shape[0], device=state.keys.device, dtype=state.keys.dtype
        )
        keep = (~mask.bool()).view(-1, 1)
        keep3 = keep.unsqueeze(-1)
        return WorkingMemoryState(
            torch.where(keep3, state.keys, fresh.keys),
            torch.where(keep3, state.values, fresh.values),
            torch.where(keep, state.confidence, fresh.confidence),
            torch.where(keep, state.age, fresh.age),
            torch.where(keep, state.protection, fresh.protection),
            torch.where(keep, state.occupied, fresh.occupied),
        )


@dataclass
class WorkspaceState:
    slots: torch.Tensor
    sources: torch.Tensor
    scores: torch.Tensor


class Workspace(nn.Module):
    """Bandwidth-limited shared workspace following Goyal et al. (2021)."""

    SOURCE_NAMES = (
        "token",
        "pfc",
        "working_memory",
        "episodic",
        "specialist",
        "verifier",
        "appraisal",
    )

    def __init__(self, value_dim: int = 256, capacity: int = 4, broadcast_dim: int = 896):
        super().__init__()
        self.value_dim = value_dim
        self.capacity = capacity
        self.source_embedding = nn.Embedding(len(self.SOURCE_NAMES), value_dim)
        self.scorer = nn.Sequential(
            nn.Linear(value_dim * 2, 256), nn.SiLU(), nn.Linear(256, 1)
        )
        self.broadcast_projection = nn.Linear(value_dim, broadcast_dim)

    def compete(
        self,
        candidates: Sequence[tuple[str, torch.Tensor]],
        masks: Sequence[torch.Tensor] | None = None,
    ) -> WorkspaceState:
        if not candidates:
            raise ValueError("workspace needs candidates")
        batch = candidates[0][1].shape[0]
        values = []
        source_ids = []
        valid_masks = []
        if masks is None:
            masks = [
                torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
                for _, value in candidates
            ]
        for (name, value), mask in zip(candidates, masks):
            if name not in self.SOURCE_NAMES:
                raise ValueError(f"unknown workspace source {name}")
            if value.ndim != 3 or value.shape[0] != batch or value.shape[2] != self.value_dim:
                raise ValueError("workspace candidates must be [batch,count,256]")
            source_id = self.SOURCE_NAMES.index(name)
            values.append(value)
            source_ids.append(
                torch.full(value.shape[:2], source_id, device=value.device, dtype=torch.long)
            )
            valid_masks.append(mask.bool())
        all_values = torch.cat(values, dim=1)
        all_sources = torch.cat(source_ids, dim=1)
        all_valid = torch.cat(valid_masks, dim=1)
        source_vectors = self.source_embedding(all_sources)
        logits = self.scorer(torch.cat([all_values, source_vectors], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~all_valid, -1e4)

        padding = max(0, self.capacity - all_values.shape[1])
        if padding:
            all_values = F.pad(all_values, (0, 0, 0, padding))
            all_sources = F.pad(all_sources, (0, padding), value=-1)
            logits = F.pad(logits, (0, padding), value=-1e4)
        scores, indices = logits.topk(self.capacity, dim=1)
        slots = torch.gather(
            all_values, 1, indices.unsqueeze(-1).expand(-1, -1, self.value_dim)
        )
        selected_sources = torch.gather(all_sources, 1, indices)
        return WorkspaceState(slots, selected_sources, scores)

    def broadcast(self, state: WorkspaceState) -> torch.Tensor:
        return self.broadcast_projection(state.slots)


@dataclass
class RoutingResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class SparseRouter(nn.Module):
    """Top-2 router with Shazeer load balancing and router z-loss."""

    def __init__(
        self, input_dim: int, locations: int = 4, experts: int = 4, hidden_dim: int = 512
    ) -> None:
        super().__init__()
        self.locations = locations
        self.experts = experts
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, locations * experts)

    def forward(
        self,
        inputs: torch.Tensor,
        temperature: torch.Tensor | float = 1.0,
        fast_hidden: torch.Tensor | None = None,
    ) -> RoutingResult:
        hidden = self.input_projection(inputs)
        if fast_hidden is not None:
            hidden = hidden + fast_hidden
        logits = self.output_projection(F.silu(hidden)).view(
            -1, self.locations, self.experts
        )
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.view(-1, 1, 1)
        probabilities = F.softmax(logits / temperature, dim=-1)
        weights, indices = probabilities.topk(2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return RoutingResult(logits, probabilities, indices, weights)

    @staticmethod
    def load_balance_loss(result: RoutingResult) -> torch.Tensor:
        importance = result.probabilities.mean(dim=(0, 1))
        assignments = F.one_hot(
            result.indices[..., 0], result.probabilities.shape[-1]
        ).float()
        load = assignments.mean(dim=(0, 1))
        experts = result.probabilities.shape[-1]
        return experts * torch.sum(importance * load)

    @staticmethod
    def z_loss(result: RoutingResult) -> torch.Tensor:
        return torch.logsumexp(result.logits, dim=-1).square().mean()

    @staticmethod
    def bootstrap_loss(result: RoutingResult, preferred_expert: torch.Tensor) -> torch.Tensor:
        logits = result.logits.mean(dim=1)
        return F.cross_entropy(logits, preferred_expert)


@dataclass(frozen=True)
class EpisodicEvent:
    key: torch.Tensor
    value: torch.Tensor
    timestamp: int
    session_id: str
    task_context: str
    goal_state: torch.Tensor
    workspace_summary: torch.Tensor
    outcome: float
    confidence: float
    provenance: str


@dataclass
class RetrievalResult:
    values: torch.Tensor
    events: list[list[EpisodicEvent]]
    scores: torch.Tensor


class EpisodicMemory:
    """Session-scoped latent event memory; it never stores answer-label logits."""

    def __init__(self, capacity: int = 8192, top_k: int = 4, value_dim: int = 256):
        self.capacity = capacity
        self.top_k = top_k
        self.value_dim = value_dim
        self._events: list[EpisodicEvent] = []

    @property
    def events(self) -> tuple[EpisodicEvent, ...]:
        return tuple(self._events)

    def write(self, event: EpisodicEvent) -> None:
        if event.key.ndim != 1 or event.value.shape != (self.value_dim,):
            raise ValueError("episodic keys and values must be vectors")
        self._events.append(
            EpisodicEvent(
                key=F.normalize(event.key.detach().cpu().float(), dim=0),
                value=event.value.detach().cpu().float(),
                timestamp=event.timestamp,
                session_id=event.session_id,
                task_context=event.task_context,
                goal_state=event.goal_state.detach().cpu().float(),
                workspace_summary=event.workspace_summary.detach().cpu().float(),
                outcome=event.outcome,
                confidence=event.confidence,
                provenance=event.provenance,
            )
        )
        if len(self._events) > self.capacity:
            self._events.pop(0)

    def retrieve(
        self,
        keys: torch.Tensor,
        *,
        session_ids: Sequence[str],
        task_contexts: Sequence[str] | None = None,
        breadth: int | None = None,
        allow_cross_session: bool = False,
    ) -> RetrievalResult:
        count = breadth or self.top_k
        device = keys.device
        batch_values = []
        batch_events: list[list[EpisodicEvent]] = []
        batch_scores = []
        for row, query in enumerate(keys):
            candidates = [
                event
                for event in self._events
                if allow_cross_session or event.session_id == session_ids[row]
            ]
            if task_contexts is not None:
                same_task = [
                    event for event in candidates if event.task_context == task_contexts[row]
                ]
                candidates = same_task or candidates
            if not candidates:
                batch_values.append(
                    torch.zeros(
                        count,
                        self.value_dim,
                        device=device,
                        dtype=keys.dtype,
                    )
                )
                batch_events.append([])
                batch_scores.append(torch.full((count,), -1e4, device=device))
                continue
            event_keys = torch.stack([event.key for event in candidates]).to(device)
            scores = event_keys @ F.normalize(query.float(), dim=0)
            selected_count = min(count, len(candidates))
            selected_scores, selected = scores.topk(selected_count)
            selected_events = [candidates[int(index)] for index in selected]
            values = torch.stack([event.value for event in selected_events]).to(
                device=device, dtype=keys.dtype
            )
            if selected_count < count:
                values = F.pad(values, (0, 0, 0, count - selected_count))
                selected_scores = F.pad(
                    selected_scores, (0, count - selected_count), value=-1e4
                )
            batch_values.append(values)
            batch_events.append(selected_events)
            batch_scores.append(selected_scores)
        return RetrievalResult(
            torch.stack(batch_values), batch_events, torch.stack(batch_scores)
        )

    def reset_session(self, session_id: str) -> None:
        self._events = [event for event in self._events if event.session_id != session_id]

    def clear(self) -> None:
        self._events.clear()


@dataclass
class FastMatrixState:
    u: torch.Tensor
    v: torch.Tensor

    @property
    def matrix(self) -> torch.Tensor:
        return self.u @ self.v.transpose(-1, -2)

    def detach(self) -> "FastMatrixState":
        return FastMatrixState(self.u.detach(), self.v.detach())


class FastWeightAdapter(nn.Module):
    """Rank-8 local fast matrix with learned update projections.

    Based on Ba et al. (2016) fast weights and Miconi et al. (2020)
    neuromodulated plasticity. State updates are local and detached.
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, decay: float = 0.995):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.decay = decay
        self.key_rank = nn.Linear(input_dim, rank, bias=False)
        self.value_rank = nn.Linear(output_dim, rank, bias=False)

    def initialize(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> FastMatrixState:
        return FastMatrixState(
            torch.zeros(batch_size, self.output_dim, self.rank, device=device, dtype=dtype),
            torch.zeros(batch_size, self.input_dim, self.rank, device=device, dtype=dtype),
        )

    def apply(self, inputs: torch.Tensor, state: FastMatrixState) -> torch.Tensor:
        projected = torch.einsum("bir,bi->br", state.v, inputs)
        return torch.einsum("bor,br->bo", state.u, projected)

    @torch.no_grad()
    def update(
        self,
        state: FastMatrixState,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        da: torch.Tensor,
        ach: torch.Tensor,
        ne: torch.Tensor,
        base_rate: float = 0.01,
    ) -> FastMatrixState:
        scale = (
            base_rate * da.to(keys.dtype) * ach.to(keys.dtype) * (0.5 + ne.to(keys.dtype))
        ).view(-1, 1, 1)
        key_rank = torch.tanh(self.key_rank(keys)).unsqueeze(1)
        value_rank = torch.tanh(self.value_rank(values)).unsqueeze(1)
        normalized_keys = F.normalize(keys, dim=-1).unsqueeze(-1)
        normalized_values = F.normalize(values, dim=-1).unsqueeze(-1)
        new_u = self.decay * state.u + scale * normalized_values * key_rank
        new_v = self.decay * state.v + scale * normalized_keys * value_rank
        return FastMatrixState(new_u.detach(), new_v.detach())


@dataclass
class FastWeightState:
    pfc: FastMatrixState
    working_read: FastMatrixState
    retrieval: FastMatrixState
    router_query: FastMatrixState

    def detach(self) -> "FastWeightState":
        return FastWeightState(
            self.pfc.detach(),
            self.working_read.detach(),
            self.retrieval.detach(),
            self.router_query.detach(),
        )


class FastWeightBank(nn.Module):
    def __init__(self, rank: int = 8) -> None:
        super().__init__()
        self.pfc = FastWeightAdapter(1280, 256, rank)
        self.working_read = FastWeightAdapter(64, 256, rank)
        self.retrieval = FastWeightAdapter(256, 256, rank)
        self.router_query = FastWeightAdapter(256, 512, rank)

    def initialize(
        self, batch_size: int, *, device: torch.device | str, dtype: torch.dtype
    ) -> FastWeightState:
        return FastWeightState(
            self.pfc.initialize(batch_size, device=device, dtype=dtype),
            self.working_read.initialize(batch_size, device=device, dtype=dtype),
            self.retrieval.initialize(batch_size, device=device, dtype=dtype),
            self.router_query.initialize(batch_size, device=device, dtype=dtype),
        )


@dataclass
class ModulatorSignals:
    da: torch.Tensor
    ne: torch.Tensor
    ach: torch.Tensor
    serotonin: torch.Tensor
    overload: torch.Tensor


@dataclass
class ControlValues:
    fast_scale: torch.Tensor
    replay_priority: torch.Tensor
    router_temperature: torch.Tensor
    retrieval_breadth: torch.Tensor
    strategy_reset: torch.Tensor
    encode_weight: torch.Tensor
    retrieve_weight: torch.Tensor
    memory_write: torch.Tensor
    verify_threshold: torch.Tensor
    continue_bias: torch.Tensor
    abstain_threshold: torch.Tensor
    conflict_pressure: torch.Tensor


class ModulatorController(nn.Module):
    """Five supervised channels wired to disjoint control paths.

    Learned neuromodulation follows Backpropamine conceptually. The specific
    channel factorization remains a hypothesis tested against a generic head.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, differentiated: bool = True):
        super().__init__()
        self.differentiated = differentiated
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5)
        )

    def forward(
        self, inputs: torch.Tensor, previous_overload: torch.Tensor
    ) -> ModulatorSignals:
        raw = self.network(inputs)
        conflict = raw[:, 4].sigmoid()
        overload = 0.98 * previous_overload + 0.02 * conflict
        return ModulatorSignals(
            da=raw[:, 0].tanh(),
            ne=raw[:, 1].sigmoid(),
            ach=raw[:, 2].sigmoid(),
            serotonin=raw[:, 3].sigmoid(),
            overload=overload,
        )

    def controls(self, signals: ModulatorSignals) -> ControlValues:
        da, ne, ach, serotonin, overload = (
            signals.da,
            signals.ne,
            signals.ach,
            signals.serotonin,
            signals.overload,
        )
        if not self.differentiated:
            mixed = torch.stack([da, ne, ach, serotonin, overload], dim=-1)
            shared = mixed.mean(-1).sigmoid()
            da = shared * 2 - 1
            ne = ach = serotonin = overload = shared
        return ControlValues(
            fast_scale=da.abs(),
            replay_priority=da.abs(),
            router_temperature=1.5 - ne,
            retrieval_breadth=(1 + (3 * ne).round()).long(),
            strategy_reset=ne,
            encode_weight=ach,
            retrieve_weight=1 - ach,
            memory_write=ach,
            verify_threshold=0.75 - 0.5 * serotonin,
            continue_bias=serotonin,
            abstain_threshold=0.75 - 0.5 * serotonin,
            conflict_pressure=overload,
        )


class Verifier(nn.Module):
    """Single-logit correctness verifier compatible with balanced PRM labels."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        unique = labels.unique()
        if unique.numel() < 2:
            raise ValueError("verifier batches must contain positive and negative labels")
        return F.binary_cross_entropy_with_logits(logits, labels.float())


@dataclass(frozen=True)
class LesionConfig:
    pfc: bool = True
    working_memory: bool = True
    workspace: bool = True
    episodic: bool = True
    fast_weights: bool = True
    verifier: bool = True
    modulators: bool = True
    routing: bool = True


@dataclass
class CognitiveState:
    pfc: PFCState
    working_memory: WorkingMemoryState
    fast_weights: FastWeightState
    overload: torch.Tensor
    cycle: int = 0

    def detach(self) -> "CognitiveState":
        return CognitiveState(
            self.pfc.detach(),
            self.working_memory.detach(),
            self.fast_weights.detach(),
            self.overload.detach(),
            self.cycle,
        )
