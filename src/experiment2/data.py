from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, TypeVar


ADAPTER_VERSION = "experiment2-v1"
T = TypeVar("T")


@dataclass
class Experiment2Example:
    example_id: str
    family: str
    input_text: str
    target_text: str
    stream_id: str
    session_id: str
    timestamp: int
    fact_spans: list[tuple[int, int]]
    question_span: tuple[int, int] | None
    support_fact_indices: list[int]
    event_id: str | None
    retrieval_target_ids: list[str]
    future_use_target: bool | None
    candidate_span: tuple[int, int] | None
    verifier_label: int | None
    reset_pfc: bool
    reset_working_memory: bool
    reset_fast_weights: bool
    reset_episodic_memory: bool

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def validate_stream(examples: Iterable[Experiment2Example]) -> list[Experiment2Example]:
    ordered = sorted(examples, key=lambda item: (item.stream_id, item.timestamp))
    last: dict[str, int] = {}
    for item in ordered:
        if item.stream_id in last and item.timestamp <= last[item.stream_id]:
            raise ValueError(f"timestamps must increase within {item.stream_id}")
        last[item.stream_id] = item.timestamp
    return ordered


def stable_order(examples: Iterable[T], seed: int) -> list[T]:
    return sorted(
        examples,
        key=lambda item: hashlib.sha256(
            f"{seed}:{getattr(item, 'example_id')}".encode()
        ).digest(),
    )
