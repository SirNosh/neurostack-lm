from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .data import Experiment2Example


@dataclass
class StreamState:
    stream_id: str
    event_ids: list[str] = field(default_factory=list)
    steps: int = 0


class StreamRunner:
    """Runs whole episodes while preserving state between events and questions."""

    def __init__(self) -> None:
        self.state: StreamState | None = None

    def reset(self, stream_id: str) -> StreamState:
        self.state = StreamState(stream_id)
        return self.state

    def run(
        self,
        stream: list[Experiment2Example],
        step: Callable[[Experiment2Example, StreamState], None],
    ) -> StreamState:
        if not stream:
            raise ValueError("cannot run an empty stream")
        stream_id = stream[0].stream_id
        if any(item.stream_id != stream_id for item in stream):
            raise ValueError("one runner call must contain exactly one stream")
        state = self.reset(stream_id)
        for index, item in enumerate(stream):
            if index and (
                item.reset_pfc
                or item.reset_working_memory
                or item.reset_fast_weights
                or item.reset_episodic_memory
            ):
                raise ValueError("state resets are only valid at a stream boundary")
            if item.event_id is not None:
                state.event_ids.append(item.event_id)
            step(item, state)
            state.steps += 1
        return state
