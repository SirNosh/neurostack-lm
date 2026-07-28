from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from .data import Experiment2Example, validate_stream


class EpisodeStreamLoader:
    """Yields complete ordered streams; never randomly breaks persistent state."""

    def __init__(self, examples: list[Experiment2Example]) -> None:
        grouped: dict[str, list[Experiment2Example]] = defaultdict(list)
        for example in validate_stream(examples):
            grouped[example.stream_id].append(example)
        self.streams = dict(grouped)

    def __iter__(self) -> Iterator[list[Experiment2Example]]:
        for stream_id in sorted(self.streams):
            yield self.streams[stream_id]
