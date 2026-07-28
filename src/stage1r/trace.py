from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .data import Stage1RExample


TRACE_TASKS = ("C-STANCE", "FOMC", "MeetingBank", "Py150")


def trace_stage1r_splits(
    root: Path, *, tasks: Sequence[str] = TRACE_TASKS
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    splits = {"train": [], "dev": [], "test": []}
    raw_files = []
    for output_split, filename in (
        ("train", "train.json"),
        ("dev", "eval.json"),
        ("test", "test.json"),
    ):
        timestamp = 0
        for task_index, task in enumerate(tasks):
            path = root / task / filename
            rows = json.loads(path.read_text(encoding="utf-8"))
            raw_files.append(path)
            for row_index, row in enumerate(rows):
                first = timestamp == 0
                splits[output_split].append(
                    Stage1RExample(
                        example_id=f"trace:{output_split}:{task}:{row_index}",
                        family="trace",
                        input_text=str(row["prompt"]),
                        target_text=str(row["answer"]),
                        session_id=f"trace:{output_split}:continual-stream",
                        task_context="trace",
                        timestamp=timestamp,
                        support_spans=[],
                        support_item_ids=[],
                        retrieval_target_ids=[],
                        encode_target=None,
                        verifier_label=None,
                        relation_label=None,
                        boundary_label=int(row_index == 0 and task_index > 0),
                        reset_pfc=first,
                        reset_working_memory=first,
                        reset_fast_weights=first,
                        reset_episodic_memory=first,
                    )
                )
                timestamp += 1
    return splits, raw_files
