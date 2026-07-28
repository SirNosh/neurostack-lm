from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .data import Stage1RExample, stable_order


def adapt_babilong_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    task: str,
    length: str,
    split: str,
    start_index: int = 0,
) -> list[Stage1RExample]:
    """Format rows from the official RMT-team BABILong release.

    The released rows expose input/question/target but not supporting-fact IDs,
    so support fields remain empty instead of inventing oracle annotations.
    """
    examples = []
    for offset, row in enumerate(rows):
        row_index = start_index + offset
        example_id = f"babilong:{length}:{task}:{split}:{row_index}"
        examples.append(
            Stage1RExample(
                example_id=example_id,
                family="babilong",
                input_text=f"{row['input'].rstrip()}\nQuestion: {row['question'].strip()}\nAnswer:",
                target_text=row["target"].strip(),
                session_id=example_id,
                task_context=f"{task}:{length}",
                timestamp=0,
                support_spans=[],
                support_item_ids=[],
                retrieval_target_ids=[],
                encode_target=None,
                verifier_label=None,
                relation_label=None,
                boundary_label=None,
                reset_pfc=True,
                reset_working_memory=True,
                reset_fast_weights=True,
                reset_episodic_memory=True,
            )
        )
    return examples


def babilong_stage1r_training_split(
    directory: Path, *, seed: int = 1729
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    selected: list[Stage1RExample] = []
    raw_files = []
    for task_number in range(1, 6):
        path = directory / f"qa{task_number}-4k.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        examples = adapt_babilong_rows(
            rows,
            task=f"qa{task_number}",
            length="4k",
            split="train",
        )
        selected.extend(stable_order(examples, seed)[:2000])
        raw_files.append(path)
    return {"train": selected, "dev": [], "test": []}, raw_files


def babilong_stage1r_evaluation_splits(
    directory: Path,
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    splits = {"train": [], "dev": [], "test": []}
    raw_files = []
    for length in ("4k", "8k", "16k", "32k"):
        output_split = "test" if length == "32k" else "dev"
        for task_number in range(1, 6):
            path = directory / f"qa{task_number}-{length}.json"
            if not path.exists():
                path = directory / "data" / f"qa{task_number}" / f"{length}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            splits[output_split].extend(
                adapt_babilong_rows(
                    rows,
                    task=f"qa{task_number}",
                    length=length,
                    split=output_split,
                )
            )
            raw_files.append(path)
    return splits, raw_files
