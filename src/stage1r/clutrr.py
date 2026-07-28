from __future__ import annotations

import csv
from pathlib import Path

from .data import Stage1RExample, stable_order


def adapt_clutrr_file(
    path: Path, *, split: str, depth: str, limit: int, seed: int = 1729
) -> list[Stage1RExample]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    examples = []
    for index, row in enumerate(rows):
        query = row["query"].strip("()").replace("'", "")
        example_id = f"clutrr:{split}:depth-{depth}:{row.get('id') or index}"
        examples.append(
            Stage1RExample(
                example_id=example_id,
                family="clutrr",
                input_text=(
                    f"{row['story']}\nQuery: What is the family relationship "
                    f"from {query}?\nAnswer:"
                ),
                target_text=row["target"],
                session_id=example_id,
                task_context=f"clutrr:depth-{depth}",
                timestamp=0,
                support_spans=[],
                support_item_ids=[],
                retrieval_target_ids=[],
                encode_target=None,
                verifier_label=None,
                relation_label=row["target"],
                boundary_label=None,
                reset_pfc=True,
                reset_working_memory=True,
                reset_fast_weights=True,
                reset_episodic_memory=True,
            )
        )
    return stable_order(examples, seed)[:limit]


def clutrr_stage1r_splits(
    root: Path, *, train_limit: int = 2_000, per_depth_limit: int = 500, seed: int = 1729
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    train_path = root / "1.2,1.3,1.4_train.csv"
    splits = {
        "train": adapt_clutrr_file(
            train_path, split="train", depth="2-4", limit=train_limit, seed=seed
        ),
        "dev": [],
        "test": [],
    }
    raw_files = [train_path]
    for depth in range(5, 11):
        path = root / f"1.{depth}_test.csv"
        split = "dev" if depth <= 7 else "test"
        splits[split].extend(
            adapt_clutrr_file(
                path, split=split, depth=str(depth), limit=per_depth_limit, seed=seed
            )
        )
        raw_files.append(path)
    return splits, raw_files
