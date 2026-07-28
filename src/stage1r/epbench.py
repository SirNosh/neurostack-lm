from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import Stage1RExample, stable_order


def _answer_text(value: object) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def adapt_epbench_book(
    book_dir: Path, *, split: str, question_limit: int, seed: int = 1729
) -> list[Stage1RExample]:
    events = pd.read_parquet(book_dir / "df_book_groundtruth.parquet")
    questions = pd.read_parquet(book_dir / "df_qa.parquet")
    session_id = f"epbench:{split}:{book_dir.name}"
    output: list[Stage1RExample] = []
    for row in events.itertuples(index=False):
        chapter = int(row.chapter)
        item_id = f"{session_id}:chapter-{chapter}"
        entities = ", ".join(map(str, row.post_entities))
        output.append(
            Stage1RExample(
                example_id=item_id,
                family="epbench",
                input_text=(
                    f"Date: {row.date}\nLocation: {row.location}\n"
                    f"Protagonist: {row.entity}\nOther people: {entities}\n"
                    f"Event: {row.content}"
                ),
                target_text="",
                session_id=session_id,
                task_context="epbench",
                timestamp=chapter - 1,
                support_spans=[],
                support_item_ids=[],
                retrieval_target_ids=[],
                encode_target=True,
                verifier_label=None,
                relation_label=None,
                boundary_label=None,
                reset_pfc=chapter == 1,
                reset_working_memory=chapter == 1,
                reset_fast_weights=chapter == 1,
                reset_episodic_memory=chapter == 1,
            )
        )
    query_examples = []
    for row in questions.itertuples(index=False):
        chapters = (
            row.correct_answer_chapters.tolist()
            if hasattr(row.correct_answer_chapters, "tolist")
            else list(row.correct_answer_chapters)
        )
        query_examples.append(
            Stage1RExample(
                example_id=f"{session_id}:question-{row.q_idx}",
                family="epbench",
                input_text=f"{row.question}\nAnswer:",
                target_text=_answer_text(row.correct_answer),
                session_id=session_id,
                task_context="epbench",
                timestamp=len(events) + int(row.q_idx),
                support_spans=[],
                support_item_ids=[],
                retrieval_target_ids=[
                    f"{session_id}:chapter-{int(chapter)}" for chapter in chapters
                ],
                encode_target=False,
                verifier_label=None,
                relation_label=None,
                boundary_label=None,
                reset_pfc=False,
                reset_working_memory=False,
                reset_fast_weights=False,
                reset_episodic_memory=False,
            )
        )
    output.extend(stable_order(query_examples, seed)[:question_limit])
    return output


def epbench_stage1r_splits(
    root: Path, *, question_limit: int = 500, seed: int = 1729
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    books = list(root.rglob("df_qa.parquet"))
    by_tokens = {
        "10k": next(path.parent for path in books if "nbtokens_10397" in str(path)),
        "100k": next(path.parent for path in books if "nbtokens_102870" in str(path)),
        "1m": next(path.parent for path in books if "nbtokens_1033475" in str(path)),
    }
    splits = {
        "train": adapt_epbench_book(
            by_tokens["10k"], split="train-10k", question_limit=question_limit, seed=seed
        ),
        "dev": adapt_epbench_book(
            by_tokens["100k"], split="dev-100k", question_limit=question_limit, seed=seed
        ),
        "test": adapt_epbench_book(
            by_tokens["1m"], split="test-1m", question_limit=question_limit, seed=seed
        ),
    }
    raw_files = [
        book / name
        for book in by_tokens.values()
        for name in ("book.json", "df_book_groundtruth.parquet", "df_qa.parquet")
    ]
    return splits, raw_files
