from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import Experiment2Example
from .streams import EpisodeStreamLoader
from src.stage1r.epbench import _answer_text


def adapt_epbench_stream(
    book_dir: Path, *, split: str, question_limit: int | None = None
) -> list[Experiment2Example]:
    events = pd.read_parquet(book_dir / "df_book_groundtruth.parquet")
    questions = pd.read_parquet(book_dir / "df_qa.parquet")
    stream_id = f"epbench:{split}:{book_dir.name}"
    referenced: set[int] = set()
    for row in questions.itertuples(index=False):
        referenced.update(map(int, row.correct_answer_chapters))

    output: list[Experiment2Example] = []
    for row in events.itertuples(index=False):
        chapter = int(row.chapter)
        event_id = f"{stream_id}:chapter-{chapter}"
        text = (
            f"Date: {row.date}\nLocation: {row.location}\n"
            f"Protagonist: {row.entity}\nEvent: {row.content}"
        )
        output.append(
            Experiment2Example(
                event_id, "epbench", text, "", stream_id, stream_id, chapter - 1,
                [(0, len(text))], None, [], event_id, [], chapter in referenced,
                None, None, chapter == 1, chapter == 1, chapter == 1, chapter == 1,
            )
        )

    rows = list(questions.itertuples(index=False))
    if question_limit is not None:
        rows = rows[:question_limit]
    for offset, row in enumerate(rows):
        chapters = list(map(int, row.correct_answer_chapters))
        question = f"{row.question}\nAnswer:"
        output.append(
            Experiment2Example(
                f"{stream_id}:question-{row.q_idx}", "epbench", question,
                _answer_text(row.correct_answer), stream_id, stream_id,
                len(events) + offset, [], (0, len(question)), [], None,
                [f"{stream_id}:chapter-{chapter}" for chapter in chapters],
                None, None, None, False, False, False, False,
            )
        )
    return output


def epbench_loader(
    book_dir: Path, *, split: str, question_limit: int | None = None
) -> EpisodeStreamLoader:
    return EpisodeStreamLoader(
        adapt_epbench_stream(book_dir, split=split, question_limit=question_limit)
    )
