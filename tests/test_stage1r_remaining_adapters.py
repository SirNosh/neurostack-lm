import csv
import json

import pandas as pd

from src.stage1r.babilong import babilong_stage1r_evaluation_splits
from src.stage1r.clutrr import clutrr_stage1r_splits
from src.stage1r.epbench import adapt_epbench_book
from src.stage1r.multisession_chat import adapt_msc_file
from src.stage1r.trace import trace_stage1r_splits


def test_epbench_preserves_exact_chapter_retrieval_targets(tmp_path):
    pd.DataFrame(
        [
            {
                "chapter": 1,
                "date": "today",
                "location": "lab",
                "entity": "Ada",
                "content": "test",
                "post_entities": ["Lin"],
            }
        ]
    ).to_parquet(tmp_path / "df_book_groundtruth.parquet")
    pd.DataFrame(
        [
            {
                "q_idx": 4,
                "question": "Where?",
                "correct_answer": ["lab"],
                "correct_answer_chapters": [1],
                "retrieval_type": "Spaces",
            }
        ]
    ).to_parquet(tmp_path / "df_qa.parquet")
    examples = adapt_epbench_book(
        tmp_path, split="test", question_limit=1
    )
    assert examples[-1].retrieval_target_ids == [
        f"epbench:test:{tmp_path.name}:chapter-1"
    ]
    assert examples[0].task_context == examples[-1].task_context == "epbench"
    assert examples[0].encode_target is True
    assert examples[-1].encode_target is False


def test_msc_preserves_memory_across_native_sessions(tmp_path):
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {
                "dialoug_id": 7,
                "session_id": session,
                "dialogue": [f"hello {session}", "reply"],
                "speaker": ["Speaker 1", "Speaker 2"],
            }
            for session in (0, 1)
        ]
    ).to_parquet(path)
    examples = adapt_msc_file(path, split="train", conversation_limit=1)
    assert len({example.session_id for example in examples}) == 1
    assert examples[0].reset_episodic_memory
    assert not any(example.reset_episodic_memory for example in examples[1:])


def test_clutrr_freezes_depth_5_to_7_dev_and_8_to_10_test(tmp_path):
    fieldnames = ["id", "story", "query", "target"]
    for name in ["1.2,1.3,1.4_train.csv", *[f"1.{d}_test.csv" for d in range(5, 11)]]:
        with (tmp_path / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {"id": name, "story": "A is B's parent.", "query": "('A','B')", "target": "parent"}
            )
    splits, _ = clutrr_stage1r_splits(tmp_path, per_depth_limit=1)
    assert len(splits["dev"]) == 3
    assert len(splits["test"]) == 3


def test_trace_boundary_is_metadata_and_does_not_reset_state(tmp_path):
    for task in ("C-STANCE", "FOMC", "MeetingBank", "Py150"):
        directory = tmp_path / task
        directory.mkdir()
        for filename in ("train.json", "eval.json", "test.json"):
            (directory / filename).write_text(
                json.dumps([{"prompt": "prompt", "answer": "answer"}]),
                encoding="utf-8",
            )
    splits, _ = trace_stage1r_splits(tmp_path)
    train = splits["train"]
    assert [example.boundary_label for example in train] == [0, 1, 1, 1]
    assert all(example.task_context == "trace" for example in train)
    assert not any(example.reset_pfc for example in train[1:])


def test_babilong_eval_freezes_all_four_length_cells(tmp_path):
    row = [{"input": "fact", "question": "q?", "target": "a"}]
    for length in ("4k", "8k", "16k", "32k"):
        for task in range(1, 6):
            (tmp_path / f"qa{task}-{length}.json").write_text(
                json.dumps(row), encoding="utf-8"
            )
    splits, raw_files = babilong_stage1r_evaluation_splits(tmp_path)
    assert len(splits["dev"]) == 15
    assert len(splits["test"]) == 5
    assert len(raw_files) == 20
