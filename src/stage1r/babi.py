from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from .data import Stage1RExample, stable_order


TASK_NUMBER = re.compile(r"qa(\d+)_")


def parse_babi(path: Path) -> list[Stage1RExample]:
    task_match = TASK_NUMBER.search(path.name)
    if task_match is None:
        raise ValueError(f"cannot infer bAbI task from {path.name}")
    task = f"qa{int(task_match.group(1))}"
    facts: dict[int, str] = {}
    story = 0
    question_index = 0
    examples: list[Stage1RExample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        item_id_text, content = raw_line.split(" ", 1)
        item_id = int(item_id_text)
        if item_id == 1:
            facts = {}
            story += 1
            question_index = 0
        if "\t" not in content:
            facts[item_id] = content
            continue
        question, answer, support_text = content.split("\t")
        question = question.rstrip()
        support_ids = [int(value) for value in support_text.split()]
        fact_lines = [facts[index] for index in sorted(facts)]
        context = "\n".join(fact_lines)
        input_text = f"{context}\nQuestion: {question}\nAnswer:"
        spans = []
        support_item_ids = []
        offset = 0
        for index in sorted(facts):
            fact = facts[index]
            if index in support_ids:
                spans.append((offset, offset + len(fact)))
                support_item_ids.append(f"{task}:story-{story}:fact-{index}")
            offset += len(fact) + 1
        question_index += 1
        example_id = f"babi:{task}:story-{story}:question-{question_index}"
        examples.append(
            Stage1RExample(
                example_id=example_id,
                family="babi",
                input_text=input_text,
                target_text=answer,
                session_id=example_id,
                task_context=task,
                timestamp=item_id,
                support_spans=spans,
                support_item_ids=support_item_ids,
                retrieval_target_ids=[],
                encode_target=True,
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


def babi_stage1r_splits(
    directory: Path, *, seed: int = 1729
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    train: list[Stage1RExample] = []
    dev: list[Stage1RExample] = []
    test: list[Stage1RExample] = []
    raw_files: list[Path] = []
    for task in range(1, 6):
        train_path = next(directory.glob(f"qa{task}_*_train.txt"))
        test_path = next(directory.glob(f"qa{task}_*_test.txt"))
        raw_files.extend([train_path, test_path])
        ordered = stable_order(parse_babi(train_path), seed)
        if len(ordered) < 5500:
            raise ValueError(f"{train_path.name} has fewer than 5,500 examples")
        train.extend(ordered[:5000])
        dev.extend(ordered[5000:5500])
        test.extend(parse_babi(test_path))
    return {"train": train, "dev": dev, "test": test}, raw_files
