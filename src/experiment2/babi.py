from __future__ import annotations

from pathlib import Path
import re

from .data import Experiment2Example


TASK = re.compile(r"qa(\d+)_")


def parse_babi(path: Path) -> list[Experiment2Example]:
    match = TASK.search(path.name)
    if match is None:
        raise ValueError(f"cannot infer task from {path.name}")
    task = f"qa{int(match.group(1))}"
    facts: dict[int, str] = {}
    story = question_index = 0
    output: list[Experiment2Example] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        number, content = raw.split(" ", 1)
        item_id = int(number)
        if item_id == 1:
            facts = {}
            story += 1
            question_index = 0
        if "\t" not in content:
            facts[item_id] = content
            continue
        question, answer, support = content.split("\t")
        ordered = sorted(facts)
        context = "\n".join(facts[index] for index in ordered)
        fact_spans, offset = [], 0
        for index in ordered:
            fact_spans.append((offset, offset + len(facts[index])))
            offset += len(facts[index]) + 1
        question_text = f"Question: {question.rstrip()}\nAnswer:"
        question_span = (len(context) + 1, len(context) + 1 + len(question_text))
        question_index += 1
        identifier = f"experiment2:babi:{task}:story-{story}:q-{question_index}"
        support_ids = set(map(int, support.split()))
        output.append(
            Experiment2Example(
                identifier, "babi", f"{context}\n{question_text}", answer,
                identifier, identifier, item_id, fact_spans, question_span,
                [position for position, index in enumerate(ordered) if index in support_ids],
                None, [], None, None, None, True, True, True, True,
            )
        )
    return output
