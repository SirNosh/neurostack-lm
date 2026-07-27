from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterator

from .data import Stage1RExample


def _problem_id(problem: str) -> str:
    return hashlib.sha256(problem.strip().encode()).hexdigest()


def iter_prm800k(path: Path) -> Iterator[Stage1RExample]:
    with path.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            row = json.loads(line)
            problem = row["question"]["problem"].strip()
            problem_id = _problem_id(problem)
            prior_steps: list[str] = []
            for step_index, step in enumerate(row["label"]["steps"]):
                completions = step.get("completions") or []
                for completion_index, completion in enumerate(completions):
                    rating = completion.get("rating")
                    if rating not in (-1, 1) or completion.get("flagged") is True:
                        continue
                    candidate = completion["text"].strip()
                    context = "\n".join(prior_steps)
                    input_text = (
                        f"Problem: {problem}\n"
                        f"Solution so far:\n{context}\n"
                        f"Candidate step: {candidate}\n"
                        "Is the candidate step correct?"
                    )
                    label = int(rating == 1)
                    yield Stage1RExample(
                            example_id=(
                                f"prm800k:{path.stem}:{row_index}:"
                                f"{step_index}:{completion_index}"
                            ),
                            family="prm800k",
                            input_text=input_text,
                            target_text="correct" if label else "incorrect",
                            session_id=f"prm800k:{problem_id}",
                            task_context="prm800k:step-verification",
                            timestamp=step_index,
                            support_spans=[],
                            support_item_ids=[],
                            retrieval_target_ids=[],
                            encode_target=None,
                            verifier_label=label,
                            relation_label=None,
                            boundary_label=None,
                            reset_pfc=True,
                            reset_working_memory=True,
                            reset_fast_weights=True,
                            reset_episodic_memory=True,
                    )
                chosen = step.get("chosen_completion")
                if chosen is not None:
                    prior_steps.append(completions[chosen]["text"].strip())
                elif step.get("human_completion"):
                    human_completion = step["human_completion"]
                    text = (
                        human_completion["text"]
                        if isinstance(human_completion, dict)
                        else human_completion
                    )
                    prior_steps.append(text.strip())


def parse_prm800k(path: Path) -> list[Stage1RExample]:
    return list(iter_prm800k(path))


def prm800k_stage1r_splits(
    raw_files: list[Path], *, seed: int = 1729
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    limits = {("train", 0): 50_000, ("train", 1): 50_000}
    limits.update({("dev", 0): 5_000, ("dev", 1): 5_000})
    heaps: dict[tuple[str, int], list[tuple[int, int, Stage1RExample]]] = {
        key: [] for key in limits
    }
    counter = 0
    for path in raw_files:
        for example in iter_prm800k(path):
            problem_hash = example.session_id.rsplit(":", 1)[-1]
            split = "dev" if int(problem_hash[:8], 16) % 10 == 0 else "train"
            label = int(example.verifier_label)
            key = (split, label)
            rank = int.from_bytes(
                hashlib.sha256(f"{seed}:{example.example_id}".encode()).digest(),
                "big",
            )
            item = (-rank, counter, example)
            counter += 1
            heap = heaps[key]
            if len(heap) < limits[key]:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    for key, limit in limits.items():
        if len(heaps[key]) < limit:
            raise ValueError(f"PRM800K has only {len(heaps[key])} examples for {key}")
    splits = {"train": [], "dev": [], "test": []}
    for (split, _label), heap in heaps.items():
        splits[split].extend(item[2] for item in heap)
    for split in ("train", "dev"):
        splits[split].sort(
            key=lambda example: hashlib.sha256(
                f"{seed}:{example.example_id}".encode()
            ).digest()
        )
    return splits, raw_files
