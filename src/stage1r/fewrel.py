from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from .data import Stage1RExample


def _sentence(instance: Mapping) -> str:
    tokens = instance["tokens"]
    head = instance["h"]
    tail = instance["t"]
    return (
        " ".join(tokens)
        + f"\nHead entity: {head[0]}\nTail entity: {tail[0]}"
    )


def build_fewrel_episodes(
    relations: Mapping[str, Sequence[Mapping]],
    *,
    split: str,
    shot: int,
    episode_count: int,
    seed: int = 1729,
) -> list[Stage1RExample]:
    if shot not in (1, 5):
        raise ValueError("Stage 1R freezes only 1-shot and 5-shot episodes")
    relation_ids = sorted(relations)
    if len(relation_ids) < 5:
        raise ValueError("FewRel episodes require at least five relations")
    rng = random.Random(f"{seed}:{split}:{shot}")
    output = []
    for episode_index in range(episode_count):
        chosen_relations = rng.sample(relation_ids, 5)
        support_ids_by_relation: dict[str, list[str]] = {}
        episode_items = []
        for relation_id in chosen_relations:
            chosen_instances = rng.sample(list(relations[relation_id]), shot + 1)
            support_ids = []
            for support_index, instance in enumerate(chosen_instances[:shot]):
                item_id = (
                    f"fewrel:{split}:{shot}shot:{episode_index}:"
                    f"{relation_id}:support-{support_index}"
                )
                support_ids.append(item_id)
                episode_items.append(
                    (item_id, relation_id, instance, True, [])
                )
            support_ids_by_relation[relation_id] = support_ids
            query_id = (
                f"fewrel:{split}:{shot}shot:{episode_index}:"
                f"{relation_id}:query"
            )
            episode_items.append(
                (
                    query_id,
                    relation_id,
                    chosen_instances[-1],
                    False,
                    support_ids,
                )
            )
        rng.shuffle(episode_items)
        support_items = [item for item in episode_items if item[3]]
        query_items = [item for item in episode_items if not item[3]]
        ordered_items = support_items + query_items
        session_id = f"fewrel:{split}:{shot}shot:{episode_index}"
        for timestamp, (
            item_id,
            relation_id,
            instance,
            is_support,
            retrieval_ids,
        ) in enumerate(ordered_items):
            first = timestamp == 0
            output.append(
                Stage1RExample(
                    example_id=item_id,
                    family="fewrel",
                    input_text=(
                        f"{_sentence(instance)}\n"
                        "What is the relation between the marked entities?"
                    ),
                    target_text=relation_id,
                    session_id=session_id,
                    task_context=(
                        f"fewrel:5way:{shot}shot:"
                        f"{'support' if is_support else 'query'}"
                    ),
                    timestamp=timestamp,
                    support_spans=[],
                    support_item_ids=[],
                    retrieval_target_ids=retrieval_ids,
                    encode_target=is_support,
                    verifier_label=None,
                    relation_label=relation_id,
                    boundary_label=None,
                    reset_pfc=first,
                    reset_working_memory=first,
                    reset_fast_weights=first,
                    reset_episodic_memory=first,
                )
            )
    return output


def fewrel_stage1r_splits(
    train_path: Path,
    heldout_path: Path,
    *,
    episodes_per_shot: int = 1_000,
    seed: int = 1729,
) -> tuple[dict[str, list[Stage1RExample]], list[Path]]:
    train_relations = json.loads(train_path.read_text(encoding="utf-8"))
    heldout_relations = json.loads(heldout_path.read_text(encoding="utf-8"))
    overlap = set(train_relations) & set(heldout_relations)
    if overlap:
        raise ValueError(f"FewRel relation leakage: {sorted(overlap)}")
    train = []
    test = []
    for shot in (1, 5):
        train.extend(
            build_fewrel_episodes(
                train_relations,
                split="meta-train",
                shot=shot,
                episode_count=episodes_per_shot,
                seed=seed,
            )
        )
        test.extend(
            build_fewrel_episodes(
                heldout_relations,
                split="heldout",
                shot=shot,
                episode_count=episodes_per_shot,
                seed=seed,
            )
        )
    return {"train": train, "dev": [], "test": test}, [train_path, heldout_path]
