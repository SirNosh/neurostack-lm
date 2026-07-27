from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch
from torch import nn

from .data import Stage1RExample
from .mechanisms import FastMatrixState, FastWeightAdapter


EPISODE_LABELS = ("A", "B", "C", "D", "E")


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


def partition_training_relations(
    relation_ids: Sequence[str], *, seed: int = 1729
) -> tuple[list[str], list[str]]:
    if len(relation_ids) != 64:
        raise ValueError("FewRel v2 expects the official 64 training relations")
    ordered = sorted(
        relation_ids,
        key=lambda relation_id: hashlib.sha256(
            f"{seed}:{relation_id}".encode()
        ).digest(),
    )
    return ordered[:48], ordered[48:]


def build_fewrel_v2_episodes(
    relations: Mapping[str, Sequence[Mapping]],
    *,
    split: str,
    shot: int,
    episode_count: int,
    seed: int = 1729,
) -> list[Stage1RExample]:
    if shot not in (1, 5):
        raise ValueError("FewRel v2 freezes only 1-shot and 5-shot episodes")
    relation_ids = sorted(relations)
    if len(relation_ids) < 5:
        raise ValueError("FewRel episodes require at least five relations")
    rng = random.Random(f"v2:{seed}:{split}:{shot}")
    output: list[Stage1RExample] = []
    previous_permutation: tuple[str, ...] | None = None
    for episode_index in range(episode_count):
        chosen_relations = rng.sample(relation_ids, 5)
        permutation = tuple(rng.sample(EPISODE_LABELS, len(EPISODE_LABELS)))
        if permutation == previous_permutation:
            permutation = permutation[1:] + permutation[:1]
        previous_permutation = permutation
        local_labels = dict(zip(chosen_relations, permutation))
        support_ids_by_relation: dict[str, list[str]] = {}
        supports = []
        queries = []
        for relation_index, relation_id in enumerate(chosen_relations):
            chosen_instances = rng.sample(list(relations[relation_id]), shot + 1)
            local_label = local_labels[relation_id]
            support_ids = []
            for support_index, instance in enumerate(chosen_instances[:shot]):
                item_id = (
                    f"fewrel-v2:{split}:{shot}shot:{episode_index}:"
                    f"support-{relation_index}-{support_index}"
                )
                support_ids.append(item_id)
                supports.append(
                    (item_id, local_label, instance, True, [])
                )
            support_ids_by_relation[relation_id] = support_ids
            queries.append(
                (
                    f"fewrel-v2:{split}:{shot}shot:{episode_index}:"
                    f"query-{relation_index}",
                    local_label,
                    chosen_instances[-1],
                    False,
                    support_ids,
                )
            )
        rng.shuffle(supports)
        rng.shuffle(queries)
        session_id = f"fewrel-v2:{split}:{shot}shot:{episode_index}"
        for timestamp, (
            item_id,
            local_label,
            instance,
            is_support,
            retrieval_ids,
        ) in enumerate(supports + queries):
            first = timestamp == 0
            prompt = _sentence(instance)
            if is_support:
                prompt += f"\nRelation label: {local_label}"
            else:
                prompt += "\nChoose relation label: A, B, C, D, or E."
            output.append(
                Stage1RExample(
                    example_id=item_id,
                    family="fewrel",
                    input_text=prompt,
                    target_text=local_label,
                    session_id=session_id,
                    task_context=(
                        f"fewrel-v2:5way:{shot}shot:"
                        f"{'support' if is_support else 'query'}"
                    ),
                    timestamp=timestamp,
                    support_spans=[],
                    support_item_ids=[],
                    retrieval_target_ids=retrieval_ids,
                    encode_target=is_support,
                    verifier_label=None,
                    relation_label=local_label,
                    boundary_label=None,
                    reset_pfc=first,
                    reset_working_memory=first,
                    reset_fast_weights=first,
                    reset_episodic_memory=first,
                )
            )
    return output


def fewrel_stage1r_v2_splits(
    train_path: Path,
    heldout_path: Path,
    *,
    episodes_per_shot: int = 1_000,
    seed: int = 1729,
) -> tuple[
    dict[str, list[Stage1RExample]],
    list[Path],
    dict[str, list[str]],
]:
    official_train = json.loads(train_path.read_text(encoding="utf-8"))
    official_test = json.loads(heldout_path.read_text(encoding="utf-8"))
    meta_train_ids, meta_dev_ids = partition_training_relations(
        list(official_train), seed=seed
    )
    partitions = {
        "meta_train": meta_train_ids,
        "meta_dev": meta_dev_ids,
        "heldout_test": sorted(official_test),
    }
    if set().union(*map(set, partitions.values())) != set(official_train) | set(
        official_test
    ):
        raise ValueError("FewRel v2 relation partitions are not exhaustive")
    if sum(map(len, partitions.values())) != len(
        set().union(*map(set, partitions.values()))
    ):
        raise ValueError("FewRel v2 relation partitions overlap")

    relation_sets = {
        "train": {key: official_train[key] for key in meta_train_ids},
        "dev": {key: official_train[key] for key in meta_dev_ids},
        "test": official_test,
    }
    splits: dict[str, list[Stage1RExample]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    for output_split, relations in relation_sets.items():
        for shot in (1, 5):
            splits[output_split].extend(
                build_fewrel_v2_episodes(
                    relations,
                    split=output_split,
                    shot=shot,
                    episode_count=episodes_per_shot,
                    seed=seed,
                )
            )
    return splits, [train_path, heldout_path], partitions


class FewRelEpisodeFastLearner(nn.Module):
    """Episode-local key/value plasticity for FewRel v2."""

    def __init__(self, representation_dim: int = 256, rank: int = 8) -> None:
        super().__init__()
        self.label_embeddings = nn.Embedding(len(EPISODE_LABELS), representation_dim)
        self.fast_weights = FastWeightAdapter(
            representation_dim, representation_dim, rank
        )

    def initialize(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> FastMatrixState:
        return self.fast_weights.initialize(
            batch_size, device=device, dtype=dtype
        )

    def label_values(self, labels: Sequence[str]) -> torch.Tensor:
        try:
            indices = [EPISODE_LABELS.index(label) for label in labels]
        except ValueError as error:
            raise ValueError("FewRel v2 labels must be A-E") from error
        return self.label_embeddings(
            torch.tensor(indices, device=self.label_embeddings.weight.device)
        )

    @torch.no_grad()
    def update_support(
        self,
        state: FastMatrixState,
        support_representations: torch.Tensor,
        labels: Sequence[str],
        *,
        da: torch.Tensor,
        ach: torch.Tensor | None = None,
        ne: torch.Tensor | None = None,
        da_permutation: torch.Tensor | None = None,
    ) -> FastMatrixState:
        if da_permutation is not None:
            da = da[da_permutation]
        ones = torch.ones_like(da)
        values = self.label_values(labels).to(
            device=support_representations.device,
            dtype=support_representations.dtype,
        )
        return self.fast_weights.update(
            state,
            support_representations,
            values,
            da=da,
            ach=ones if ach is None else ach,
            ne=ones if ne is None else ne,
        )

    def logits(
        self, query_representations: torch.Tensor, state: FastMatrixState
    ) -> torch.Tensor:
        recalled = self.fast_weights.apply(query_representations, state)
        return recalled @ self.label_embeddings.weight.to(recalled.dtype).T

    def set_evaluation_mode(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
