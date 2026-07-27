import json
from pathlib import Path

import torch

from src.stage1r.fewrel import (
    FewRelEpisodeFastLearner,
    build_fewrel_episodes,
    build_fewrel_v2_episodes,
    partition_training_relations,
)
from src.stage1r.prm800k import parse_prm800k


def test_prm800k_adapter_emits_binary_non_neutral_steps(tmp_path: Path):
    raw = tmp_path / "phase2_train.jsonl"
    row = {
        "question": {"problem": "Compute 1+1."},
        "label": {
            "steps": [
                {
                    "completions": [
                        {"text": "It is 2.", "rating": 1, "flagged": False},
                        {"text": "It is 3.", "rating": -1, "flagged": False},
                        {"text": "Consider addition.", "rating": 0, "flagged": False},
                    ],
                    "chosen_completion": None,
                    "human_completion": {
                        "text": "A human suggests calculating directly.",
                        "rating": None,
                    },
                },
                {
                    "completions": [
                        {"text": "Therefore 1+1=2.", "rating": 1, "flagged": False}
                    ],
                    "chosen_completion": 0,
                    "human_completion": None,
                },
            ]
        },
    }
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples = parse_prm800k(raw)
    assert [example.verifier_label for example in examples] == [1, 0, 1]
    assert "A human suggests" in examples[-1].input_text
    assert len({example.session_id for example in examples}) == 1
    assert all("ground_truth" not in example.input_text for example in examples)


def _fewrel_relations():
    return {
        f"P{relation}": [
            {
                "tokens": ["Alice", "knows", f"entity-{index}"],
                "h": ["Alice", "Q1", [[0]]],
                "t": [f"entity-{index}", f"Q{index}", [[2]]],
            }
            for index in range(8)
        ]
        for relation in range(5)
    }


def test_fewrel_episode_resets_once_and_queries_target_supports():
    examples = build_fewrel_episodes(
        _fewrel_relations(),
        split="heldout",
        shot=1,
        episode_count=1,
    )
    assert len(examples) == 10
    assert sum(example.reset_fast_weights for example in examples) == 1
    supports = {example.example_id for example in examples if example.encode_target}
    queries = [example for example in examples if not example.encode_target]
    assert len(supports) == len(queries) == 5
    assert all(set(query.retrieval_target_ids) <= supports for query in queries)


def test_fewrel_five_shot_has_twenty_five_supports_and_five_queries():
    examples = build_fewrel_episodes(
        _fewrel_relations(),
        split="meta-train",
        shot=5,
        episode_count=1,
    )
    assert len(examples) == 30
    assert sum(bool(example.encode_target) for example in examples) == 25


def test_fewrel_v2_support_labels_visible_and_query_labels_hidden():
    examples = build_fewrel_v2_episodes(
        _fewrel_relations(), split="test", shot=1, episode_count=1
    )
    supports = [item for item in examples if item.encode_target]
    queries = [item for item in examples if not item.encode_target]
    assert all(
        f"Relation label: {item.relation_label}" in item.input_text
        for item in supports
    )
    assert all("Relation label:" not in item.input_text for item in queries)
    assert all("Choose relation label: A, B, C, D, or E." in item.input_text for item in queries)


def test_fewrel_v2_permutation_changes_between_episodes():
    examples = build_fewrel_v2_episodes(
        _fewrel_relations(), split="test", shot=1, episode_count=2
    )
    episodes = {}
    for item in examples:
        if item.encode_target:
            entity = item.input_text.split("entity-", 1)[1].splitlines()[0]
            episodes.setdefault(item.session_id, {})[entity] = item.relation_label
    assert len(episodes) == 2
    assert list(episodes.values())[0] != list(episodes.values())[1]


def test_fewrel_v2_relation_partitions_are_disjoint():
    train, dev = partition_training_relations(
        [f"P{index}" for index in range(64)]
    )
    test = {f"T{index}" for index in range(16)}
    assert len(train) == 48
    assert len(dev) == 16
    assert not (set(train) & set(dev))
    assert not ((set(train) | set(dev)) & test)


def test_fewrel_v2_fast_update_uses_episode_label_embedding():
    learner = FewRelEpisodeFastLearner(representation_dim=8, rank=2)
    state = learner.initialize(1, device="cpu", dtype=torch.float32)
    support = torch.randn(1, 8)
    captured = []
    handle = learner.fast_weights.value_rank.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    learner.update_support(state, support, ["C"], da=torch.ones(1))
    handle.remove()
    assert torch.equal(captured[0], learner.label_values(["C"]))


def test_fewrel_v2_evaluation_updates_only_fast_state():
    learner = FewRelEpisodeFastLearner(representation_dim=8, rank=2)
    learner.set_evaluation_mode()
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in learner.named_parameters()
    }
    state = learner.initialize(1, device="cpu", dtype=torch.float32)
    updated = learner.update_support(
        state, torch.randn(1, 8), ["A"], da=torch.ones(1)
    )
    assert updated.u.grad_fn is None
    assert updated.v.grad_fn is None
    assert all(
        torch.equal(parameter, parameters[name])
        for name, parameter in learner.named_parameters()
    )
