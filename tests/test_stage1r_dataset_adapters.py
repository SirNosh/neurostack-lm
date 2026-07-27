import json
from pathlib import Path

from src.stage1r.fewrel import build_fewrel_episodes
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
                    "chosen_completion": 0,
                    "human_completion": None,
                }
            ]
        },
    }
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples = parse_prm800k(raw)
    assert [example.verifier_label for example in examples] == [1, 0]
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
