from pathlib import Path

import torch

from src.neurostack_pilot import NeuroStack, parse_babi, parameter_count, GenericAdapter


def test_babi_parser_preserves_supporting_facts():
    path = next(
        Path("data/raw/tasks_1-20_v1-2/en-10k").glob("qa1_*_train.txt")
    )
    example = parse_babi(path, 1)[0]
    assert example.target_text
    assert example.supporting_fact_indices
    assert len(example.should_encode) == len(example.facts)


def test_full_shapes_and_parameter_match():
    full = NeuroStack(20)
    generic = GenericAdapter(20, parameter_count(full))
    logits, auxiliary = full(torch.randn(5, 896), torch.randn(896))
    assert logits.shape == (20,)
    assert auxiliary["gate_logits"].shape == (5,)
    difference = abs(parameter_count(full) - parameter_count(generic))
    assert difference / parameter_count(full) < 0.02

