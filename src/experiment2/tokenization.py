from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import torch

from .data import Experiment2Example


@dataclass
class TokenizedExperiment2Example:
    example: Experiment2Example
    input_ids: list[int]
    prompt_length: int
    fact_token_spans: list[tuple[int, int]]
    question_token_span: tuple[int, int] | None


def _token_span(
    offsets: list[tuple[int, int]], span: tuple[int, int]
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > span[0] and start < span[1]
    ]
    if not indices:
        raise ValueError(f"character span {span} maps to no tokens")
    return indices[0], indices[-1] + 1


def tokenize_example(
    example: Experiment2Example, tokenizer, *, max_length: int = 512
) -> TokenizedExperiment2Example:
    prompt = tokenizer(
        example.input_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    answer = tokenizer(example.target_text, add_special_tokens=False)
    prompt_ids = list(prompt["input_ids"])
    answer_ids = list(answer["input_ids"])
    if not answer_ids and example.target_text:
        raise ValueError("non-empty target produced no tokens")
    input_ids = (prompt_ids + answer_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    offsets = [tuple(pair) for pair in prompt["offset_mapping"]]
    fact_spans = [_token_span(offsets, span) for span in example.fact_spans]
    question_span = (
        _token_span(offsets, example.question_span)
        if example.question_span is not None
        else None
    )
    if any(end > prompt_length for _, end in fact_spans):
        raise ValueError("fact span was truncated")
    return TokenizedExperiment2Example(
        example, input_ids, prompt_length, fact_spans, question_span
    )


def fit_example_to_token_budget(
    example: Experiment2Example, tokenizer, *, max_length: int = 512
) -> Experiment2Example:
    """Drop oldest distractors only; never remove support facts or the question."""
    if not example.fact_spans:
        return example
    if (
        len(tokenizer(example.input_text, add_special_tokens=False)["input_ids"])
        + len(tokenizer(example.target_text, add_special_tokens=False)["input_ids"])
        <= max_length
    ):
        return example
    facts = [example.input_text[start:end] for start, end in example.fact_spans]
    support = set(example.support_fact_indices)
    kept = list(range(len(facts)))
    question = example.input_text[example.question_span[0] :] if example.question_span else ""
    while True:
        context = "\n".join(facts[index] for index in kept)
        text = f"{context}\n{question}" if context else question
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"]) + len(
            tokenizer(example.target_text, add_special_tokens=False)["input_ids"]
        )
        if length <= max_length:
            spans, offset = [], 0
            for index in kept:
                spans.append((offset, offset + len(facts[index])))
                offset += len(facts[index]) + 1
            remap = {old: new for new, old in enumerate(kept)}
            return replace(
                example,
                input_text=text,
                fact_spans=spans,
                question_span=(len(context) + 1, len(text)),
                support_fact_indices=[remap[index] for index in sorted(support)],
            )
        removable = next((index for index in kept if index not in support), None)
        if removable is None:
            raise ValueError("support facts and question exceed the token budget")
        kept.remove(removable)


def collate_tokenized(
    examples: list[TokenizedExperiment2Example], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    length = max(len(item.input_ids) for item in examples)
    input_ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, item in enumerate(examples):
        values = torch.tensor(item.input_ids)
        input_ids[row, : len(values)] = values
        attention_mask[row, : len(values)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": torch.tensor([item.prompt_length for item in examples]),
    }
