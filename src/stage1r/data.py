from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ADAPTER_VERSION = "stage1r-v1"


@dataclass
class Stage1RExample:
    example_id: str
    family: str
    input_text: str
    target_text: str
    session_id: str
    task_context: str
    timestamp: int
    support_spans: list[tuple[int, int]]
    support_item_ids: list[str]
    retrieval_target_ids: list[str]
    encode_target: bool | None
    verifier_label: int | None
    relation_label: str | None
    boundary_label: int | None
    reset_pfc: bool
    reset_working_memory: bool
    reset_fast_weights: bool
    reset_episodic_memory: bool

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(examples: Iterable[Stage1RExample], seed: int) -> list[Stage1RExample]:
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"{seed}:{example.example_id}".encode()
        ).digest(),
    )


def build_manifest(
    splits: Mapping[str, Sequence[Stage1RExample]],
    *,
    source_url: str,
    source_revision: str,
    raw_files: Sequence[Path],
    split_procedure: str,
) -> dict:
    if not raw_files:
        raise ValueError("a frozen manifest requires at least one raw source file")
    selected_ids = {
        split: [example.example_id for example in examples]
        for split, examples in sorted(splits.items())
    }
    formatted_hashes = {
        split: [example.sha256() for example in examples]
        for split, examples in sorted(splits.items())
    }
    label_counts: dict[str, dict[str, int]] = {}
    for split, examples in sorted(splits.items()):
        counts: dict[str, int] = {}
        for example in examples:
            labels = {
                "family": example.family,
                "task_context": example.task_context,
                "encode_target": example.encode_target,
                "verifier_label": example.verifier_label,
                "relation_label": example.relation_label,
                "boundary_label": example.boundary_label,
            }
            for name, value in labels.items():
                key = f"{name}={value}"
                counts[key] = counts.get(key, 0) + 1
        label_counts[split] = dict(sorted(counts.items()))
    return {
        "source_url": source_url,
        "source_revision": source_revision,
        "raw_file_sha256": {
            path.name: sha256_file(path) for path in sorted(raw_files)
        },
        "adapter_version": ADAPTER_VERSION,
        "split_procedure": split_procedure,
        "selected_row_ids": selected_ids,
        "formatted_example_hashes": formatted_hashes,
        "counts": {split: len(items) for split, items in selected_ids.items()},
        "label_frequency_summary": label_counts,
    }
