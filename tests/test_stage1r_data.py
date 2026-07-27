from pathlib import Path

from src.stage1r.babi import parse_babi
from src.stage1r.babilong import adapt_babilong_rows
from src.stage1r.data import build_manifest


def test_babi_adapter_preserves_support_spans(tmp_path: Path):
    raw = tmp_path / "qa1_example_train.txt"
    raw.write_text(
        "1 Mary moved to the bathroom.\n"
        "2 John went to the hall.\n"
        "3 Where is Mary?\tbathroom\t1\n",
        encoding="utf-8",
    )
    example = parse_babi(raw)[0]
    assert example.target_text == "bathroom"
    assert example.input_text[slice(*example.support_spans[0])] == (
        "Mary moved to the bathroom."
    )
    assert example.support_item_ids == ["qa1:story-1:fact-1"]


def test_babilong_adapter_uses_official_release_fields():
    example = adapt_babilong_rows(
        [{"input": "Long distractor text.", "question": "Where?", "target": "office"}],
        task="qa1",
        length="4k",
        split="train",
    )[0]
    assert example.example_id == "babilong:4k:qa1:train:0"
    assert example.target_text == "office"
    assert example.support_item_ids == []


def test_manifest_is_deterministic_and_records_raw_hash(tmp_path: Path):
    raw = tmp_path / "qa1_example_train.txt"
    raw.write_text("1 Mary moved home.\n2 Where is Mary?\thome\t1\n", encoding="utf-8")
    examples = parse_babi(raw)
    first = build_manifest(
        {"train": examples},
        source_url="official",
        source_revision="v1",
        raw_files=[raw],
        split_procedure="fixture",
    )
    second = build_manifest(
        {"train": examples},
        source_url="official",
        source_revision="v1",
        raw_files=[raw],
        split_procedure="fixture",
    )
    assert first == second
    assert first["counts"] == {"train": 1}
