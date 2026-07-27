from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.stage1r.babi import babi_stage1r_splits
from src.stage1r.babilong import babilong_stage1r_training_split
from src.stage1r.data import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--babi-dir",
        type=Path,
        default=Path("data/raw/tasks_1-20_v1-2/en-10k"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/babi_stage1r.json")
    )
    parser.add_argument(
        "--babilong-dir",
        type=Path,
        default=Path("data/raw/babilong-train-5k-samples"),
    )
    args = parser.parse_args()
    splits, raw_files = babi_stage1r_splits(args.babi_dir)
    manifest = build_manifest(
        splits,
        source_url="https://research.facebook.com/downloads/babi/",
        source_revision="tasks_1-20_v1-2",
        raw_files=raw_files,
        split_procedure=(
            "Tasks qa1-qa5. For each official en-10k train file, order examples by "
            "SHA-256('1729:'+example_id), select first 5,000 train and next 500 dev. "
            "Keep the complete official test file as test."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    if args.babilong_dir.exists():
        long_splits, long_files = babilong_stage1r_training_split(args.babilong_dir)
        long_manifest = build_manifest(
            long_splits,
            source_url=(
                "https://huggingface.co/datasets/"
                "RMT-team/babilong-train-5k-samples"
            ),
            source_revision="b3513ef7c25c54ce706054530d47668c532019d6",
            raw_files=long_files,
            split_procedure=(
                "Official 4k training release, qa1-qa5. For each task, order the "
                "5,000 rows by SHA-256('1729:'+example_id) and select the first 2,000."
            ),
        )
        long_output = args.output.with_name("babilong_4k_stage1r.json")
        long_output.write_text(
            json.dumps(long_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(long_manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
