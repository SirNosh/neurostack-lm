from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.stage1r.babi import babi_stage1r_splits
from src.stage1r.babilong import babilong_stage1r_training_split
from src.stage1r.data import build_manifest
from src.stage1r.fewrel import fewrel_stage1r_splits
from src.stage1r.prm800k import prm800k_stage1r_splits


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
    parser.add_argument(
        "--prm800k-dir", type=Path, default=Path("data/raw/prm800k/data")
    )
    parser.add_argument(
        "--fewrel-dir", type=Path, default=Path("data/raw/fewrel/data")
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
    if args.prm800k_dir.exists():
        prm_files = sorted(args.prm800k_dir.glob("phase*_train.jsonl"))
        prm_splits, prm_files = prm800k_stage1r_splits(prm_files)
        prm_manifest = build_manifest(
            prm_splits,
            source_url="https://github.com/openai/prm800k",
            source_revision="7ecc794703b2877f63226f2477a49b34f9b25163",
            raw_files=prm_files,
            split_procedure=(
                "Use ratings +1 as positive and -1 as negative; omit neutral and "
                "flagged completions. Assign complete source problems to dev when "
                "the first 32 bits of SHA-256(problem) modulo 10 equal zero; all "
                "other problems are train. Within split/class, select by "
                "SHA-256('1729:'+example_id): 50,000/class train and 5,000/class dev."
            ),
        )
        prm_manifest["selected_problem_ids"] = {
            split: sorted({item.session_id for item in examples})
            for split, examples in prm_splits.items()
        }
        prm_output = args.output.with_name("prm800k_stage1r.json")
        prm_output.write_text(
            json.dumps(prm_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(prm_manifest["counts"], sort_keys=True))
    if args.fewrel_dir.exists():
        train_path = args.fewrel_dir / "train_wiki.json"
        heldout_path = args.fewrel_dir / "val_wiki.json"
        fewrel_splits, fewrel_files = fewrel_stage1r_splits(
            train_path, heldout_path
        )
        fewrel_manifest = build_manifest(
            fewrel_splits,
            source_url="https://github.com/thunlp/FewRel",
            source_revision="278a2315d2138810a379cd8d5718914dc56e2582",
            raw_files=fewrel_files,
            split_procedure=(
                "FewRel 1.0 official 64 train_wiki relations are meta-train and "
                "official 16 val_wiki relations are held out. Freeze 1,000 seeded "
                "5-way episodes for each of 1-shot and 5-shot in both partitions; "
                "each episode has one query per relation. Evaluation permits no "
                "gradient updates."
            ),
        )
        train_relations = json.loads(train_path.read_text(encoding="utf-8"))
        heldout_relations = json.loads(heldout_path.read_text(encoding="utf-8"))
        fewrel_manifest["relation_partitions"] = {
            "meta_train": sorted(train_relations),
            "heldout": sorted(heldout_relations),
        }
        fewrel_manifest["episode_protocol"] = {
            "ways": 5,
            "shots": [1, 5],
            "episodes_per_shot_per_partition": 1000,
            "queries_per_relation": 1,
            "evaluation_gradient_updates": False,
        }
        fewrel_output = args.output.with_name("fewrel_stage1r.json")
        fewrel_output.write_text(
            json.dumps(fewrel_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(fewrel_manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
