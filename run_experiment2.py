from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment2.a1_runner import A1Runner


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
)
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("A1",), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dev-limit", type=int)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "outputs" / "experiment2" / f"A1-{args.seed}"
    runner = A1Runner(
        model_path=MODEL_ROOT / REVISION,
        data_dir=ROOT / "data/raw/tasks_1-20_v1-2/en-10k",
        output_dir=output,
        seed=args.seed,
        max_steps=args.max_steps,
        dev_limit=args.dev_limit,
    )
    if args.resume:
        runner.resume()
    runner.train()


if __name__ == "__main__":
    main()
