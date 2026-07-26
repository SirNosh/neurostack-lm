from pathlib import Path
import json

from src.neurostack_pilot import run


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)


if __name__ == "__main__":
    result = run(
        {
            "root": str(ROOT),
            "model_path": str(MODEL_PATH),
            "seed": 1729,
            "train_per_task": 240,
            "test_per_task": 100,
            "bootstrap_per_task": 40,
            "bootstrap_epochs": 2,
            "sleep_epochs": 2,
            "old_replay_size": 100,
            "encoder_batch_size": 128,
        }
    )
    print(json.dumps(result, indent=2))

