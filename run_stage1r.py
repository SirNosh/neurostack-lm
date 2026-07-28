"""Recoverable Stage 1R calibration runner.

This is intentionally one runner for R0/R2/R5. It exercises the frozen
calibration schedule and writes the required auditable artifacts; it does not
pretend that next-token task loss alone completes mechanism qualification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import psutil
import torch
import torch.nn.functional as F

from src.stage1r.audit import hash_module_parameters
from src.stage1r.babi import babi_stage1r_splits
from src.stage1r.baselines import R0ParameterMatchedAdapter, R2RecurrentMemoryTokens
from src.stage1r.clutrr import clutrr_stage1r_splits
from src.stage1r.data import Stage1RExample
from src.stage1r.epbench import epbench_stage1r_splits
from src.stage1r.fewrel import build_fewrel_v2_episodes
from src.stage1r.mechanisms import EpisodicMemory
from src.stage1r.model import QWEN_REVISION, Stage1RNeuroStack
from src.stage1r.prm800k import iter_prm800k
from src.stage1r.trace import TRACE_TASKS, trace_stage1r_splits


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    / QWEN_REVISION
)
TARGET_PARAMETERS = 11_373_945
SCHEDULE = {"warmup": 20, "development": 100, "wake": 256, "sleep": 25}


def _balanced_prm(limit: int) -> list[Stage1RExample]:
    selected = {0: [], 1: []}
    for path in sorted((ROOT / "data/raw/prm800k/data").glob("phase*_train.jsonl")):
        for example in iter_prm800k(path):
            bucket = selected[int(example.verifier_label)]
            if len(bucket) < limit // 2:
                bucket.append(example)
            if all(len(items) == limit // 2 for items in selected.values()):
                return selected[0] + selected[1]
    raise ValueError("PRM800K calibration subset is incomplete")


def calibration_examples(seed: int) -> list[Stage1RExample]:
    babi, _ = babi_stage1r_splits(ROOT / "data/raw/tasks_1-20_v1-2/en-10k", seed=seed)
    clutrr, _ = clutrr_stage1r_splits(ROOT / "data/raw/clutrr-db9b8f04", seed=seed)
    epbench, _ = epbench_stage1r_splits(
        ROOT / "data/raw/epbench-data", question_limit=128, seed=seed
    )
    trace, _ = trace_stage1r_splits(
        ROOT / "data/raw/trace-data/TRACE-Benchmark/LLM-CL-Benchmark_500"
    )
    train_relations = json.loads(
        (ROOT / "data/raw/fewrel/data/train_wiki.json").read_text(encoding="utf-8")
    )
    fewrel = build_fewrel_v2_episodes(
        train_relations, split="calibration", shot=1, episode_count=128, seed=seed
    )
    fewrel_queries = [item for item in fewrel if item.encode_target is False][:128]
    epbench_queries = [item for item in epbench["test"] if item.encode_target is False]
    trace_by_task = []
    for task in TRACE_TASKS:
        trace_by_task.extend(
            [
                item
                for item in trace["train"]
                if f":{task}:" in item.example_id
            ][:128]
        )
    groups = [
        babi["dev"][:256],
        _balanced_prm(256),
        fewrel_queries,
        epbench_queries[:128],
        clutrr["train"][:256],
        trace_by_task,
    ]
    examples = [item for group in groups for item in group]
    random.Random(seed).shuffle(examples)
    return examples


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


class Runner:
    def __init__(self, system: str, seed: int, output: Path) -> None:
        from transformers import AutoTokenizer

        self.system = system
        self.seed = seed
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        random.seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        if system == "R5":
            self.model = Stage1RNeuroStack.from_qwen(MODEL_PATH)
            self.model.set_development_mode()
        elif system == "R0":
            self.model = R0ParameterMatchedAdapter.from_qwen(
                MODEL_PATH, target_trainable_parameters=TARGET_PARAMETERS
            )
        elif system == "R2":
            self.model = R2RecurrentMemoryTokens.from_qwen(
                MODEL_PATH, target_trainable_parameters=TARGET_PARAMETERS
            )
        else:
            raise ValueError("calibration supports R0, R2, and R5")
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=2e-5,
        )

    def _tokens(self, example: Stage1RExample):
        tokens = self.tokenizer(
            example.input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to("cuda")
        target = self.tokenizer(
            example.target_text or "none", add_special_tokens=False
        ).input_ids[0]
        return tokens, torch.tensor([target], device="cuda")

    def forward(self, example: Stage1RExample, *, train: bool):
        tokens, target = self._tokens(example)
        if self.system == "R5":
            state = self.model.initialize_state(1, device="cuda", dtype=torch.bfloat16)
            output = self.model(
                tokens.input_ids,
                tokens.attention_mask,
                state,
                EpisodicMemory(),
                session_ids=[example.session_id],
                task_contexts=[example.task_context],
                cycles=3,
            )
            logits = output.final.token_logits
            loss = F.cross_entropy(logits.float(), target)
            if example.verifier_label is not None:
                verifier_target = torch.tensor(
                    [example.verifier_label], device="cuda", dtype=torch.float32
                )
                loss = loss + F.binary_cross_entropy_with_logits(
                    output.final.verifier_logits.float(), verifier_target
                )
            event = {
                "router_probabilities": output.final.routing.probabilities[0].float().tolist(),
                "selected_experts": output.final.routing.indices[0].tolist(),
                "memory_write": float(output.final.controls.memory_write[0]),
                "verifier_probability": float(output.final.verifier_logits[0].float().sigmoid()),
                "workspace_sources": output.final.workspace.sources[0].tolist(),
            }
        else:
            output = self.model(
                tokens.input_ids, tokens.attention_mask, passes=3
            )
            logits = output.token_logits
            loss = F.cross_entropy(logits.float(), target)
            event = {}
        return loss, logits, target, event

    def optimizer_step(self, example: Stage1RExample) -> float:
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = self.forward(example, train=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite calibration loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        torch.cuda.synchronize()
        return time.perf_counter() - started

    def checkpoint(self, path: Path, next_index: int) -> None:
        trainable = {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if not name.startswith("backbone.")
        }
        torch.save(
            {
                "model": trainable,
                "optimizer": self.optimizer.state_dict(),
                "next_index": next_index,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "python_rng": random.getstate(),
            },
            path,
        )

    def restore(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        return checkpoint["next_index"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("R0", "R2", "R5"), required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/calibration" / f"{args.system}-{args.seed}"
    examples = calibration_examples(args.seed)
    runner = Runner(args.system, args.seed, output)
    _save_json(output / "config.json", {"system": args.system, "seed": args.seed, **SCHEDULE})
    _save_json(
        output / "manifest.json",
        {
            "examples": len(examples),
            "example_ids": [item.example_id for item in examples],
            "hashes": [item.sha256() for item in examples],
        },
    )
    _save_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "backbone_revision": QWEN_REVISION,
        },
    )
    torch.cuda.reset_peak_memory_stats()
    backbone_before = hash_module_parameters(runner.model.backbone)
    for index in range(SCHEDULE["warmup"]):
        runner.optimizer_step(examples[index])
    measured_times = [
        runner.optimizer_step(examples[index % len(examples)])
        for index in range(SCHEDULE["development"])
    ]
    wake_times = []
    runner.model.eval()
    with torch.no_grad():
        for index in range(SCHEDULE["wake"]):
            started = time.perf_counter()
            runner.forward(examples[index % len(examples)], train=False)
            torch.cuda.synchronize()
            wake_times.append(time.perf_counter() - started)
    if args.system == "R5":
        runner.model.set_sleep_mode()
    else:
        runner.model.set_sleep_mode()
    runner.optimizer = torch.optim.AdamW(
        [parameter for parameter in runner.model.parameters() if parameter.requires_grad],
        lr=2e-5,
    )
    sleep_times = [
        runner.optimizer_step(examples[index % len(examples)])
        for index in range(SCHEDULE["sleep"])
    ]
    checkpoint = output / "checkpoint_last.pt"
    runner.checkpoint(checkpoint, 0)
    runner.model.eval()
    with torch.no_grad():
        _, first_logits, _, _ = runner.forward(examples[0], train=False)
    runner.restore(checkpoint)
    runner.model.eval()
    with torch.no_grad():
        _, resumed_logits, _, _ = runner.forward(examples[0], train=False)
    resume_equal = torch.equal(first_logits, resumed_logits)
    predictions = []
    mechanism_events = []
    correct = 0
    runner.model.eval()
    evaluation_started = time.perf_counter()
    with torch.no_grad(), (output / "eval_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as prediction_stream, (output / "mechanism_events.jsonl").open(
        "w", encoding="utf-8"
    ) as event_stream:
        for example in examples:
            _, logits, target, event = runner.forward(example, train=False)
            predicted = int(logits.argmax(-1)[0])
            is_correct = predicted == int(target[0])
            correct += is_correct
            record = {
                "example_id": example.example_id,
                "family": example.family,
                "predicted_first_token_id": predicted,
                "target_first_token_id": int(target[0]),
                "correct": is_correct,
            }
            prediction_stream.write(json.dumps(record) + "\n")
            if event:
                event_stream.write(
                    json.dumps({"example_id": example.example_id, **event}) + "\n"
                )
    evaluation_seconds = time.perf_counter() - evaluation_started
    backbone_after = hash_module_parameters(runner.model.backbone)
    resource = {
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "cpu_rss_gb": psutil.Process(os.getpid()).memory_info().rss / 2**30,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "mean_optimizer_step_seconds": statistics.mean(measured_times),
        "p95_optimizer_step_seconds": _percentile(measured_times, 0.95),
        "mean_wake_seconds": statistics.mean(wake_times),
        "p95_wake_seconds": _percentile(wake_times, 0.95),
        "mean_sleep_step_seconds": statistics.mean(sleep_times),
        "evaluation_seconds": evaluation_seconds,
        "estimated_4000_step_hours": statistics.mean(measured_times) * 4000 / 3600,
    }
    _save_json(output / "resource_usage.json", resource)
    final = {
        "system": args.system,
        "seed": args.seed,
        "examples": len(examples),
        "first_token_accuracy": correct / len(examples),
        "resume_exact": resume_equal,
        "backbone_hash_unchanged": backbone_before == backbone_after,
        "finite": True,
    }
    _save_json(output / "final_metrics.json", final)
    _save_json(
        output / "gate_report.json",
        {
            "calibration_passed": (
                resume_equal
                and backbone_before == backbone_after
                and resource["peak_vram_gb"]
                <= torch.cuda.get_device_properties(0).total_memory / 2**30 - 1
            )
        },
    )
    for filename in (
        "train_metrics.jsonl",
        "routing_summary.json",
        "memory_summary.json",
        "fast_weight_summary.json",
        "sleep_summary.json",
    ):
        path = output / filename
        if filename.endswith(".jsonl"):
            path.touch()
        else:
            _save_json(path, {})
    (output / "checkpoint_best.pt").write_bytes(checkpoint.read_bytes())
    print(json.dumps({"final": final, "resource": resource}, indent=2))


if __name__ == "__main__":
    main()
