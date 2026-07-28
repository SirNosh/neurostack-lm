"""Recoverable common Stage 1R calibration and qualification runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import math
import shutil

import psutil
import torch
import torch.nn.functional as F

from src.stage1r.audit import hash_module_parameters
from src.stage1r.babi import babi_stage1r_splits
from src.stage1r.baselines import (
    R0ParameterMatchedAdapter,
    R1OrdinaryRAG,
    R2RecurrentMemoryTokens,
)
from src.stage1r.clutrr import clutrr_stage1r_splits
from src.stage1r.data import Stage1RExample
from src.stage1r.epbench import epbench_stage1r_splits
from src.stage1r.fewrel import build_fewrel_v2_episodes
from src.stage1r.mechanisms import EpisodicEvent, EpisodicMemory, LesionConfig
from src.stage1r.model import QWEN_REVISION, Stage1RNeuroStack
from src.stage1r.prm800k import iter_prm800k
from src.stage1r.prm800k import prm800k_stage1r_splits
from src.stage1r.multisession_chat import msc_stage1r_splits
from src.stage1r.training import router_qualification_loss, working_memory_loss
from src.stage1r.trace import TRACE_TASKS, trace_stage1r_splits


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    / QWEN_REVISION
)
TARGET_PARAMETERS = 11_380_218
SCHEDULE = {"warmup": 20, "development": 100, "wake": 256, "sleep": 25}
QUALIFICATION_STEPS = 4_000


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


def qualification_data(seed: int) -> tuple[
    dict[str, list[Stage1RExample]], dict[str, list[Stage1RExample]]
]:
    babi, _ = babi_stage1r_splits(
        ROOT / "data/raw/tasks_1-20_v1-2/en-10k", seed=seed
    )
    clutrr, _ = clutrr_stage1r_splits(
        ROOT / "data/raw/clutrr-db9b8f04", seed=seed
    )
    msc, _ = msc_stage1r_splits(
        ROOT / "data/raw/multi_session_chat",
        conversation_limit=100,
        seed=seed,
    )
    prm, _ = prm800k_stage1r_splits(
        sorted((ROOT / "data/raw/prm800k/data").glob("phase*_train.jsonl")),
        seed=seed,
    )
    epbench, _ = epbench_stage1r_splits(
        ROOT / "data/raw/epbench-data", question_limit=500, seed=seed
    )
    train = {
        "babi": babi["train"],
        "clutrr": clutrr["train"],
        "multisession_chat": msc["train"],
        "prm800k": prm["train"],
    }
    dev = {
        "babi": babi["dev"],
        "prm800k": prm["dev"],
        "epbench": epbench["dev"],
    }
    return train, dev


def average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), reverse=True)
    correct = 0
    precisions = []
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            correct += 1
            precisions.append(correct / rank)
    return sum(precisions) / positives


def binary_auroc(scores: list[float], labels: list[int]) -> float:
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return float("nan")
    wins = sum(
        p > n for p in positive for n in negative
    ) + 0.5 * sum(p == n for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def macro_f1(scores: list[float], labels: list[int]) -> float:
    predictions = [int(score >= 0.5) for score in scores]
    f1s = []
    for target in (0, 1):
        tp = sum(p == target and y == target for p, y in zip(predictions, labels))
        fp = sum(p == target and y != target for p, y in zip(predictions, labels))
        fn = sum(p != target and y == target for p, y in zip(predictions, labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-12, precision + recall))
    return sum(f1s) / 2


def evaluate_stage1(
    runner: "Runner",
    dev: dict[str, list[Stage1RExample]],
    output: Path,
) -> tuple[dict[str, float | dict], dict]:
    runner.model.eval()
    support_scores: list[float] = []
    support_labels: list[int] = []
    support_by_task: dict[str, tuple[list[float], list[int]]] = {}
    verifier_scores: list[float] = []
    verifier_labels: list[int] = []
    router_by_family: dict[str, list[list[float]]] = {}
    workspace_sources: set[int] = set()
    write_probabilities: list[float] = []
    answer_correct = 0
    answer_total = 0
    predictions_path = output / "eval_predictions.jsonl"
    events_path = output / "mechanism_events.jsonl"
    with torch.no_grad(), predictions_path.open("w", encoding="utf-8") as predictions, \
            events_path.open("w", encoding="utf-8") as events:
        for family in ("babi", "prm800k"):
            for example in dev[family]:
                _, logits, target, event = runner.forward(example, train=False)
                final = runner.last_output.final
                correct = int(logits.argmax(-1)[0]) == int(target[0])
                answer_correct += correct
                answer_total += 1
                predictions.write(
                    json.dumps(
                        {
                            "example_id": example.example_id,
                            "family": family,
                            "correct": correct,
                            "predicted_first_token_id": int(logits.argmax(-1)[0]),
                            "target_first_token_id": int(target[0]),
                        }
                    )
                    + "\n"
                )
                probabilities = (
                    final.routing.probabilities[0].float().mean(0).tolist()
                )
                router_by_family.setdefault(family, []).append(probabilities)
                workspace_sources.update(final.workspace.sources[0].tolist())
                write_probability = final.working_operation_logits.float().softmax(-1)[
                    0, 1:3
                ].sum()
                write_probabilities.append(float(write_probability))
                if example.verifier_label is not None:
                    verifier_scores.append(float(final.verifier_logits[0].float().sigmoid()))
                    verifier_labels.append(int(example.verifier_label))
                if example.support_spans:
                    encoded = runner.tokenizer(
                        example.input_text,
                        return_offsets_mapping=True,
                        truncation=True,
                        max_length=512,
                    )
                    logits_for_tokens = runner.last_output.cycles[0].support_logits[
                        0, : len(encoded["offset_mapping"])
                    ].float().sigmoid().tolist()
                    task_scores, task_labels = support_by_task.setdefault(
                        example.task_context, ([], [])
                    )
                    for score, (start, end) in zip(
                        logits_for_tokens, encoded["offset_mapping"]
                    ):
                        if end <= start:
                            continue
                        label = int(
                            any(
                                start < support_end and end > support_start
                                for support_start, support_end in example.support_spans
                            )
                        )
                        support_scores.append(score)
                        support_labels.append(label)
                        task_scores.append(score)
                        task_labels.append(label)
                events.write(json.dumps({"example_id": example.example_id, **event}) + "\n")

    memory = EpisodicMemory(capacity=8192)
    state = runner.model.initialize_state(1, device="cuda", dtype=torch.bfloat16)
    retrieval_hits = 0
    retrieval_targets = 0
    episodic_correct = 0
    no_memory_correct = 0
    query_count = 0
    runner.model.set_wake_mode()
    with torch.no_grad():
        for example in sorted(dev["epbench"], key=lambda item: item.timestamp):
            tokens, target = runner._tokens(example)
            result = runner.model(
                tokens.input_ids,
                tokens.attention_mask,
                state,
                memory,
                session_ids=[example.session_id],
                task_contexts=[example.task_context],
                cycles=3,
            )
            if example.encode_target:
                state = runner.model.apply_wake_feedback(
                    result.final,
                    outcome=torch.ones(1, device="cuda"),
                    episodic_memory=memory,
                    session_ids=[example.session_id],
                    task_contexts=[example.task_context],
                    timestamps=[example.timestamp],
                    provenances=[example.example_id],
                    encode_targets=torch.ones(1, device="cuda", dtype=torch.bool),
                    bootstrap_mode=True,
                )
                continue
            query_count += 1
            retrieved_ids = {
                event.provenance for event in result.final.retrieval.events[0]
            }
            target_ids = set(example.retrieval_target_ids)
            if target_ids:
                retrieval_hits += len(retrieved_ids & target_ids)
                retrieval_targets += len(target_ids)
            episodic_correct += (
                int(result.final.token_logits.argmax(-1)[0]) == int(target[0])
            )
            no_memory = runner.model(
                tokens.input_ids,
                tokens.attention_mask,
                runner.model.initialize_state(
                    1, device="cuda", dtype=torch.bfloat16
                ),
                EpisodicMemory(),
                session_ids=[example.session_id],
                task_contexts=[example.task_context],
                cycles=3,
                lesions=LesionConfig(episodic=False),
            )
            no_memory_correct += (
                int(no_memory.final.token_logits.argmax(-1)[0]) == int(target[0])
            )

    router_summary = {}
    all_router_rows = []
    for family, rows in router_by_family.items():
        all_router_rows.extend(rows)
        router_summary[family] = [
            sum(row[index] for row in rows) / len(rows) for index in range(4)
        ]
    router_mass = [
        sum(row[index] for row in all_router_rows) / len(all_router_rows)
        for index in range(4)
    ]
    per_task_support = {
        task: average_precision(scores, labels)
        for task, (scores, labels) in support_by_task.items()
    }
    metrics = {
        "support_selection_auprc": average_precision(
            support_scores, support_labels
        ),
        "support_selection_per_task_auprc": per_task_support,
        "router_expert_mass": router_mass,
        "router_by_family": router_summary,
        "episodic_recall_at_4": retrieval_hits / max(1, retrieval_targets),
        "episodic_answer_accuracy": episodic_correct / max(1, query_count),
        "no_memory_answer_accuracy": no_memory_correct / max(1, query_count),
        "episodic_answer_gain_points": 100
        * (episodic_correct - no_memory_correct)
        / max(1, query_count),
        "verifier_auroc": binary_auroc(verifier_scores, verifier_labels),
        "verifier_macro_f1": macro_f1(verifier_scores, verifier_labels),
        "working_memory_write_fraction": sum(
            probability >= 0.5 for probability in write_probabilities
        )
        / max(1, len(write_probabilities)),
        "workspace_sources_used": len(workspace_sources),
        "first_token_accuracy": answer_correct / max(1, answer_total),
        "episodic_events": len(memory.events),
    }
    acceptance = json.loads(
        (ROOT / "configs/stage1r.json").read_text(encoding="utf-8")
    )["acceptance"]
    gates = {
        "support_selection": (
            metrics["support_selection_auprc"]
            >= acceptance["support_selection"]["pooled_auprc_min"]
            and min(per_task_support.values())
            >= acceptance["support_selection"]["per_task_auprc_min"]
        ),
        "router_balance": (
            min(router_mass) >= acceptance["router"]["expert_mass_min"]
            and max(router_mass) <= acceptance["router"]["expert_mass_max"]
            and all(
                max(masses) <= acceptance["router"]["per_family_mass_max"]
                for masses in router_summary.values()
            )
        ),
        "episodic_retrieval": (
            metrics["episodic_recall_at_4"]
            >= acceptance["episodic"]["recall_at_4_min"]
            and metrics["episodic_answer_gain_points"]
            >= acceptance["episodic"]["answer_gain_points_min"]
        ),
        "verifier": (
            metrics["verifier_auroc"] >= acceptance["verifier"]["auroc_min"]
            and metrics["verifier_macro_f1"]
            >= acceptance["verifier"]["macro_f1_min"]
        ),
        "working_memory_noncollapsed": 0.05
        < metrics["working_memory_write_fraction"]
        < 0.95,
        "workspace_utilized": metrics["workspace_sources_used"] >= 3,
    }
    return metrics, gates


def run_qualification(args: argparse.Namespace) -> None:
    if args.system != "R5":
        raise ValueError("the preregistered qualification wave starts with R5")
    output = args.output or ROOT / "outputs/qualification" / f"R5-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    print("loading frozen Stage 1 qualification data", flush=True)
    train, dev = qualification_data(args.seed)
    runner = Runner(args.system, args.seed, output)
    adapter_parameters = []
    controller_parameters = []
    for name, parameter in runner.model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("adapters."):
            adapter_parameters.append(parameter)
        else:
            controller_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": 5e-5},
            {"params": controller_parameters, "lr": 2e-4},
        ],
        weight_decay=0.01,
    )

    def schedule_fraction(step: int) -> float:
        if step < 200:
            return (step + 1) / 200
        progress = (step - 200) / max(1, QUALIFICATION_STEPS - 200)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fraction)
    checkpoint_path = output / "checkpoint_last.pt"
    start_step = 0
    if checkpoint_path.exists() and args.resume:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        runner.model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        start_step = checkpoint["step"]

    config = {
        "system": args.system,
        "seed": args.seed,
        "phase": "stage1_mechanism_bootstrap",
        "optimizer_steps": QUALIFICATION_STEPS,
        "microbatch": 2,
        "gradient_accumulation": 8,
        "effective_batch": 16,
        "maximum_sequence_length": 512,
        "maximum_cycles": 3,
        "sampling": {
            "babi": 0.30,
            "clutrr": 0.20,
            "multisession_chat": 0.20,
            "prm800k": 0.30,
        },
    }
    _save_json(output / "config.json", config)
    manifest_paths = sorted((ROOT / "data/manifests").glob("*stage1r*.json"))
    _save_json(
        output / "manifest.json",
        {
            "dataset_manifest_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in manifest_paths
            },
            "counts": {
                "train": {family: len(rows) for family, rows in train.items()},
                "dev": {family: len(rows) for family, rows in dev.items()},
            },
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
    backbone_before = hash_module_parameters(runner.model.backbone)
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process(os.getpid())
    step_times = []
    family_names = list(config["sampling"])
    family_weights = [config["sampling"][name] for name in family_names]
    metrics_mode = "a" if start_step else "w"
    with (output / "train_metrics.jsonl").open(
        metrics_mode, encoding="utf-8"
    ) as metrics_stream:
        for step in range(start_step, QUALIFICATION_STEPS):
            started = time.perf_counter()
            runner.model.train()
            optimizer.zero_grad(set_to_none=True)
            component_sums: dict[str, float] = {}
            for _ in range(8):
                family = random.choices(
                    family_names, weights=family_weights, k=1
                )[0]
                rows = train[family]
                examples = [rows[random.randrange(len(rows))] for _ in range(2)]
                loss, components = runner.qualification_batch_loss(
                    examples, step=step
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite Stage 1 loss at step {step}"
                    )
                (loss / 8).backward()
                for name, value in components.items():
                    component_sums[name] = component_sums.get(name, 0.0) + value / 8
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in runner.model.parameters()
                 if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - started
            step_times.append(step_seconds)
            metrics_stream.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "seconds": step_seconds,
                        "learning_rates": [
                            group["lr"] for group in optimizer.param_groups
                        ],
                        **component_sums,
                    }
                )
                + "\n"
            )
            metrics_stream.flush()
            if (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "mean_recent_seconds": statistics.mean(
                                step_times[-100:]
                            ),
                        }
                    ),
                    flush=True,
                )
            if (step + 1) % 500 == 0 or step + 1 == QUALIFICATION_STEPS:
                torch.save(
                    {
                        "model": {
                            name: value.detach().cpu()
                            for name, value in runner.model.state_dict().items()
                            if not name.startswith("backbone.")
                        },
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step + 1,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(),
                        "python_rng": random.getstate(),
                    },
                    checkpoint_path,
                )
    shutil.copyfile(checkpoint_path, output / "checkpoint_best.pt")
    print("running full Stage 1 development gates", flush=True)
    metrics, gates = evaluate_stage1(runner, dev, output)
    backbone_after = hash_module_parameters(runner.model.backbone)
    final = {
        **metrics,
        "system": args.system,
        "seed": args.seed,
        "stage1_foundational_pass": all(gates.values()),
        "backbone_hash_unchanged": backbone_before == backbone_after,
    }
    _save_json(output / "final_metrics.json", final)
    _save_json(
        output / "gate_report.json",
        {
            "gates": gates,
            "all_foundational_gates_passed": all(gates.values()),
            "stop_after_stage1": not all(gates.values()),
        },
    )
    _save_json(
        output / "routing_summary.json",
        {
            "expert_mass": metrics["router_expert_mass"],
            "by_family": metrics["router_by_family"],
        },
    )
    _save_json(
        output / "memory_summary.json",
        {
            "recall_at_4": metrics["episodic_recall_at_4"],
            "events": metrics["episodic_events"],
            "answer_gain_points": metrics["episodic_answer_gain_points"],
        },
    )
    _save_json(output / "fast_weight_summary.json", {"evaluated": False})
    _save_json(output / "sleep_summary.json", {"evaluated": False})
    _save_json(
        output / "resource_usage.json",
        {
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "cpu_rss_gb": process.memory_info().rss / 2**30,
            "mean_optimizer_step_seconds": statistics.mean(step_times),
            "p95_optimizer_step_seconds": _percentile(step_times, 0.95),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
        },
    )
    print(json.dumps({"metrics": metrics, "gates": gates}, indent=2), flush=True)


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
        self.neurostack = system in ("R3", "R3+aux", "R4", "R5")
        self.lesions = LesionConfig(fast_weights=system != "R4")
        if self.neurostack:
            self.model = Stage1RNeuroStack.from_qwen(
                MODEL_PATH,
                differentiated_modulators=system in ("R4", "R5"),
            )
            self.model.set_development_mode()
        elif system == "R0":
            self.model = R0ParameterMatchedAdapter.from_qwen(
                MODEL_PATH, target_trainable_parameters=TARGET_PARAMETERS
            )
        elif system == "R2":
            self.model = R2RecurrentMemoryTokens.from_qwen(
                MODEL_PATH, target_trainable_parameters=TARGET_PARAMETERS
            )
        elif system == "R1":
            self.model = R1OrdinaryRAG.from_qwen(
                MODEL_PATH, target_trainable_parameters=TARGET_PARAMETERS
            )
        else:
            raise ValueError(f"unknown Stage 1R system {system}")
        self.model.train()
        self.last_output = None
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
        if self.neurostack:
            state = self.model.initialize_state(1, device="cuda", dtype=torch.bfloat16)
            output = self.model(
                tokens.input_ids,
                tokens.attention_mask,
                state,
                EpisodicMemory(),
                session_ids=[example.session_id],
                task_contexts=[example.task_context],
                cycles=3,
                lesions=self.lesions,
            )
            self.last_output = output
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
        elif self.system == "R1":
            output = self.model(
                tokens.input_ids,
                tokens.attention_mask,
                session_ids=[example.session_id],
                passes=3,
            )
            logits = output.token_logits
            loss = F.cross_entropy(logits.float(), target)
            event = {"retrieval_indices": output.retrieval_indices[0]}
        else:
            output = self.model(
                tokens.input_ids, tokens.attention_mask, passes=3
            )
            logits = output.token_logits
            loss = F.cross_entropy(logits.float(), target)
            event = {}
        return loss, logits, target, event

    def qualification_batch_loss(
        self, examples: list[Stage1RExample], *, step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        encoded = self.tokenizer(
            [example.input_text for example in examples],
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512,
        )
        offsets = encoded.pop("offset_mapping")
        tokens = encoded.to("cuda")
        targets = torch.tensor(
            [
                self.tokenizer(
                    example.target_text or "none", add_special_tokens=False
                ).input_ids[0]
                for example in examples
            ],
            device="cuda",
        )
        state = self.model.initialize_state(
            len(examples), device="cuda", dtype=torch.bfloat16
        )
        memory = EpisodicMemory()
        retrieval_targets: dict[int, list[bool]] = {}
        with torch.no_grad():
            for row, example in enumerate(examples):
                if not example.support_spans:
                    continue
                context = example.input_text.split("\nQuestion:", 1)[0]
                offset = 0
                retrieval_targets[row] = []
                for fact_index, fact in enumerate(context.splitlines()):
                    start, end = offset, offset + len(fact)
                    is_support = any(
                        start < support_end and end > support_start
                        for support_start, support_end in example.support_spans
                    )
                    fact_tokens = self.tokenizer(
                        fact,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                    ).to("cuda")
                    embedded = self.model.backbone.model.embed_tokens(
                        fact_tokens.input_ids
                    ).mean(1)
                    value = self.model.token_projection(embedded)[0]
                    key = self.model.episodic_key(value)
                    memory.write(
                        EpisodicEvent(
                            key=key,
                            value=value,
                            timestamp=fact_index,
                            session_id=example.session_id,
                            task_context=example.task_context,
                            goal_state=value.new_zeros(256),
                            workspace_summary=value.new_zeros(256),
                            outcome=1.0 if is_support else 0.0,
                            confidence=1.0,
                            provenance=f"train:{row}:{fact_index}:{int(is_support)}",
                        )
                    )
                    retrieval_targets[row].append(is_support)
                    offset = end + 1
        output = self.model(
            tokens.input_ids,
            tokens.attention_mask,
            state,
            memory,
            session_ids=[example.session_id for example in examples],
            task_contexts=[example.task_context for example in examples],
            cycles=3,
            lesions=self.lesions,
        )
        self.last_output = output
        final = output.final
        components: dict[str, torch.Tensor] = {
            "answer": F.cross_entropy(final.token_logits.float(), targets)
        }

        support_logits = output.cycles[0].support_logits.float()
        support_values = []
        support_targets = []
        for row, example in enumerate(examples):
            if not example.support_spans:
                continue
            for column, (start, end) in enumerate(offsets[row].tolist()):
                if end <= start:
                    continue
                support_values.append(support_logits[row, column])
                support_targets.append(
                    any(
                        start < support_end and end > support_start
                        for support_start, support_end in example.support_spans
                    )
                )
        if support_values:
            components["support"] = F.binary_cross_entropy_with_logits(
                torch.stack(support_values),
                torch.tensor(
                    support_targets, device="cuda", dtype=torch.float32
                ),
            )

        retrieval_losses = []
        for row, target_flags in retrieval_targets.items():
            row_events = [
                event
                for event in memory.events
                if event.session_id == examples[row].session_id
                and event.task_context == examples[row].task_context
            ]
            if not row_events or not any(target_flags):
                continue
            keys = torch.stack([event.key for event in row_events]).to(
                final.hidden_summary
            )
            query = F.normalize(
                self.model.episodic_key(final.hidden_summary[row]), dim=-1
            )
            scores = keys @ query
            positive = torch.tensor(
                target_flags, device="cuda", dtype=torch.bool
            )
            retrieval_losses.append(
                torch.logsumexp(scores, dim=0)
                - torch.logsumexp(scores[positive], dim=0)
            )
        if retrieval_losses:
            components["retrieval"] = torch.stack(retrieval_losses).mean()

        operation_targets = torch.tensor(
            [1 if example.encode_target else 0 for example in examples],
            device="cuda",
        )
        slot_targets = torch.zeros(
            len(examples), device="cuda", dtype=torch.long
        )
        components["working_memory"], _ = working_memory_loss(
            final.working_operation_logits.float(),
            final.working_slot_logits.float(),
            operation_targets,
            slot_targets,
            occupied=final.state.working_memory.occupied,
            protection=final.state.working_memory.protection.float(),
        )
        verifier_rows = [
            row
            for row, example in enumerate(examples)
            if example.verifier_label is not None
        ]
        if verifier_rows:
            components["verify"] = F.binary_cross_entropy_with_logits(
                final.verifier_logits[verifier_rows].float(),
                torch.tensor(
                    [examples[row].verifier_label for row in verifier_rows],
                    device="cuda",
                    dtype=torch.float32,
                ),
            )
        components["router"], _ = router_qualification_loss(
            final.routing,
            [example.family for example in examples],
            step=step,
            anneal_steps=2_000,
        )
        components["compute"] = (
            final.action_logits.float().softmax(-1)[:, 0].mean()
        )
        modulator_losses = []
        for row, example in enumerate(examples):
            if example.verifier_label is not None:
                modulator_losses.append(
                    F.mse_loss(
                        final.modulators.da[row].float(),
                        final.modulators.da[row].new_tensor(
                            2 * example.verifier_label - 1
                        ).float(),
                    )
                )
            if example.encode_target is not None:
                modulator_losses.append(
                    F.mse_loss(
                        final.modulators.ach[row].float(),
                        final.modulators.ach[row].new_tensor(
                            float(example.encode_target)
                        ).float(),
                    )
                )
        if modulator_losses:
            components["modulators"] = torch.stack(modulator_losses).mean()
        zero = components["answer"].new_zeros(())
        total = (
            components["answer"]
            + 0.5 * components.get("support", zero)
            + 0.5 * components["working_memory"]
            + 0.5 * components.get("retrieval", zero)
            + 0.3 * components.get("verify", zero)
            + 0.2 * components.get("modulators", zero)
            + components["router"]
            + 0.01 * components["compute"]
        )
        return total, {
            name: float(value.detach()) for name, value in components.items()
        }

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
    parser.add_argument(
        "--system",
        choices=("R0", "R1", "R2", "R3", "R3+aux", "R4", "R5"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode", choices=("calibration", "qualification"), default="calibration"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "qualification":
        run_qualification(args)
        return
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
    if runner.neurostack:
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
