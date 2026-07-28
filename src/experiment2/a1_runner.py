from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from .a1 import A1SupportWorkingMemoryModel
from .babi import experiment2_babi_splits
from .data import Experiment2Example
from .model import DenseFrozenBackbone, QWEN_REVISION
from .tokenization import collate_tokenized, tokenize_example


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _backbone_hash(model: DenseFrozenBackbone) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.backbone.state_dict().items():
        digest.update(name.encode())
        digest.update(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), reverse=True)
    hits = 0
    precision = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            precision += hits / rank
    return precision / positives


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


class A1Runner:
    def __init__(
        self,
        *,
        model_path: Path,
        data_dir: Path,
        output_dir: Path,
        seed: int,
        max_steps: int = 4000,
        device: str = "cuda",
        dev_limit: int | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        self.seed = seed
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.dev_limit = dev_limit
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.splits, raw_files = experiment2_babi_splits(data_dir, seed=seed)
        self.tokenized_train = [
            tokenize_example(item, self.tokenizer) for item in self.splits["train"]
        ]
        dense = DenseFrozenBackbone.from_qwen(model_path, device=device)
        self.model = A1SupportWorkingMemoryModel(dense)
        for name, module in self.model.named_children():
            if name != "dense_backbone":
                module.to(device=self.device, dtype=torch.bfloat16)
        self.initial_backbone_hash = _backbone_hash(dense)
        adapters = [
            parameter
            for bank in dense.adapters
            for parameter in bank.branches["relational"].parameters()
            if parameter.requires_grad
        ]
        adapter_ids = {id(parameter) for parameter in adapters}
        heads = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in adapter_ids
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": adapters, "lr": 5e-5},
                {"params": heads, "lr": 2e-4},
            ],
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: (
                min(1.0, (step + 1) / 200)
                if step < 200
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi * (step - 200) / max(1, self.max_steps - 200)
                    )
                )
            ),
        )
        self.generator = random.Random(seed)
        self.step = 0
        self.best_score = -math.inf
        self._write_contract(raw_files)

    def _write_contract(self, raw_files: list[Path]) -> None:
        config = {
            "phase": "A1",
            "seed": self.seed,
            "max_steps": self.max_steps,
            "microbatch": 2,
            "gradient_accumulation": 8,
            "effective_batch": 16,
            "cycles": 2,
            "adapter_lr": 5e-5,
            "head_lr": 2e-4,
            "warmup": 200,
            "scheduler": "cosine",
            "backbone_revision": QWEN_REVISION,
            "dev_limit": self.dev_limit,
        }
        _json(self.output_dir / "config.json", config)
        _json(
            self.output_dir / "manifest.json",
            {
                "adapter_version": "experiment2-v1",
                "counts": {key: len(value) for key, value in self.splits.items()},
                "raw_sha256": {path.name: _sha256(path) for path in raw_files},
                "selected_ids_sha256": {
                    key: hashlib.sha256(
                        "\n".join(item.example_id for item in value).encode()
                    ).hexdigest()
                    for key, value in self.splits.items()
                },
            },
        )
        _json(
            self.output_dir / "environment.json",
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
        )
        for name in (
            "eval_predictions.jsonl",
            "mechanism_events.jsonl",
            "workspace_attention.jsonl",
            "retrieval_candidates.jsonl",
            "memory_writes.jsonl",
            "fast_weight_updates.jsonl",
            "sleep_metrics.jsonl",
        ):
            (self.output_dir / name).touch(exist_ok=True)

    def save(self, *, best: bool = False) -> None:
        state = {
            "step": self.step,
            "model": {
                name: value
                for name, value in self.model.state_dict().items()
                if not name.startswith("dense_backbone.backbone.")
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "python_rng": self.generator.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "best_score": self.best_score,
            "backbone_hash": self.initial_backbone_hash,
        }
        torch.save(state, self.output_dir / "checkpoint_last.pt")
        if best:
            torch.save(state, self.output_dir / "checkpoint_best.pt")

    def resume(self) -> None:
        state = torch.load(
            self.output_dir / "checkpoint_last.pt", map_location=self.device, weights_only=False
        )
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        if unexpected or any(
            not name.startswith("dense_backbone.backbone.") for name in missing
        ):
            raise RuntimeError("checkpoint does not match the A1 trainable state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.generator.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.step = int(state["step"])
        self.best_score = float(state["best_score"])
        if _backbone_hash(self.model.dense_backbone) != state["backbone_hash"]:
            raise RuntimeError("frozen backbone changed across resume")

    def _sample(self) -> tuple[list, dict[str, torch.Tensor]]:
        items = [self.generator.choice(self.tokenized_train) for _ in range(2)]
        batch = collate_tokenized(items, pad_token_id=self.tokenizer.pad_token_id)
        return items, {key: value.to(self.device) for key, value in batch.items()}

    def train(self) -> None:
        self.model.train()
        while self.step < self.max_steps:
            started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            for _ in range(8):
                examples, batch = self._sample()
                output = self.model(batch, examples)
                (output.loss / 8).backward()
                values = {
                    "loss": output.loss,
                    "sequence_nll": output.sequence_nll.mean(),
                    "lesioned_sequence_nll": output.lesioned_sequence_nll.mean(),
                }
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach()) / 8
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                1.0,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            record = {
                "step": self.step,
                **sums,
                "seconds": time.perf_counter() - started,
                "lr_adapter": self.optimizer.param_groups[0]["lr"],
                "lr_heads": self.optimizer.param_groups[1]["lr"],
            }
            _append(self.output_dir / "train_metrics.jsonl", record)
            if self.step % 250 == 0 or self.step == self.max_steps:
                metrics = self.evaluate_support(limit=self.dev_limit)
                score = (
                    max(metrics["fact_auprc"], 1e-9)
                    * max(metrics["support_recall"], 1e-9)
                    * max(metrics["teacher_forced_exact_match"], 1e-9)
                ) ** (1 / 3)
                is_best = score > self.best_score
                if is_best:
                    self.best_score = score
                self.save(best=is_best)
                _append(
                    self.output_dir / "mechanism_events.jsonl",
                    {"step": self.step, "dev": metrics, "checkpoint_score": score},
                )
        self.finalize()

    @torch.inference_mode()
    def evaluate_support(self, *, limit: int | None = None) -> dict:
        self.model.eval()
        scores, labels, recalls, exact = [], [], [], []
        per_task: dict[str, tuple[list[float], list[int]]] = {}
        examples = self.splits["dev"][:limit]
        for start in range(0, len(examples), 16):
            source = examples[start : start + 16]
            tokenized = [tokenize_example(item, self.tokenizer) for item in source]
            batch = collate_tokenized(tokenized, pad_token_id=self.tokenizer.pad_token_id)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.model(batch, tokenized)
            for row, item in enumerate(tokenized):
                count = len(item.fact_token_spans)
                row_scores = output.fact_logits[row, :count].float().cpu().tolist()
                row_labels = [
                    int(index in item.example.support_fact_indices)
                    for index in range(count)
                ]
                scores.extend(row_scores)
                labels.extend(row_labels)
                task = item.example.example_id.split(":")[2]
                task_scores, task_labels = per_task.setdefault(task, ([], []))
                task_scores.extend(row_scores)
                task_labels.extend(row_labels)
                k = max(1, sum(row_labels))
                top = sorted(range(count), key=row_scores.__getitem__, reverse=True)[:k]
                recalls.append(
                    sum(row_labels[index] for index in top) / max(1, sum(row_labels))
                )
                prefix = 8
                predictions = output.answer_logits[row, prefix:-1].argmax(-1)
                target_ids = batch["input_ids"][row]
                prompt = int(batch["prompt_lengths"][row])
                answer_length = int(batch["attention_mask"][row].sum()) - prompt
                predicted_answer = predictions[prompt - 1 : prompt - 1 + answer_length]
                exact.append(
                    int(torch.equal(predicted_answer, target_ids[prompt : prompt + answer_length]))
                )
        self.model.train()
        return {
            "fact_auprc": _average_precision(scores, labels),
            "per_task_fact_auprc": {
                task: _average_precision(*values) for task, values in per_task.items()
            },
            "support_recall": sum(recalls) / max(1, len(recalls)),
            "teacher_forced_exact_match": sum(exact) / max(1, len(exact)),
        }

    def finalize(self) -> None:
        metrics = self.evaluate_support(limit=self.dev_limit)
        metrics.update(
            {
                "seed": self.seed,
                "backbone_hash_unchanged": (
                    _backbone_hash(self.model.dense_backbone) == self.initial_backbone_hash
                ),
            }
        )
        gates = {
            "pooled_fact_auprc": metrics["fact_auprc"] >= 0.8,
            "per_task_fact_auprc": all(
                value >= 0.7 for value in metrics["per_task_fact_auprc"].values()
            ),
            "support_recall": metrics["support_recall"] >= 0.8,
        }
        gates["phase_passed"] = all(gates.values())
        _json(self.output_dir / "final_metrics.json", metrics)
        _json(self.output_dir / "gate_report.json", gates)
        _json(
            self.output_dir / "resource_usage.json",
            {
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
            },
        )
