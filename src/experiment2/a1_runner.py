from __future__ import annotations

from dataclasses import replace
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
from .tokenization import (
    collate_tokenized,
    fit_example_to_token_budget,
    tokenize_example,
)


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
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.dev_limit = dev_limit
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.splits, raw_files = experiment2_babi_splits(data_dir, seed=seed)
        self.trimmed_counts = {}
        for split, examples in self.splits.items():
            fitted = [
                fit_example_to_token_budget(item, self.tokenizer) for item in examples
            ]
            self.trimmed_counts[split] = sum(
                before.input_text != after.input_text
                for before, after in zip(examples, fitted)
            )
            self.splits[split] = fitted
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
        self.bad_evaluations = 0
        self.stopped_early = False
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
                "support_preserving_trimmed_examples": self.trimmed_counts,
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
            "bad_evaluations": self.bad_evaluations,
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
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        self.step = int(state["step"])
        self.best_score = float(state["best_score"])
        self.bad_evaluations = int(state.get("bad_evaluations", 0))
        if _backbone_hash(self.model.dense_backbone) != state["backbone_hash"]:
            raise RuntimeError("frozen backbone changed across resume")

    def _sample(self) -> tuple[list, dict[str, torch.Tensor]]:
        items = [self.generator.choice(self.tokenized_train) for _ in range(2)]
        batch = collate_tokenized(items, pad_token_id=self.tokenizer.pad_token_id)
        return items, {key: value.to(self.device) for key, value in batch.items()}

    def train(self, *, stop_after_step: int | None = None) -> None:
        self.model.train()
        target_step = min(self.max_steps, stop_after_step or self.max_steps)
        while self.step < target_step:
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
                metrics = self.evaluate_final(limit=self.dev_limit, log=False)
                score = (
                    max(metrics["fact_auprc"], 1e-9)
                    * max(metrics["support_recall"], 1e-9)
                    * max(metrics["answer_exact_match"], 1e-9)
                ) ** (1 / 3)
                is_best = score > self.best_score
                if is_best:
                    self.best_score = score
                    self.bad_evaluations = 0
                else:
                    self.bad_evaluations += 1
                self.save(best=is_best)
                _append(
                    self.output_dir / "mechanism_events.jsonl",
                    {"step": self.step, "dev": metrics, "checkpoint_score": score},
                )
                if self.step >= 2000 and self.bad_evaluations >= 4:
                    self.stopped_early = True
                    break
        if self.step == self.max_steps or self.stopped_early:
            self.finalize()
        else:
            self.save()

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

    @torch.inference_mode()
    def _generate(
        self,
        prompt_ids: list[list[int]],
        prefix: torch.Tensor,
        *,
        max_new_tokens: int = 2,
    ) -> list[list[int]]:
        sequences = [list(values) for values in prompt_ids]
        generated = [[] for _ in sequences]
        finished = [False] * len(sequences)
        eos = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            length = max(map(len, sequences))
            ids = torch.full(
                (len(sequences), length),
                self.tokenizer.pad_token_id,
                device=self.device,
                dtype=torch.long,
            )
            mask = torch.zeros_like(ids)
            lengths = []
            for row, values in enumerate(sequences):
                ids[row, : len(values)] = torch.tensor(values, device=self.device)
                mask[row, : len(values)] = 1
                lengths.append(len(values))
            output = self.model._cycle(ids, mask, prefix)
            next_tokens = output.logits[
                torch.arange(len(sequences), device=self.device),
                prefix.shape[1] + torch.tensor(lengths, device=self.device) - 1,
            ].argmax(-1)
            for row, token in enumerate(next_tokens.tolist()):
                if finished[row]:
                    continue
                if token == eos:
                    finished[row] = True
                else:
                    generated[row].append(token)
                    sequences[row].append(token)
            if all(finished):
                break
        return generated

    @torch.inference_mode()
    def evaluate_final(
        self, *, limit: int | None = None, log: bool = False
    ) -> dict:
        self.model.eval()
        scores, labels, recalls = [], [], []
        answers, lesioned_answers = [], []
        write_predictions, write_targets = [], []
        per_task: dict[str, tuple[list[float], list[int]]] = {}
        examples = self.splits["dev"][:limit]
        for start in range(0, len(examples), 16):
            source = examples[start : start + 16]
            blank = [replace(item, target_text="") for item in source]
            tokenized = [tokenize_example(item, self.tokenizer) for item in blank]
            batch = collate_tokenized(tokenized, pad_token_id=self.tokenizer.pad_token_id)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            first = self.model.dense_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            facts, questions, fact_mask, support_mask = self.model._pool(
                first.hidden_states[-1], tokenized
            )
            fact_values = self.model.fact_projection(facts)
            question_values = self.model.question_projection(questions)
            fact_logits = self.model.fact_scorer(fact_values, question_values).masked_fill(
                ~fact_mask, -1e4
            )
            memory, operation_logits, _ = self.model.working_memory.write(
                fact_values, support_mask
            )
            prefix = self.model.memory_prefix(memory.values) * memory.occupied.unsqueeze(-1)
            prompt_ids = [item.input_ids for item in tokenized]
            generated = self._generate(prompt_ids, prefix)
            generated_lesioned = self._generate(prompt_ids, torch.zeros_like(prefix))
            for row, (item, predicted, predicted_lesioned) in enumerate(
                zip(tokenized, generated, generated_lesioned)
            ):
                count = len(item.fact_token_spans)
                row_scores = fact_logits[row, :count].float().cpu().tolist()
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
                generated_text = _normalize(self.tokenizer.decode(predicted))
                generated_lesioned_text = _normalize(
                    self.tokenizer.decode(predicted_lesioned)
                )
                answer = generated_text.split()[0] if generated_text else ""
                lesioned_answer = (
                    generated_lesioned_text.split()[0]
                    if generated_lesioned_text
                    else ""
                )
                target = _normalize(source[row].target_text)
                answers.append(int(answer == target))
                lesioned_answers.append(int(lesioned_answer == target))
                operations = operation_logits[row, :count].argmax(-1).tolist()
                write_predictions.extend(int(value == 1) for value in operations)
                write_targets.extend(row_labels)
                if log:
                    _append(
                        self.output_dir / "eval_predictions.jsonl",
                        {
                            "example_id": item.example.example_id,
                            "target": target,
                            "prediction": answer,
                            "generated_text": generated_text,
                            "lesioned_prediction": lesioned_answer,
                            "lesioned_generated_text": generated_lesioned_text,
                            "correct": answer == target,
                            "lesioned_correct": lesioned_answer == target,
                        },
                    )
                    _append(
                        self.output_dir / "memory_writes.jsonl",
                        {
                            "example_id": item.example.example_id,
                            "predicted_replace": [
                                index for index, value in enumerate(operations) if value == 1
                            ],
                            "support_fact_indices": item.example.support_fact_indices,
                            "occupied_slots": memory.occupied[row].nonzero(
                                as_tuple=False
                            ).flatten().tolist(),
                        },
                    )
        write_fraction = sum(write_predictions) / max(1, len(write_predictions))
        false_writes = sum(
            predicted and not target
            for predicted, target in zip(write_predictions, write_targets)
        )
        negative_count = sum(not value for value in write_targets)
        result = {
            "fact_auprc": _average_precision(scores, labels),
            "per_task_fact_auprc": {
                task: _average_precision(*values) for task, values in per_task.items()
            },
            "support_recall": sum(recalls) / max(1, len(recalls)),
            "answer_exact_match": sum(answers) / max(1, len(answers)),
            "lesioned_answer_exact_match": sum(lesioned_answers)
            / max(1, len(lesioned_answers)),
            "working_memory_lesion_drop_points": 100
            * (sum(answers) - sum(lesioned_answers))
            / max(1, len(answers)),
            "predicted_write_fraction": write_fraction,
            "false_write_rate": false_writes / max(1, negative_count),
        }
        self.model.train()
        return result

    def finalize(self) -> None:
        selected_step = self.step
        best_path = self.output_dir / "checkpoint_best.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location=self.device, weights_only=False)
            missing, unexpected = self.model.load_state_dict(best["model"], strict=False)
            if unexpected or any(
                not name.startswith("dense_backbone.backbone.") for name in missing
            ):
                raise RuntimeError("best checkpoint does not match A1 model")
            selected_step = int(best["step"])
        metrics = self.evaluate_final(limit=self.dev_limit, log=True)
        metrics.update(
            {
                "seed": self.seed,
                "optimizer_steps": self.step,
                "selected_checkpoint_step": selected_step,
                "stopped_early": self.stopped_early,
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
            "working_memory_utility": (
                metrics["working_memory_lesion_drop_points"] >= 3.0
            ),
            "noncollapsed_writes": (
                0.0 < metrics["predicted_write_fraction"] < 1.0
            ),
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
