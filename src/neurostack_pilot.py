from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import random
import time

import torch
from torch import nn
import torch.nn.functional as F


MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class CognitiveExample:
    example_id: str
    family: str
    input_text: str
    target_text: str
    session_id: str | None
    timestamp: int | None
    context_id: str | None
    supporting_spans: list[tuple[int, int]]
    relation_depth: int | None
    step_labels: list[int] | None
    should_encode: list[int] | None
    should_retrieve: bool | None
    should_verify: bool | None
    answer_available: bool
    reset_working_memory: bool
    reset_episodic_memory: bool
    facts: list[str]
    question: str
    supporting_fact_indices: list[int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_babi(path: Path, task: int) -> list[CognitiveExample]:
    examples: list[CognitiveExample] = []
    facts: list[tuple[int, str]] = []
    story_number = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        number_text, content = raw.split(" ", 1)
        line_number = int(number_text)
        if line_number == 1:
            facts = []
            story_number += 1
        if "\t" not in content:
            facts.append((line_number, content))
            continue
        question, answer, support_text = content.split("\t")
        support_ids = [int(value) for value in support_text.split()]
        id_to_index = {fact_id: index for index, (fact_id, _) in enumerate(facts)}
        support_indices = [id_to_index[value] for value in support_ids if value in id_to_index]
        fact_texts = [text for _, text in facts]
        input_text = "\n".join(fact_texts + [question])
        examples.append(
            CognitiveExample(
                example_id=f"babi-qa{task}-{path.stem}-{story_number}-{line_number}",
                family=f"babi_qa{task}",
                input_text=input_text,
                target_text=answer.strip().lower(),
                session_id=None,
                timestamp=None,
                context_id=f"{path.stem}-{story_number}",
                supporting_spans=[],
                relation_depth=len(support_ids),
                step_labels=None,
                should_encode=[int(index in support_indices) for index in range(len(facts))],
                should_retrieve=False,
                should_verify=False,
                answer_available=True,
                reset_working_memory=True,
                reset_episodic_memory=True,
                facts=fact_texts,
                question=question,
                supporting_fact_indices=support_indices,
            )
        )
    return examples


class FrozenQwenEncoder:
    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        # A globally installed, incompatible flash-attn build must not be selected.
        import transformers

        transformers.utils.is_flash_attn_2_available = lambda: False
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 128) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        unique = list(dict.fromkeys(texts))
        for offset in range(0, len(unique), batch_size):
            batch_texts = unique[offset : offset + batch_size]
            tokens = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=96,
            ).to(self.device)
            result = self.model.model(**tokens)
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (result.last_hidden_state.float() * mask).sum(1) / mask.sum(1)
            for text, embedding in zip(batch_texts, pooled.cpu()):
                output[text] = embedding
        return output


class SpecialistAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(256)
        self.down = nn.Linear(256, 128)
        self.up = nn.Linear(128, 256)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(F.silu(self.down(self.norm(value))))


class NeuroStack(nn.Module):
    """Reduced pilot implementation of the specified integrated controller."""

    def __init__(self, answers: int) -> None:
        super().__init__()
        self.project = nn.Linear(896, 256)
        self.fact_key = nn.Linear(256, 64)
        self.query_key = nn.Linear(256, 64)
        self.pfc = nn.ModuleList([nn.GRUCell(1280, 256) for _ in range(4)])
        self.specialists = nn.ModuleList([SpecialistAdapter() for _ in range(4)])
        self.router = nn.Sequential(nn.Linear(768, 512), nn.SiLU(), nn.Linear(512, 4))
        self.modulators = nn.Sequential(
            nn.Linear(768, 256), nn.SiLU(), nn.Linear(256, 5)
        )
        self.verifier = nn.Sequential(
            nn.Linear(1024, 256), nn.SiLU(), nn.Linear(256, 2)
        )
        self.answer = nn.Linear(1024, answers)

    def forward(
        self, fact_embeddings: torch.Tensor, question_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        facts = self.project(fact_embeddings)
        question = self.project(question_embedding)
        gate_logits = (
            self.fact_key(facts) * self.query_key(question).unsqueeze(0)
        ).sum(-1) / 8.0

        wm_count = min(8, facts.shape[0])
        wm_indices = gate_logits.topk(wm_count).indices
        working = facts[wm_indices]
        workspace_count = min(4, working.shape[0])
        workspace = working[:workspace_count]
        wm_summary = working.mean(0)
        workspace_summary = workspace.mean(0)
        current = facts[-1]

        slots = []
        for index, cell in enumerate(self.pfc):
            previous = torch.zeros(256, device=facts.device, dtype=facts.dtype)
            pfc_input = torch.cat(
                [question, wm_summary, workspace_summary, current, question]
            )
            slots.append(cell(pfc_input, previous))
        pfc_summary = torch.stack(slots).mean(0)

        controller_input = torch.cat(
            [question, pfc_summary, workspace_summary]
        )
        route_logits = self.router(controller_input)
        route_weights = F.softmax(route_logits, dim=-1)
        top_values, top_indices = route_weights.topk(2)
        top_values = top_values / top_values.sum()
        specialist = sum(
            weight * self.specialists[int(index)](pfc_summary)
            for weight, index in zip(top_values, top_indices)
        )

        raw_modulators = self.modulators(controller_input)
        modulator_values = torch.cat(
            [raw_modulators[:1].tanh(), raw_modulators[1:].sigmoid()]
        )
        combined = torch.cat(
            [question, pfc_summary, workspace_summary, specialist]
        )
        logits = self.answer(combined)
        verify_logits = self.verifier(combined)
        return logits, {
            "gate_logits": gate_logits,
            "route_weights": route_weights,
            "modulators": modulator_values,
            "verify_logits": verify_logits,
            "memory_key": F.normalize(question.detach(), dim=0),
        }


class GenericAdapter(nn.Module):
    def __init__(self, answers: int, target_parameters: int) -> None:
        super().__init__()
        self.project = nn.Linear(896, 256)
        fixed = sum(p.numel() for p in self.project.parameters()) + 257 * answers
        hidden = max(1, round((target_parameters - fixed - 256) / 769))
        self.adapter = nn.Sequential(
            nn.Linear(512, hidden), nn.SiLU(), nn.Linear(hidden, 256)
        )
        self.answer = nn.Linear(256, answers)

    def forward(
        self, fact_embeddings: torch.Tensor, question_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        facts = self.project(fact_embeddings)
        question = self.project(question_embedding)
        summary = facts.mean(0)
        value = self.adapter(torch.cat([question, summary]))
        return self.answer(value), {}


class EpisodicMemory:
    def __init__(self, capacity: int = 8192, top_k: int = 4) -> None:
        self.capacity = capacity
        self.top_k = top_k
        self.keys: list[torch.Tensor] = []
        self.answers: list[int] = []

    def write(self, key: torch.Tensor, answer: int) -> None:
        if len(self.keys) == self.capacity:
            self.keys.pop(0)
            self.answers.pop(0)
        self.keys.append(key.cpu())
        self.answers.append(answer)

    def logits(self, key: torch.Tensor, classes: int, device: str) -> torch.Tensor:
        result = torch.zeros(classes, device=device)
        if not self.keys:
            return result
        keys = torch.stack(self.keys)
        similarities = keys @ key.cpu()
        count = min(self.top_k, len(self.keys))
        values, indices = similarities.topk(count)
        weights = F.softmax(values * 10.0, dim=0)
        for weight, index in zip(weights.to(device), indices):
            result[self.answers[int(index)]] += weight
        return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def features_for(
    example: CognitiveExample, embeddings: dict[str, torch.Tensor], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    facts = torch.stack([embeddings[text] for text in example.facts]).to(device)
    question = embeddings[example.question].to(device)
    return facts, question


def train_epoch(
    model: nn.Module,
    examples: list[CognitiveExample],
    embeddings: dict[str, torch.Tensor],
    answer_to_id: dict[str, int],
    optimizer: torch.optim.Optimizer,
    device: str,
    seed: int,
) -> float:
    model.train()
    order = list(examples)
    random.Random(seed).shuffle(order)
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for index, example in enumerate(order):
        facts, question = features_for(example, embeddings, device)
        logits, aux = model(facts, question)
        target = torch.tensor(answer_to_id[example.target_text], device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        if "gate_logits" in aux:
            support = torch.zeros(len(example.facts), device=device)
            support[example.supporting_fact_indices] = 1.0
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                aux["gate_logits"], support
            )
            verifier_target = torch.ones(2, device=device)
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                aux["verify_logits"], verifier_target
            )
            balance = aux["route_weights"]
            loss = loss + 0.01 * (balance * balance.clamp_min(1e-8).log()).sum()
        (loss / 8).backward()
        total += float(loss.detach())
        if (index + 1) % 8 == 0 or index + 1 == len(order):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total / max(1, len(order))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    examples: list[CognitiveExample],
    embeddings: dict[str, torch.Tensor],
    answer_to_id: dict[str, int],
    id_to_answer: list[str],
    device: str,
    memory: EpisodicMemory | None = None,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    correct = 0
    support_scores: list[float] = []
    support_labels: list[int] = []
    routes = torch.zeros(4)
    predictions = []
    for example in examples:
        facts, question = features_for(example, embeddings, device)
        logits, aux = model(facts, question)
        if memory is not None and "memory_key" in aux:
            episodic = memory.logits(
                aux["memory_key"], len(id_to_answer), device
            )
            logits = logits + 2.0 * episodic
        prediction_id = int(logits.argmax())
        target_id = answer_to_id[example.target_text]
        correct += int(prediction_id == target_id)
        if "gate_logits" in aux:
            support_scores.extend(aux["gate_logits"].sigmoid().cpu().tolist())
            support_labels.extend(
                [
                    int(index in example.supporting_fact_indices)
                    for index in range(len(example.facts))
                ]
            )
            routes += aux["route_weights"].cpu()
        predictions.append(
            {
                "example_id": example.example_id,
                "target": example.target_text,
                "prediction": id_to_answer[prediction_id],
                "correct": prediction_id == target_id,
            }
        )
    metrics = {"accuracy": correct / max(1, len(examples))}
    if support_scores:
        from sklearn.metrics import average_precision_score

        metrics["support_auprc"] = float(
            average_precision_score(support_labels, support_scores)
        )
        usage = routes / routes.sum()
        metrics.update(
            {f"route_{index}_share": float(value) for index, value in enumerate(usage)}
        )
    return metrics, predictions


def run(config: dict) -> dict:
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(config["root"])
    data_dir = root / "data" / "raw" / "tasks_1-20_v1-2" / "en-10k"
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    train_by_task: dict[int, list[CognitiveExample]] = {}
    test_by_task: dict[int, list[CognitiveExample]] = {}
    for task in range(1, 6):
        train_path = next(data_dir.glob(f"qa{task}_*_train.txt"))
        test_path = next(data_dir.glob(f"qa{task}_*_test.txt"))
        train_by_task[task] = parse_babi(train_path, task)[: config["train_per_task"]]
        test_by_task[task] = parse_babi(test_path, task)[: config["test_per_task"]]

    all_examples = sum(train_by_task.values(), []) + sum(test_by_task.values(), [])
    answers = sorted({example.target_text for example in all_examples})
    answer_to_id = {answer: index for index, answer in enumerate(answers)}
    texts = [
        text
        for example in all_examples
        for text in [*example.facts, example.question]
    ]

    cache_path = output_dir / "qwen_babi_features.pt"
    if cache_path.exists():
        embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        encoder = FrozenQwenEncoder(Path(config["model_path"]), device)
        embeddings = encoder.encode(texts, config["encoder_batch_size"])
        torch.save(embeddings, cache_path)
        del encoder
        torch.cuda.empty_cache()

    full = NeuroStack(len(answers)).to(device)
    full_parameters = parameter_count(full)
    generic = GenericAdapter(len(answers), full_parameters).to(device)
    generic_parameters = parameter_count(generic)
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=2e-4, weight_decay=0.01)
    generic_optimizer = torch.optim.AdamW(
        generic.parameters(), lr=2e-4, weight_decay=0.01
    )

    developmental = sum(
        [examples[: config["bootstrap_per_task"]] for examples in train_by_task.values()],
        [],
    )
    for epoch in range(config["bootstrap_epochs"]):
        train_epoch(
            full, developmental, embeddings, answer_to_id, full_optimizer, device, seed + epoch
        )
        train_epoch(
            generic,
            developmental,
            embeddings,
            answer_to_id,
            generic_optimizer,
            device,
            seed + epoch,
        )

    results: dict = {
        "protocol": "bounded Experiment 1 pilot; not the confirmatory experiment",
        "seed": seed,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "frozen": True},
        "trainable_parameters": {
            "neurostack": full_parameters,
            "generic": generic_parameters,
            "difference_fraction": abs(full_parameters - generic_parameters)
            / full_parameters,
        },
        "tasks": {},
    }
    all_predictions: list[dict] = []
    old_replay: list[CognitiveExample] = []
    memory = EpisodicMemory()
    start = time.perf_counter()
    for task in range(1, 6):
        test = test_by_task[task]
        pre_full, pre_predictions = evaluate(
            full, test, embeddings, answer_to_id, answers, device
        )
        pre_generic, _ = evaluate(
            generic, test, embeddings, answer_to_id, answers, device
        )

        wake = train_by_task[task][config["bootstrap_per_task"] :]
        full.eval()
        with torch.inference_mode():
            for example in wake:
                facts, question = features_for(example, embeddings, device)
                _, aux = full(facts, question)
                memory.write(aux["memory_key"], answer_to_id[example.target_text])
        memory_metrics, memory_predictions = evaluate(
            full, test, embeddings, answer_to_id, answers, device, memory
        )

        replay = wake + old_replay[-config["old_replay_size"] :]
        for epoch in range(config["sleep_epochs"]):
            train_epoch(
                full,
                replay,
                embeddings,
                answer_to_id,
                full_optimizer,
                device,
                seed + task * 100 + epoch,
            )
            train_epoch(
                generic,
                replay,
                embeddings,
                answer_to_id,
                generic_optimizer,
                device,
                seed + task * 100 + epoch,
            )
        old_replay.extend(wake)

        post_full, post_predictions = evaluate(
            full, test, embeddings, answer_to_id, answers, device
        )
        post_generic, _ = evaluate(
            generic, test, embeddings, answer_to_id, answers, device
        )
        results["tasks"][str(task)] = {
            "pre_sleep_slow_only": pre_full,
            "wake_full_with_episodic_memory": memory_metrics,
            "post_sleep_slow_only": post_full,
            "generic_pre_sleep": pre_generic,
            "generic_post_sleep": post_generic,
            "slow_only_consolidation_gain": post_full["accuracy"]
            - pre_full["accuracy"],
        }
        for phase, rows in [
            ("pre_sleep", pre_predictions),
            ("wake_memory", memory_predictions),
            ("post_sleep", post_predictions),
        ]:
            all_predictions.extend(
                [{"task": task, "phase": phase, **row} for row in rows]
            )

    results["runtime_seconds"] = time.perf_counter() - start
    results["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
    results["mean_consolidation_gain"] = sum(
        task["slow_only_consolidation_gain"] for task in results["tasks"].values()
    ) / 5
    results["mean_post_sleep_accuracy"] = sum(
        task["post_sleep_slow_only"]["accuracy"] for task in results["tasks"].values()
    ) / 5
    results["mean_generic_post_sleep_accuracy"] = sum(
        task["generic_post_sleep"]["accuracy"] for task in results["tasks"].values()
    ) / 5

    (output_dir / "pilot_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (output_dir / "pilot_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row) + "\n")
    torch.save(full.state_dict(), output_dir / "neurostack_pilot.pt")
    torch.save(generic.state_dict(), output_dir / "generic_pilot.pt")
    return results

