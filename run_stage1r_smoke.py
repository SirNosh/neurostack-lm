"""Deterministic real-Qwen token-level Stage 1R smoke lifetime.

This validates execution and state boundaries only. It is not a qualification
benchmark and does not evaluate any Stage 1R acceptance threshold.
"""

from pathlib import Path
import json

import torch

from src.stage1r.mechanisms import EpisodicMemory
from src.stage1r.model import QWEN_REVISION, Stage1RNeuroStack


ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / QWEN_REVISION
)


def main() -> None:
    import transformers

    transformers.utils.is_flash_attn_2_available = lambda: False
    from transformers import AutoTokenizer

    torch.manual_seed(1729)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Stage1RNeuroStack.from_qwen(MODEL_PATH)
    prompts = [
        "Mary went to the kitchen. Where is Mary?",
        "John travelled to the office. Where is John?",
    ]
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")
    state = model.initialize_state(2, device="cuda", dtype=torch.bfloat16)
    memory = EpisodicMemory()

    model.set_wake_mode()
    wake = model(
        tokens.input_ids,
        tokens.attention_mask,
        state,
        memory,
        session_ids=["smoke-a", "smoke-b"],
        task_contexts=["babi", "babi"],
        cycles=2,
    )
    state = model.apply_wake_feedback(
        wake.final,
        outcome=torch.tensor([1.0, -1.0], device="cuda"),
        encode_mask=torch.tensor([True, True], device="cuda"),
        episodic_memory=memory,
        session_ids=["smoke-a", "smoke-b"],
        task_contexts=["babi", "babi"],
        timestamps=[1, 1],
        provenances=["smoke-1", "smoke-2"],
    )
    recalled = model(
        tokens.input_ids,
        tokens.attention_mask,
        state,
        memory,
        session_ids=["smoke-a", "smoke-b"],
        task_contexts=["babi", "babi"],
        cycles=1,
    )

    result = {
        "backbone_revision": QWEN_REVISION,
        "backbone_frozen": not any(
            parameter.requires_grad for parameter in model.backbone.parameters()
        ),
        "adapter_layers_one_indexed": [6, 12, 18, 24],
        "cognitive_cycles": 2,
        "workspace_shape": list(wake.final.workspace.slots.shape),
        "episodic_events_written": len(memory.events),
        "retrieved_events": [len(items) for items in recalled.final.retrieval.events],
        "fast_pfc_matrix_l1": float(
            state.fast_weights.pfc.matrix.abs().sum().float().cpu()
        ),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "status": "passed" if len(memory.events) == 2 else "failed",
    }
    output = ROOT / "outputs" / "stage1r_smoke.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
