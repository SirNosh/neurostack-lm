"""Bounded real-Qwen execution and parameter audit for the R0 baseline."""

import gc
import json
from pathlib import Path

import torch

from src.stage1r.audit import hash_module_parameters
from src.stage1r.baselines import R0ParameterMatchedAdapter
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

    r5 = Stage1RNeuroStack.from_qwen(MODEL_PATH)
    target = sum(
        parameter.numel() for parameter in r5.parameters() if parameter.requires_grad
    )
    del r5
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = R0ParameterMatchedAdapter.from_qwen(
        MODEL_PATH, target_trainable_parameters=target
    )
    r0_trainable = model.trainable_parameter_count
    before = hash_module_parameters(model.backbone)
    model.set_evaluation_mode()
    tokens = tokenizer(
        ["Mary went to the kitchen. Where is Mary?"],
        return_tensors="pt",
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    output = model(tokens.input_ids, tokens.attention_mask, passes=3)
    after = hash_module_parameters(model.backbone)
    result = {
        "backbone_revision": QWEN_REVISION,
        "backbone_hash_unchanged": before == after,
        "r5_trainable_parameter_target": target,
        "r0_trainable_parameters": r0_trainable,
        "parameter_match_error_fraction": abs(r0_trainable - target) / target,
        "adapter_bottlenecks": [
            adapter.bottleneck for adapter in model.adapters
        ],
        "backbone_passes": output.backbone_passes,
        "adapter_matmul_flops": model.adapter_matmul_flops(
            batch_size=1,
            tokens=tokens.input_ids.shape[1],
            passes=3,
        ),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "status": (
            "passed"
            if before == after
            and abs(r0_trainable - target) / target <= 0.02
            and output.backbone_passes == 3
            else "failed"
        ),
    }
    path = ROOT / "outputs" / "r0_smoke.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
