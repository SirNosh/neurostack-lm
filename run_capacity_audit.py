"""Generate the frozen Stage 1R parameter/capacity audit."""

import gc
import json
from pathlib import Path

import torch

from src.stage1r.audit import (
    expert_flop_audit,
    qwen_backbone_flops,
    tensor_payload_bytes,
)
from src.stage1r.baselines import (
    R0ParameterMatchedAdapter,
    R1OrdinaryRAG,
    R2RecurrentMemoryTokens,
)
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
WORKLOAD = {
    "batch_size": 1,
    "sequence_length": 512,
    "passes": 3,
    "episodic_entries": 128,
    "retrieval_breadth": 4,
}


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    sequence_lengths = [512, 516, 516]
    backbone_flops = qwen_backbone_flops(
        sequence_lengths=sequence_lengths,
        batch_size=1,
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        layers=config["num_hidden_layers"],
        attention_heads=config["num_attention_heads"],
        key_value_heads=config["num_key_value_heads"],
        vocabulary_size=config["vocab_size"],
    )

    r5 = Stage1RNeuroStack.from_qwen(MODEL_PATH, device="cpu")
    backbone_parameters = parameter_count(r5.backbone)
    total_parameters = parameter_count(r5)
    development_parameters = total_parameters - backbone_parameters
    sleep_parameters = parameter_count(r5.adapters) + parameter_count(
        r5.retrieval_integration
    )
    fast_parameters = parameter_count(r5.fast_weights)
    state = r5.initialize_state(1, device="cpu", dtype=torch.bfloat16)
    state_bytes = tensor_payload_bytes(state)
    episodic_bytes = WORKLOAD["episodic_entries"] * 4 * 256 * 4

    expert_flops = expert_flop_audit(
        batch_size=1,
        tokens=sum(sequence_lengths),
        cycles=1,
    )["actual_expert_flops"]
    summary_flops = (
        2
        * WORKLOAD["passes"]
        * 4
        * 4
        * config["hidden_size"]
        * 256
    )
    nonadapter_weights = sum(
        parameter.numel()
        for name, parameter in r5.named_parameters()
        if not name.startswith("backbone.")
        and not name.startswith("adapters.")
        and parameter.ndim >= 2
    )
    mechanism_flops = (
        expert_flops
        + summary_flops
        + 2 * WORKLOAD["passes"] * nonadapter_weights
    )

    neuro_common = {
        "total_parameters": total_parameters,
        "development_trainable_parameters": development_parameters,
        "sleep_trainable_parameters": sleep_parameters,
        "wake_mutable_state_bytes": state_bytes + episodic_bytes,
        "active_parameters_per_pass": [
            total_parameters,
            total_parameters,
            total_parameters,
        ],
        "total_backbone_flops": backbone_flops,
        "adapter_mechanism_flops": mechanism_flops,
        "memory_bytes": state_bytes + episodic_bytes,
        "maximum_passes": 3,
    }
    r4_state_bytes = state_bytes + episodic_bytes - (
        8 * (256 + 1280 + 256 + 64 + 256 + 256 + 512 + 256) * 2
    )
    systems = {
        "R1": {},
        "R2": {},
        "R3": {
            "status": "measured",
            **neuro_common,
            "controller": "generic five-dimensional",
        },
        "R3+aux": {
            "status": "measured",
            **neuro_common,
            "controller": "generic five-dimensional",
            "difference_from_R3": "auxiliary supervision objective; no added parameters",
        },
        "R4": {
            "status": "measured",
            **neuro_common,
            "wake_mutable_state_bytes": r4_state_bytes,
            "active_parameters_per_pass": [
                total_parameters - fast_parameters,
                total_parameters - fast_parameters,
                total_parameters - fast_parameters,
            ],
            "memory_bytes": r4_state_bytes,
            "controller": "differentiated; fast plasticity disabled",
        },
        "R5": {
            "status": "measured",
            **neuro_common,
            "controller": "differentiated",
        },
    }
    del state, r5
    gc.collect()

    r0 = R0ParameterMatchedAdapter.from_qwen(
        MODEL_PATH,
        target_trainable_parameters=development_parameters,
        device="cpu",
    )
    r0_parameters = parameter_count(r0)
    r0_trainable = r0.trainable_parameter_count
    r0_state_bytes = 4 * config["hidden_size"] * 2
    systems["R0"] = {
        "status": "measured",
        "total_parameters": r0_parameters,
        "development_trainable_parameters": r0_trainable,
        "sleep_trainable_parameters": r0_trainable,
        "wake_mutable_state_bytes": r0_state_bytes,
        "active_parameters_per_pass": [
            r0_parameters - parameter_count(r0.feedback_projection),
            r0_parameters,
            r0_parameters,
        ],
        "total_backbone_flops": backbone_flops,
        "adapter_mechanism_flops": r0.mechanism_matmul_flops(
            batch_size=1, tokens=512, passes=3
        ),
        "memory_bytes": r0_state_bytes,
        "maximum_passes": 3,
        "parameter_match_error_fraction": abs(
            r0_trainable - development_parameters
        )
        / development_parameters,
        "feedback_tokens": 4,
    }
    del r0
    gc.collect()

    r1 = R1OrdinaryRAG.from_qwen(
        MODEL_PATH,
        target_trainable_parameters=development_parameters,
        device="cpu",
    )
    r1_parameters = parameter_count(r1)
    rag_bytes = r1.capacity * 2 * config["hidden_size"] * 4
    systems["R1"] = {
        "status": "measured",
        "total_parameters": r1_parameters,
        "development_trainable_parameters": r1.trainable_parameter_count,
        "sleep_trainable_parameters": r1.trainable_parameter_count,
        "wake_mutable_state_bytes": rag_bytes,
        "active_parameters_per_pass": [r1_parameters] * 3,
        "total_backbone_flops": backbone_flops,
        "adapter_mechanism_flops": r1.mechanism_matmul_flops(
            batch_size=1, tokens=516, passes=3
        ),
        "memory_bytes": rag_bytes,
        "maximum_passes": 3,
        "retrieval": "session-scoped top-4 from 8192 latent entries",
        "parameter_match_error_fraction": abs(
            r1.trainable_parameter_count - development_parameters
        ) / development_parameters,
    }
    del r1
    gc.collect()

    r2 = R2RecurrentMemoryTokens.from_qwen(
        MODEL_PATH,
        target_trainable_parameters=development_parameters,
        device="cpu",
    )
    r2_parameters = parameter_count(r2)
    recurrent_bytes = 16 * config["hidden_size"] * 2
    systems["R2"] = {
        "status": "measured",
        "total_parameters": r2_parameters,
        "development_trainable_parameters": r2.trainable_parameter_count,
        "sleep_trainable_parameters": r2.trainable_parameter_count,
        "wake_mutable_state_bytes": recurrent_bytes,
        "active_parameters_per_pass": [r2_parameters] * 3,
        "total_backbone_flops": qwen_backbone_flops(
            sequence_lengths=[528, 528, 528],
            batch_size=1,
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            layers=config["num_hidden_layers"],
            attention_heads=config["num_attention_heads"],
            key_value_heads=config["num_key_value_heads"],
            vocabulary_size=config["vocab_size"],
        ),
        "adapter_mechanism_flops": r2.adapter_matmul_flops(
            batch_size=1, tokens=528, passes=3
        ),
        "memory_bytes": recurrent_bytes,
        "maximum_passes": 3,
        "memory_tokens": 16,
        "parameter_match_error_fraction": abs(
            r2.trainable_parameter_count - development_parameters
        ) / development_parameters,
    }
    del r2
    gc.collect()

    sleep_r0 = R0ParameterMatchedAdapter.from_qwen(
        MODEL_PATH,
        target_trainable_parameters=sleep_parameters,
        device="cpu",
    )
    consolidation_control = {
        "system": "R0 ordinary replay",
        "target_R5_sleep_trainable_parameters": sleep_parameters,
        "sleep_trainable_parameters": sleep_r0.trainable_parameter_count,
        "match_error_fraction": abs(
            sleep_r0.trainable_parameter_count - sleep_parameters
        )
        / sleep_parameters,
        "method": "narrower iterative R0; all non-backbone parameters trainable",
    }

    result = {
        "schema_version": "stage1r-capacity-v1",
        "backbone_revision": QWEN_REVISION,
        "workload": WORKLOAD,
        "flop_method": (
            "multiply-adds count as two FLOPs; exact Qwen linear and attention "
            "terms, dense four-expert adapter execution, and standardized "
            "non-backbone linear/einsum estimate"
        ),
        "memory_method": (
            "tensor payload bytes for online state plus 128 episodic entries "
            "(key, value, goal, workspace vectors); Python metadata excluded"
        ),
        "systems": {name: systems[name] for name in ("R0", "R1", "R2", "R3", "R3+aux", "R4", "R5")},
        "consolidation_control": consolidation_control,
    }
    output = ROOT / "outputs" / "parameter_capacity_audit.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
