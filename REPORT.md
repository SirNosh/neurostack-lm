# Experiment 1 pilot report

## Stage 1R final outcome (2026-07-28)

**Negative / stopped at the three-seed foundational gate.** The frozen R5
qualification completed seeds 1729, 2718, and 3141. Support selection,
episodic retrieval, and verification failed in every seed:

| Metric | Required | Mean | Range |
|---|---:|---:|---:|
| Supporting-fact AUPRC | >= 0.80 | 0.350 | 0.342-0.355 |
| Episodic recall@4 | >= 0.75 | 0.016 | 0.011-0.026 |
| Episodic answer gain | >= 3 points | -0.2 | -0.4-0.0 |
| Verifier AUROC | >= 0.75 | 0.615 | 0.600-0.627 |
| Verifier macro-F1 | >= 0.65 | 0.577 | 0.563-0.593 |

Working-memory non-collapse passed in 3/3 seeds. Router balance passed in 1/3,
and workspace utilization passed in 2/3. All frozen-backbone hashes were
unchanged. The preregistered stop rule was applied, so R3/R4 attribution and
R0/R1/R2 baseline waves were not run. Their comparison would not rescue or
meaningfully interpret an R5 system that did not qualify.

This result falsifies the readiness of this implementation, not the general
neuro-inspired hypothesis. The main diagnosis is direct: the learned support
selector is near prevalence-level, episodic retrieval is effectively absent
and provides no answer gain, and verifier discrimination remains below its
minimum threshold. Full artifacts are in `outputs/qualification/`, including
`r5_wave_summary.json`.

The earlier bounded pilot report is retained below as historical context.

## Outcome

**Negative / stopped at the Stage 1 acceptance gate.**

The run used the fixed frozen `Qwen/Qwen2.5-0.5B-Instruct` revision and official bAbI tasks 1–5. It completed a bounded diagnostic wake/sleep lifetime for seed 1729, but it did not satisfy the mechanism-bootstrap prerequisites. Under the experiment specification, the architecture is not ready for the main comparison.

This is not the completed eight-system, five-seed, five-domain confirmatory experiment. Completing that campaign requires the remaining dataset adapters, full mechanism implementations, seven comparison systems, 35 training runs, held-out evaluations, lesions, and hierarchical statistics.

## Executed protocol

- Official bAbI 10k English release, tasks 1–5.
- 240 training and 100 held-out test examples per task.
- First 40 examples per task used for two bootstrap epochs.
- Remaining examples presented during wake without backpropagation.
- Episodic store retained up to 8,192 normalized keys and used top-4 cosine retrieval.
- Two replay epochs during each sleep phase, with 100 old examples retained.
- Frozen Qwen hidden states supplied 896-dimensional semantic features.
- NeuroStack and generic baseline had 6,122,136 and 6,122,231 trainable parameters respectively (0.0016% mismatch).

## Results

| bAbI task | Pre-sleep slow-only | Wake + episodic | Post-sleep slow-only | Generic post-sleep | Consolidation gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 17% | 20% | 29% | 32% | +12 |
| 2 | 31% | 15% | 25% | 15% | -6 |
| 3 | 18% | 21% | 22% | 20% | +4 |
| 4 | 16% | 16% | 64% | 68% | +48 |
| 5 | 17% | 20% | 49% | 30% | +32 |
| **Mean** | **19.8%** | **18.4%** | **37.8%** | **33.0%** | **+18.0** |

Runtime after feature extraction was 33.4 seconds. Peak recorded VRAM was 1.00 GiB because the frozen features were cached before controller training.

## Why the gate failed

1. Supporting-fact AUPRC ranged from 0.078 to 0.486 before each task’s sleep phase, far below the required 0.80.
2. Routing was not balanced. On task 1, expert 2 received 95.7% of routing mass; on task 2, expert 1 received 99.6%.
3. The pilot lacks a valid Stage 1 retrieval recall@4 target.
4. The verifier saw only positive labels in this bAbI-only run, so verifier AUROC cannot be measured.
5. The five controller outputs are present as heads but are not yet wired to distinct control paths with channel-specific supervision.

The apparent +18-point sleep gain therefore cannot validate the integrated architecture. The parameter-matched generic adapter also improved substantially, episodic retrieval was harmful on average, and there is only one seed.

## Most useful diagnosis

The current system mainly demonstrates ordinary replay learning around frozen Qwen features. It does **not** yet demonstrate differentiated cognitive control. Retrieval contamination across tasks explains the task-2 collapse plausibly, but this is an inference requiring targeted ablation. The next justified work is to implement and validate each mechanism against its acceptance metric before rerunning the lifetime.

Raw aggregate metrics are in [outputs/pilot_metrics.json](outputs/pilot_metrics.json), and 1,500 example/phase predictions are in [outputs/pilot_predictions.jsonl](outputs/pilot_predictions.jsonl).
