# Stage 1R mechanism qualification

## Current status

The complete three-seed R5-first qualification wave is finished. It failed the
foundational gate in all three seeds, so the preregistered protocol stopped
before attribution and conventional-baseline waves.

The previous proxy remains preserved at Git tag `pilot-v0-negative`.

## Implemented

- Frozen token-level Qwen execution.
- Routed adapters after one-indexed layers 6, 12, 18, and 24.
- First-pass routing from the current contextual input after layer 5.
- Up to three recurrent cognitive cycles.
- Four persistent 256-dimensional PFC slots with gated updates and scoped reset.
- Eight addressable working-memory slots with differentiable operation/address selection and explicit supervision losses.
- Differentiable four-slot workspace competition across token, PFC, working memory, episodic retrieval, specialists, verifier, and appraisal.
- Workspace broadcast as latent tokens on the next Qwen cycle.
- Top-2 sparse routing, batch load balancing, router z-loss, and annealed functional bootstrap labels.
- Session/task-scoped latent episodic events with no answer-label storage.
- Rank-8 fast matrices for PFC, working-memory read, retrieval integration, and router queries.
- Distinct DA/NE/ACh/5HT/overload control paths and a same-sized generic-controller mode.
- Single-logit verifier that refuses single-class training batches.
- Full-to-slow KL consolidation loss, retention term, and EWC penalty.
- Lesion switches and wake/sleep parameter permissions.
- Per-example retrieval breadth and model-controlled evaluation-time episodic writes.
- Honest dense-expert FLOP accounting.
- Common `Stage1RExample` schema and frozen manifests for bAbI, BABILong,
  PRM800K, FewRel, EPBench, Multi-Session Chat, CLUTRR, and TRACE.
- FewRel v2 with deterministic 48/16/16 relation partitions, episode-local
  A-E labels, support-label fast updates, and no evaluation gradients.
- A 21-run machine-readable matrix covering R0-R5 plus R3+aux.
- Parameter-matched R0, ordinary top-4 RAG R1, and 16-token recurrent-memory R2.
- A real-Qwen parameter/capacity audit with an R5 sleep-capacity-matched
  ordinary-replay R0 control and measured R1/R2 capacity.

## Validation completed

- 67 deterministic tests pass.
- Ten are integration/lifetime tests.
- A real-Qwen token-level smoke lifetime passed:

```json
{
  "backbone_snapshot_sha256": "9db106ad212ac058fee2222246427c9e63f1e5e1bafe2d1633b02f3939da6b67",
  "backbone_hash_unchanged": true,
  "backbone_frozen": true,
  "adapter_layers_one_indexed": [6, 12, 18, 24],
  "workspace_shape": [2, 4, 256],
  "episodic_events_written": 2,
  "retrieved_events": [1, 1],
  "fast_pfc_matrix_l1": 0.0022735595703125,
  "peak_vram_gb": 0.9620,
  "actual_dense_expert_flops": 440401920,
  "status": "passed"
}
```

This proves execution and state transitions, not learning quality.

The real-Qwen R0 execution audit also passed:

```json
{
  "r5_trainable_parameter_target": 11373945,
  "r0_trainable_parameters": 11374787,
  "parameter_match_error_fraction": 0.0000740,
  "feedback_tokens": 4,
  "backbone_passes": 3,
  "successive_logit_max_abs_deltas": [17.25, 13.625],
  "iterative_vs_zero_feedback_logit_max_abs_delta": 14.25,
  "total_mechanism_matmul_flops": 632110080,
  "backbone_hash_unchanged": true,
  "peak_vram_gb": 0.9599,
  "status": "passed"
}
```

The primary R0 now mean-pools the previous pass, projects it into four generic
feedback tokens, and prepends those tokens on the next pass. It uses the same
frozen Qwen revision, token-level execution, adapter locations, and maximum
three passes. It has no PFC, workspace, external episodic memory, or fast
weights. Its zero-feedback condition keeps the same four-token prefix shape.
Replay examples will be supplied by the common sleep runner.

## Qualification result

Timed R0/R2/R5 calibration passed with exact checkpoint-resume equivalence,
unchanged backbone hashes, and 2.51/2.64/2.68 GB peak CUDA allocation. The
frozen R5 qualification then completed seeds 1729, 2718, and 3141:

| Metric | Required | Mean | Range |
|---|---:|---:|---:|
| Supporting-fact AUPRC | >= 0.80 | 0.350 | 0.342-0.355 |
| Episodic recall@4 | >= 0.75 | 0.016 | 0.011-0.026 |
| Episodic answer gain | >= 3 points | -0.2 | -0.4-0.0 |
| Verifier AUROC | >= 0.75 | 0.615 | 0.600-0.627 |
| Verifier macro-F1 | >= 0.65 | 0.577 | 0.563-0.593 |

Support selection, episodic retrieval, and verification failed in 3/3 seeds.
Working-memory non-collapse passed in 3/3, router balance in 1/3, and workspace
utilization in 2/3. The backbone hash remained unchanged in every seed.

Decision: stop after Stage 1R. R3/R4 attribution and R0/R1/R2 baseline waves
were not run because their interpretation depends on a qualified R5 system.
Raw per-seed artifacts and the aggregate are under `outputs/qualification/`.

## Controller-supervision fairness contract

R3 and R5 receive identical controller inputs and downstream control losses.
R5 may receive the preregistered channel-specific auxiliary targets. The
secondary `R3+aux` condition applies the corresponding auxiliary objective to
five generic latents without changing parameter count or imposing a fixed
semantic assignment. This separates additional supervision from the proposed
biological channel-to-control factorization.

The run matrix must not begin merely because the code executes.

## Resource coordination

`neurostack-lm` records GPU/RAM/disk reservations in:

```text
C:\Users\devya\OneDrive\Desktop\resources.txt
```

No long GPU job may start until an active reservation and calibrated duration are recorded there.
