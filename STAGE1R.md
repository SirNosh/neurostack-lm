# Stage 1R mechanism qualification

## Current status

The mechanism core and execution smoke test are implemented. Qualification training has **not** started, and none of the eight acceptance gates is claimed as passed.

The previous proxy remains preserved at Git tag `pilot-v0-negative`.

## Implemented

- Frozen token-level Qwen execution.
- Routed adapters after one-indexed layers 6, 12, 18, and 24.
- Up to three recurrent cognitive cycles.
- Four persistent 256-dimensional PFC slots with gated updates and scoped reset.
- Eight addressable working-memory slots with keep, replace, merge, clear, and protect.
- Four-slot workspace competition across token, PFC, working memory, episodic retrieval, specialists, verifier, and appraisal.
- Workspace broadcast as latent tokens on the next Qwen cycle.
- Top-2 sparse routing, batch load balancing, router z-loss, and annealed functional bootstrap labels.
- Session/task-scoped latent episodic events with no answer-label storage.
- Rank-8 fast matrices for PFC, working-memory read, retrieval integration, and router queries.
- Distinct DA/NE/ACh/5HT/overload control paths and a same-sized generic-controller mode.
- Single-logit verifier that refuses single-class training batches.
- Full-to-slow KL consolidation loss, retention term, and EWC penalty.
- Lesion switches and wake/sleep parameter permissions.

## Validation completed

- 25 deterministic tests pass.
- Six are integration/lifetime tests.
- A real-Qwen token-level smoke lifetime passed:

```json
{
  "backbone_frozen": true,
  "adapter_layers_one_indexed": [6, 12, 18, 24],
  "workspace_shape": [2, 4, 256],
  "episodic_events_written": 2,
  "retrieved_events": [1, 1],
  "fast_pfc_matrix_l1": 0.00299072265625,
  "peak_vram_gb": 1.0352,
  "status": "passed"
}
```

This proves execution and state transitions, not learning quality.

## Still required before qualification runs

1. Deterministic adapters/manifests for BABILong, EPBench, Multi-Session Chat, FewRel, PRM800K, CLUTRR, and the four TRACE blocks.
2. The R0, R1, and R2 conventional baseline implementations and parameter/FLOP matching.
3. Training/evaluation loops with example-level outputs and checkpoint recovery.
4. Short calibration jobs to measure VRAM, throughput, and wall-clock duration.
5. The full 18-run matrix after resource scheduling.
6. All eight acceptance gates passing across three seeds.

The run matrix must not begin merely because the code executes.

## Resource coordination

`neurostack-lm` records GPU/RAM/disk reservations in:

```text
C:\Users\devya\OneDrive\Desktop\resources.txt
```

No long GPU job may start until an active reservation and calibrated duration are recorded there.

