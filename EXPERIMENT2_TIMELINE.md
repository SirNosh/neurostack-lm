# Experiment 2 timeline

## 2026-07-28

- Created `codex/experiment2` from Experiment 1 final commit `8b01d49`. Experiment 1 code, artifacts, tags, and stopping decision remain unchanged.
- Decision: implement Experiment 2 as a separate `src/experiment2` package and new manifests/configuration. Reuse frozen source data and small proven primitives only where their contracts still match.
- Phase 0 began with the four-branch dense adapter bank, zero-output identity initialization, full-sequence objective, fact-level support utilities, Experiment 2 schema, and stream-aware EPBench adapter.
- Added a stream runner that preserves event state through later questions and rejects mid-stream resets. The Experiment 2 core suite passes 9 tests, while all 67 Experiment 1 tests remain unchanged and passing.
- Real-Qwen identity audit passed at the pinned revision: all four branches executed at all four insertion layers, every initial adapter update was exactly zero, logits matched the unwrapped backbone exactly (max delta 0.0), and the backbone hash was unchanged. Peak CUDA allocation was 0.950 GiB.
- Corrected the new wrapper so adapter dtype conversion never recasts frozen Qwen buffers. This was caught by the identity audit before any training or milestone tag.
- Began A1 without changing the dense-core tag. Added a fact-level bAbI adapter, interaction scorer, and eight-slot bootstrap working memory. Tests prove relational-only adapter gradients, chronological distinct slot targets, operation/address output shapes, and a causal answer-logit change under the WM lesion. The complete suite now passes 79 tests.
- Added exact character-to-token span mapping and a real two-cycle A1 forward path: cycle-one fact scoring and slot writes, cycle-two eight-slot latent prefix conditioning, a shape-matched zero-memory lesion, and full-sequence per-example NLL. The causal tiny-backbone test reaches only relational branches and observes different lesioned logits.
- Added the recoverable A1 runner with the frozen 25,000/2,500/5,000 bAbI split, 2x8 effective batch 16, separate adapter/head learning rates, cosine warmup, 250-step checkpoints, trainable-only state serialization, frozen-backbone hash checks, and the complete required artifact filenames. Pre-calibration validation passes 81 tests.
