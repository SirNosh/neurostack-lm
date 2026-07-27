# NeuroStack-LM experiment timeline

## 2026-07-26

- Began by reading the project brief and Experiment 1 specification.
- Found an empty workspace with no existing Git repository or implementation.
- Decision: treat the broad project brief as scientific motivation and the Experiment 1 document as the implementation contract. Build the smallest auditable end-to-end pilot that preserves the requested comparisons and measurements; do not invent substitute datasets or silently claim the full confirmatory study.
- Downloaded the official bAbI v1.2 release and pinned the fixed Qwen2.5-0.5B-Instruct revision.
- Built and ran a one-seed diagnostic pilot with a parameter-matched generic baseline.
- Result: Stage 1 acceptance failed because supporting-fact selection was weak, routing collapsed, and retrieval/verifier acceptance metrics were unavailable. Per the preregistration, later wake/sleep results are diagnostic only.
- Added explicit research provenance for every current mechanism or scaffold. Key decision: do not equate a named head with an implemented biological mechanism; fast plasticity and differentiated modulation remain unimplemented.
- User requested periodic publication to `SirNosh/neurostack-lm`; repository was confirmed empty and GitHub authentication is active.
- Published the complete reproducibility package to GitHub in commit `9ea3e45`, including Git LFS checkpoints, cached frozen features, metrics, and example-level predictions.
- Research decision: use PBWM, shared global workspace, sparse MoE, Neural Episodic Control/DNC, Backpropamine, PRM process supervision, and complementary learning systems as algorithmic bases. Clearly mark reduced proxies and unimplemented mechanisms instead of relying on anatomical names.
- Tagged the audited negative pilot as `pilot-v0-negative` and pushed the tag.
- Implemented the Stage 1R stateful mechanism core and token-level frozen-Qwen adapter hooks.
- Added 25 deterministic tests (including six integration/lifetime tests); all pass.
- Ran a real-Qwen two-cycle smoke lifetime: four-slot broadcast, two scoped event writes, retrieval in both sessions, and nonzero rank-8 fast state all executed with the backbone frozen.
- Added shared-resource coordination under the name `neurostack-lm`; no long training job will start without an active reservation and calibrated duration.
- Decision: do not start the 18-run qualification matrix until dataset adapters, R0/R1/R2 baselines, and a short resource calibration exist. Execution success is not a qualification result.
