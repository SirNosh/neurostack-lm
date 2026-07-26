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
