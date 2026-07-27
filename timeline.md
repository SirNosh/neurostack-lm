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
- Audited the mechanism core against the prequalification review. Corrected the R3 fairness issue with a same-sized unrestricted 5-to-12 controller, moved routing to the current contextual state after layer 5, and made working-memory and workspace decisions differentiable.
- Made retrieval breadth row-specific and evaluation-time episodic writes model-controlled. Added explicit bootstrap losses for working memory, workspace admission, and episodic write targets.
- Decision: count the current adapter bank honestly as dense four-expert execution. Sparse routing does not imply sparse compute until dispatch itself is sparse.
- Implemented the exact common `Stage1RExample` schema. Froze deterministic bAbI qa1-qa5 manifests (25,000 train, 2,500 dev, 5,000 official test) and the official BABILong 4K qa1-qa5 training selection (10,000 examples) at dataset revision `b3513ef7c25c54ce706054530d47668c532019d6`.
- Decision: do not invent BABILong supporting-fact annotations because the official training rows expose only input, question, and target.
- CPU validation now passes 42 tests, including an integrated two-cycle task-loss gradient test. The real-Qwen hash-verified smoke is waiting for the shared GPU ledger to become free.
- The integrated gradient test exposed an in-place PFC strategy-reset update that invalidated autograd version tracking. Replaced it with an out-of-place slot scale; the task loss now reaches working-memory operation/address heads and the workspace scorer.
- Re-ran the bounded real-Qwen lifetime after obtaining an exclusive ledger reservation. It passed in about 43 seconds at 0.962 GB peak CUDA allocation; the in-memory backbone SHA-256 was identical before and after, scoped recall returned one event per session, and the FLOP audit counted all four densely evaluated experts.
- Implemented R0 as four parameter-matched generic adapters on the same frozen Qwen token path and insertion layers. The real-Qwen three-pass smoke matched R5's 11,373,945 trainable parameters with 11,374,788 parameters (0.0074% error), preserved the backbone hash, used 0.964 GB peak CUDA allocation, and counted 681,676,800 adapter matmul FLOPs.
- Froze official PRM800K data at commit `7ecc794703b2877f63226f2477a49b34f9b25163`: 50,000 positive and 50,000 negative training steps plus 5,000/5,000 development steps, with complete source problems assigned to exactly one split.
- Froze official FewRel data at commit `278a2315d2138810a379cd8d5718914dc56e2582`: 64 meta-train and 16 held-out relations, with 1,000 deterministic 5-way episodes for each of 1-shot and 5-shot per partition.
- Fairness decision: R3 and R5 must receive identical controller inputs and downstream losses. A secondary R3+aux condition will match R5's auxiliary-prediction capacity without imposing semantic channel assignments.
