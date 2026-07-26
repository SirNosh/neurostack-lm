# Research basis and implementation status

This document separates three things that must not be conflated:

1. biological inspiration,
2. an algorithmic mechanism supported by prior machine-learning research,
3. what this repository actually implements today.

No source code below is copied into this repository. The papers and author-maintained implementations are design references; their licenses must be checked before any future code reuse.

## Frozen semantic backbone

- Basis: [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115) and [official model checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Current status: implemented for the pilot as a completely frozen feature extractor at revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Limitation: the pilot trains against cached final hidden states rather than inserting specialist adapters at Qwen layers 6, 12, 18, and 24.

## PFC-like persistent state and basal-ganglia-like working-memory gates

- Foundational computational basis: O'Reilly & Frank, [Making Working Memory Work](https://doi.org/10.1162/089976606775093909) (2006). The PBWM model uses actively maintained PFC representations and learned BG-mediated update gates.
- Related open code: [Emergent](https://github.com/emer/emergent), the successor framework from the O'Reilly lab for biologically based neural simulations. It is a framework reference, not a drop-in PBWM implementation for this project.
- Current status: partial proxy. Four GRU cells produce four 256-dimensional slots; a supervised relevance gate selects up to eight fact values.
- Missing: stripe-specific keep/replace/merge/clear/protect operations, persistent state across chunks, and actor-critic gate learning. The current implementation should not be described as a faithful PBWM reproduction.

## Capacity-limited global workspace

- Algorithmic basis: Goyal et al., [Coordination Among Neural Modules Through a Shared Global Workspace](https://arxiv.org/abs/2103.01197) (2021), which introduces competition for a bandwidth-limited shared workspace among specialist modules.
- Official code: no author-maintained repository was identified during the 2026-07-26 provenance search.
- Current status: partial proxy. The four highest-ranked working-memory values form a workspace summary.
- Missing: competition across all specified candidate systems and broadcast back into each module/Qwen cycle.

## Sparse specialist routing

- Algorithmic basis: Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (2017).
- Reference code: [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), the Google research codebase containing the early sparse MoE implementation.
- Current status: implemented in reduced form as four bottleneck experts with top-2 weighted routing.
- Known failure: the pilot router collapsed onto one expert in early tasks. Load balancing must pass the preregistered 10–50% utilization bounds before this mechanism is accepted.

## External working and episodic memory

- External differentiable memory basis: Graves et al., [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (2014) and the Differentiable Neural Computer; [official DeepMind DNC code](https://github.com/google-deepmind/dnc).
- Fast episodic retrieval basis: Pritzel et al., [Neural Episodic Control](https://arxiv.org/abs/1703.01988) (2017).
- Current status: a CPU-resident 8,192-entry store with normalized keys, top-4 cosine lookup, and answer-value aggregation.
- Missing: learned keys/values, recency and outcome biases, provenance-aware entries, FAISS indexing, task-context protection, write gates, and the specified eviction score.
- Pilot finding: retrieval reduced mean accuracy, so the current store is not an accepted mechanism.

## Fast plastic weights and neuromodulated plasticity

- Fast-weight basis: Ba et al., [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258) (2016).
- Differentiable Hebbian basis: Miconi et al., [Differentiable Plasticity](https://arxiv.org/abs/1804.02464) (2018), with [official Uber research code](https://github.com/uber-research/differentiable-plasticity).
- Neuromodulated update basis: Miconi et al., [Backpropamine](https://arxiv.org/abs/2002.10585), with [official Uber research code](https://github.com/uber-research/backpropamine).
- Current status: **not implemented**. Five scalar prediction heads exist, but no rank-8 fast matrix is updated and the heads do not yet control distinct pathways.
- Scientific constraint: calling the outputs dopamine-, norepinephrine-, acetylcholine-, serotonin-, or overload-like is a preregistered functional hypothesis. Backpropamine supports learned neuromodulation of plasticity, not this specific five-channel biological factorization.

## Predictive verification

- Algorithmic basis: Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023).
- Official data/code artifact: [OpenAI PRM800K](https://github.com/openai/prm800k).
- Current status: verifier head scaffold only.
- Missing: PRM800K positive and negative step supervision, earliest-error prediction, calibration, and ProcessBench held-out evaluation. The bAbI pilot supplied no meaningful negative labels, so verifier AUROC is unavailable.

## Wake/sleep separation and consolidation

- Biological theory basis: McClelland, McNaughton & O'Reilly, [Why There Are Complementary Learning Systems in the Hippocampus and Neocortex](https://doi.org/10.1037/0033-295X.102.3.419) (1995).
- Modern formalization and open code: Sharma et al., [Organizing memories for generalization in complementary learning systems](https://doi.org/10.1038/s41593-023-01382-9) (2023), with [author code](https://github.com/neuroai/Go-CLS_v2).
- Current status: wake writes only external state; sleep applies replay gradients to slow modules. Slow-only evaluation disables episodic retrieval.
- Missing: full-to-slow distillation, EWC, the preregistered replay mixture, TRACE task sequence, and old-task retention measurements.

## Adaptive computation and control labels

- The current `CONTINUE` / `ANSWER` / `ABSTAIN` and differentiated controller plan is an engineering hypothesis, not an implementation derived uniquely from a single biological paper.
- Before implementation, each control path requires an algorithmic source, an explicit target, a matched generic-controller baseline, and a lesion test.

