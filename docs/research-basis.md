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

## Conventional memory baselines

- R1 basis: Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020), with the original implementation released through [Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/rag). Stage 1R implements the deliberately ordinary control required by the protocol: session-scoped cosine retrieval over at most 8,192 detached latent entries, top-4 content insertion, and no learned write gate, workspace, or fast plasticity.
- R2 basis: Bulatov, Kuratov & Burtsev, [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) (NeurIPS 2022), with [author-maintained code](https://github.com/booydar/recurrent-memory-transformer). Stage 1R prepends exactly 16 memory tokens, recurs their output representations across at most three complete frozen-backbone passes, and has no external store.
- Both controls use the same frozen Qwen revision, token path, insertion layers, and trainable-parameter envelope as R5. They are reduced controls, not reproductions of every training detail in the source papers.

## PFC-like persistent state and basal-ganglia-like working-memory gates

- Foundational computational basis: O'Reilly & Frank, [Making Working Memory Work](https://doi.org/10.1162/089976606775093909) (2006). The PBWM model uses actively maintained PFC representations and learned BG-mediated update gates.
- Related open code: [Emergent](https://github.com/emer/emergent), the successor framework from the O'Reilly lab for biologically based neural simulations. It is a framework reference, not a drop-in PBWM implementation for this project.
- Pilot status: four zero-initialized GRU transformations were non-persistent.
- Stage 1R status: four persistent 256-dimensional slots expose initialize, gated update, masked reset, detach, and cross-cycle state passing. Working memory has addressable keep/replace/merge/clear/protect operations; operation and address choices use straight-through Gumbel-softmax during training.
- Missing: a faithful PBWM actor-critic/PVLV learning rule. The implementation is a functional abstraction, not a PBWM reproduction.

## Capacity-limited global workspace

- Algorithmic basis: Goyal et al., [Coordination Among Neural Modules Through a Shared Global Workspace](https://arxiv.org/abs/2103.01197) (2021), which introduces competition for a bandwidth-limited shared workspace among specialist modules.
- Official code: no author-maintained repository was identified during the 2026-07-26 provenance search.
- Pilot status: the four highest-ranked facts were averaged.
- Stage 1R status: candidates compete from token state, PFC, working memory, episodic retrieval, specialists, verifier, and appraisal; exactly four selected slots are projected into latent tokens and prepended on the next Qwen cycle. Training uses straight-through soft admission gradients and exposes every candidate logit and label surface.
- Missing: evidence that competition learns useful admission policies and that workspace lesions cause a targeted held-out deficit.

## Sparse specialist routing

- Algorithmic basis: Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (2017).
- Reference code: [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), the Google research codebase containing the early sparse MoE implementation.
- Current status: implemented in reduced form as four bottleneck experts with top-2 weighted routing. Routing is computed from the contextual state after decoder layer 5 before the first adapter after layer 6.
- Known failure: the pilot router collapsed onto one expert in early tasks. Load balancing must pass the preregistered 10–50% utilization bounds before this mechanism is accepted.
- Stage 1R update: batch-level importance/assignment balancing is based on Shazeer et al.; logit stabilization follows [Switch Transformers](https://arxiv.org/abs/2101.03961) and its router z-loss. Weak family-to-expert labels are annealed to zero rather than retained as task routing.
- Compute qualification: the current implementation evaluates all four experts and then selects two. The FLOP audit therefore reports all four expert matmuls; true sparse dispatch is not claimed.

## External working and episodic memory

- External differentiable memory basis: Graves et al., [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (2014) and the Differentiable Neural Computer; [official DeepMind DNC code](https://github.com/google-deepmind/dnc).
- Fast episodic retrieval basis: Pritzel et al., [Neural Episodic Control](https://arxiv.org/abs/1703.01988) (2017).
- Pilot status: an answer-ID nearest-neighbor cache; retrieval reduced mean accuracy.
- Stage 1R status: session/task-scoped entries store latent event values, timestamps, goal/workspace state, outcome, confidence, and provenance. Retrieval returns latent tokens and never answer labels. Breadth is selected independently per example, and evaluation writes are controlled by the learned memory-write probability rather than an oracle annotation.
- Missing: trained recall@4, recency/outcome bias validation, FAISS scaling, eviction scoring, and the required misleading-memory causal test.

## Fast plastic weights and neuromodulated plasticity

- Fast-weight basis: Ba et al., [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258) (2016).
- Differentiable Hebbian basis: Miconi et al., [Differentiable Plasticity](https://arxiv.org/abs/1804.02464) (2018), with [official Uber research code](https://github.com/uber-research/differentiable-plasticity).
- Neuromodulated update basis: Miconi et al., [Backpropamine](https://arxiv.org/abs/2002.10585), with [official Uber research code](https://github.com/uber-research/backpropamine).
- Biological hypotheses behind the labels: dopamine as reward-prediction error follows Schultz, Dayan & Montague, [A neural substrate of prediction and reward](https://doi.org/10.1126/science.275.5306.1593) (1997); NE-like adaptive gain follows Aston-Jones & Cohen, [An integrative theory of locus coeruleus-norepinephrine function](https://doi.org/10.1146/annurev.neuro.28.061604.135709) (2005); uncertainty-sensitive ACh follows Yu & Dayan, [Uncertainty, neuromodulation, and attention](https://doi.org/10.1016/j.neuron.2005.04.026) (2005); and the four-way computational factorization is informed by Doya, [Metalearning and neuromodulation](https://doi.org/10.1016/S0893-6080(02)00044-8) (2002).
- Open-code status: those biological papers did not publish drop-in implementations for this architecture. The actual learned update rule is based on the author-released Backpropamine and differentiable-plasticity code above.
- Pilot status: **not implemented** in the tagged negative pilot.
- Stage 1R status: rank-8 local matrices and distinct control paths are implemented and covered by update/reset and control-effect tests. FewRel v2 explicitly writes support representations as keys and episode-local A-E label embeddings as values. Few-shot learning quality has not yet been measured.
- Scientific constraint: calling the outputs dopamine-, norepinephrine-, acetylcholine-, serotonin-, or overload-like is a preregistered functional hypothesis. Backpropamine supports learned neuromodulation of plasticity, not this specific five-channel biological factorization.

## Few-shot fast-learning benchmark

- Benchmark basis: Han et al., [FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset](https://aclanthology.org/D18-1514/) (2018).
- Official dataset/code: [THUNLP FewRel](https://github.com/thunlp/FewRel).
- Stage 1R use: 5-way 1-shot and 5-shot episodes, with held-out relations and no evaluation-time gradient updates.
- Current status: v1 remains frozen. V2 deterministically partitions the official 64 training relations into 48 meta-train and 16 meta-dev relations and retains the official 16 validation relations as held-out test. Each partition has 1,000 1-shot and 1,000 5-shot episodes with independently permuted episode-local labels A-E, visible support labels, hidden query labels, and instance-disjoint support/query rows. No fast-plasticity qualification result exists yet.

## Supporting-fact and long-context data

- bAbI basis: Weston et al., [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](https://arxiv.org/abs/1502.05698), with the [official Meta release](https://research.facebook.com/downloads/babi/).
- BABILong basis: Kuratov et al., [BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack](https://arxiv.org/abs/2406.10149), with [official code](https://github.com/booydar/babilong) and the [official 5K training release](https://huggingface.co/datasets/RMT-team/babilong-train-5k-samples).
- Stage 1R status: a common example schema, deterministic bAbI qa1-qa5 selection, exact supporting spans/IDs, and the official BABILong 4K qa1-qa5 2,000-per-task selection are implemented. Raw and formatted hashes are frozen in committed manifests.
- Limitation: the BABILong training release exposes `input`, `question`, and `target`, but no support IDs; the adapter leaves support annotations empty rather than manufacturing labels.

## Remaining memory, reasoning, and continual-learning benchmarks

- EPBench basis: He et al., [Episodic Memories Generation and Evaluation Benchmark for Large Language Models](https://openreview.net/forum?id=6OKsdO7BgO) (ICLR 2025), with [official code](https://github.com/ahstat/episodic-memory-benchmark) and the authors' Figshare data release. Stage 1R uses the Claude 10K/100K/1M books as train/dev/test, retains all events, and preserves exact answer-chapter IDs as retrieval targets. The downloaded supplemental archive contains answer outputs rather than a second book style, so no unseen-style claim is made.
- Multi-Session Chat basis: Xu et al., [Beyond Goldfish Memory: Long-Term Open-Domain Conversation](https://aclanthology.org/2022.acl-long.356/) (ACL 2022). The pinned public dataset mirror preserves native splits; Stage 1R groups all sessions for a conversation under one memory scope and resets only between conversations.
- CLUTRR basis: Sinha et al., [CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text](https://arxiv.org/abs/1908.06177) (EMNLP 2019), with [official generator/data code](https://github.com/facebookresearch/clutrr). Stage 1R trains at relation depths 2–4, develops at 5–7, and tests extrapolation at 8–10.
- TRACE basis: Wang et al., [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762) (2023), with [official code and data scripts](https://github.com/BeyonderXX/TRACE). Stage 1R uses the official 500-example release for C-STANCE, FOMC, MeetingBank, and Py150 in published order. State persists across blocks; task-boundary labels are evaluation metadata and are absent from prompts.

## Predictive verification

- Algorithmic basis: Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023).
- Official data/code artifact: [OpenAI PRM800K](https://github.com/openai/prm800k).
- Pilot status: invalid all-positive two-logit scaffold.
- Stage 1R status: one correctness logit with binary cross-entropy; single-class batches are rejected by construction.
- Stage 1R data status: the official PRM800K release is pinned at commit `7ecc794703b2877f63226f2477a49b34f9b25163`. Ratings `+1` and `-1` produce 50,000/50,000 balanced training steps and 5,000/5,000 development steps with source-problem-disjoint splits; neutral and flagged completions are omitted.
- Missing: earliest-error prediction, calibration, and held-out AUROC/F1.

## Wake/sleep separation and consolidation

- Biological theory basis: McClelland, McNaughton & O'Reilly, [Why There Are Complementary Learning Systems in the Hippocampus and Neocortex](https://doi.org/10.1037/0033-295X.102.3.419) (1995).
- Modern formalization and open code: Sharma et al., [Organizing memories for generalization in complementary learning systems](https://doi.org/10.1038/s41593-023-01382-9) (2023), with [author code](https://github.com/neuroai/Go-CLS_v2).
- Pilot status: ordinary supervised rehearsal.
- Stage 1R status: wake parameter immutability, allowed-only sleep parameters, full-to-slow KL, retention, and EWC terms are implemented and tested.
- Missing: the preregistered replay mixture, TRACE task sequence, and held-out consolidation measurements.
- Stage 1R update: the full-to-slow KL objective and EWC penalty are implemented. EWC follows Kirkpatrick et al., [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114) (2017). Dataset replay selection and qualification results remain outstanding.

## Adaptive computation and control labels

- The current `CONTINUE` / `ANSWER` / `ABSTAIN` and differentiated controller plan is an engineering hypothesis, not an implementation derived uniquely from a single biological paper.
- Before implementation, each control path requires an algorithmic source, an explicit target, a matched generic-controller baseline, and a lesion test.
