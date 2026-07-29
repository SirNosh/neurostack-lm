# Experiment 2 report

## Outcome

**Experiment 2 stopped at Phase A1: foundational negative.**

The dense-core identity milestone passed, but the first support/working-memory
qualification seed failed three mandatory gates. The protocol requires every
seed to pass and says to stop Experiment 2 if A1 fails. Seeds 2718 and 3141
therefore were not launched: once seed 1729 failed, they could not make an
all-seeds criterion true. No A2, A3, A4, integration, sleep, or confirmatory
run was started, and no `experiment2-support-qualified` tag was created.

Experiment 1 and its Stage 1R negative conclusion remain unchanged.

## Dense-core qualification

The new Experiment 2 path uses the pinned frozen Qwen2.5-0.5B-Instruct
revision, four dense branches at layers 6, 12, 18, and 24, and no sparse
router. At zero initialization:

- all four branches executed at every insertion layer;
- the wrapped and unwrapped backbone logits matched exactly (maximum
  difference 0.0);
- the frozen-backbone hash was unchanged;
- peak CUDA allocation was 0.950 GiB.

The A1 runner also passed an interrupted/resumed equivalence audit: two-step
continuous and resumed runs produced identical trainable tensors (maximum
difference 0.0), metrics, learning rates, scheduler state, and backbone hash.

## Phase A1 result

Seed 1729 trained for 3,500 optimizer steps and stopped after four consecutive
non-improving evaluations. Checkpoint selection used the preregistered
geometric mean of pooled fact AUPRC, support recall, and answer accuracy. The
selected checkpoint was step 2,500.

| Metric | Required | Result | Gate |
|---|---:|---:|---|
| Pooled fact AUPRC | >= 0.80 | 0.1818 | Fail |
| Every-task fact AUPRC | >= 0.70 | 0.1727-0.8009 | Fail |
| Support recall at number of supports | >= 0.80 | 0.5968 | Fail |
| WM lesion answer drop | >= 3 points | 79.72 points | Pass |
| Noncollapsed writes | required | 2.08% predicted writes | Pass |
| Answer exact match | selection component | 84.56% | — |
| Lesioned answer exact match | diagnostic | 4.84% | — |

Per-task fact AUPRC was:

- qa1: 0.2914
- qa2: 0.2544
- qa3: 0.1727
- qa4: 0.8009
- qa5: 0.4845

The frozen-backbone hash remained unchanged.

## Interpretation

The dense A1 system learned a highly useful answer-conditioning path through
working memory: disabling that path reduced answer accuracy from 84.56% to
4.84%. It did **not** learn the intended general supporting-fact selector.
Only qa4 crossed the per-task ranking threshold, pooled AUPRC remained near
0.18, and support recall remained below 0.60.

This is evidence for a shortcut or task-specific latent memory solution, not
qualified support selection. The causal lesion result is real but cannot
override the failed fact-level gates. The correct preregistered conclusion is:

> One foundational mechanism could not reach minimum competence under direct,
> aligned development, so Experiment 2 stops before integration.

## Reproducibility

The complete run artifacts are under
`outputs/experiment2/A1-1729/`, including both checkpoints, the exact config
and manifest, train metrics, full final predictions, memory writes, resource
usage, final metrics, and the machine-readable gate report. Empty log files
for mechanisms not active in A1 are retained deliberately to satisfy the
common artifact contract.

The research and open-source basis for the mechanisms is documented in
`docs/research-basis.md`. Experiment 2 reused those documented primary
sources and official implementations as design references; it did not copy
third-party source code.
