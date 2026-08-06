# NeuroStack-LM

NeuroStack-LM tests whether a frozen pretrained language model benefits from an integrated, multi-timescale cognitive architecture: capacity-limited working memory, sparse specialist routing, episodic retrieval, fast plasticity, verification, and offline consolidation.

## Current result

**Experiment 1 stopped at the Stage 1R foundational gate.** The complete
three-seed R5 qualification wave ran on the frozen protocol, but support
selection, episodic retrieval, and verification failed in all three seeds.
The preregistered stop rule therefore prohibits launching attribution and
conventional-baseline waves.

| Foundational metric | Required | Three-seed result |
|---|---:|---:|
| Supporting-fact AUPRC | >= 0.80 | 0.350 mean (0.342-0.355) |
| Episodic recall@4 | >= 0.75 | 0.016 mean (0.011-0.026) |
| Episodic answer gain | >= 3 points | -0.2 mean (-0.4-0.0) |
| Verifier AUROC | >= 0.75 | 0.615 mean (0.600-0.627) |
| Verifier macro-F1 | >= 0.65 | 0.577 mean (0.563-0.593) |

Working-memory non-collapse passed in 3/3 seeds; router balance passed in 1/3
and workspace utilization in 2/3. Backbone hashes remained unchanged. This is
a valid negative result, not evidence against the broader hypothesis: the
implemented R5 mechanism stack did not qualify for the downstream comparison.
See [REPORT.md](REPORT.md) and the
[machine-readable summary](outputs/qualification/r5_wave_summary.json).

**Experiment 2 also stopped at its first foundational gate.** Its dense
four-branch core passed exact frozen-backbone identity tests, and A1 learned a
causally useful working-memory path (84.56% answer EM, falling to 4.84% when
lesioned). It did not qualify supporting-fact selection: pooled fact AUPRC was
0.182 and support recall 0.597. Because A1 required every seed to pass, the
first seed's failure triggered the registered stop before later seeds or
mechanisms. See [EXPERIMENT2_REPORT.md](EXPERIMENT2_REPORT.md) and the
[machine-readable stop record](outputs/experiment2/a1_foundational_stop.json).

## Reproduce

The pilot expects Windows, an NVIDIA GPU, Python 3.11, the official bAbI archive under `data/raw`, and the fixed Qwen snapshot in the Hugging Face cache.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python run_pilot.py
```

The exact dataset/model hashes and selection parameters are in [manifest.json](manifest.json). Research provenance and implementation status are in [docs/research-basis.md](docs/research-basis.md).

The fixed Stage 1R run matrix and gates are machine-readable in [configs/stage1r.json](configs/stage1r.json).

## Scientific guardrails

- Biological names are hypotheses about functional decomposition, not claims of biological fidelity.
- A module is not considered implemented merely because a similarly named tensor or head exists.
- Failed acceptance gates stop confirmatory interpretation.
- Existing benchmark language is used unchanged; only deterministic formatting and metadata are added.
- Matched baselines, lesions, multiple seeds, and held-out evaluations are required before a positive claim.
