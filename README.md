# NeuroStack-LM

NeuroStack-LM tests whether a frozen pretrained language model benefits from an integrated, multi-timescale cognitive architecture: capacity-limited working memory, sparse specialist routing, episodic retrieval, fast plasticity, verification, and offline consolidation.

## Current result

The repository currently contains a **bounded, one-seed bAbI pilot**, not the complete confirmatory Experiment 1.

That result is permanently preserved as Git tag `pilot-v0-negative`. Development has moved to [Stage 1R mechanism qualification](STAGE1R.md): the stateful mechanism core and real-Qwen execution smoke test now pass, but dataset qualification and the 18-run matrix have not started.

The pilot failed the preregistered Stage 1 mechanism gate:

- Supporting-fact AUPRC never approached the required 0.80.
- Specialist routing collapsed onto one expert during early tasks.
- The current verifier and differentiated modulator heads do not yet have the required negative/control supervision.

Diagnostic results after the failed gate:

| Metric | Result |
|---|---:|
| Mean NeuroStack pre-sleep slow-only accuracy | 19.8% |
| Mean NeuroStack post-sleep slow-only accuracy | 37.8% |
| Mean consolidation gain | +18.0 points |
| Mean parameter-matched generic post-sleep accuracy | 33.0% |
| Mean NeuroStack accuracy with online episodic retrieval | 18.4% |
| Trainable parameter mismatch | 0.0016% |

These numbers are useful for debugging, but the protocol prohibits treating them as positive evidence after the mechanism gate fails. See [REPORT.md](REPORT.md) for the full interpretation.

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
