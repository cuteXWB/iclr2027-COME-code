# COME

Complementary Reliability Monitoring with Expert Allocation for Multi-Source-Free Graph Domain Adaptation.

## Contents

- `come.py`: structural stability, accumulated conflict, state-dependent losses, and Sparsemax expert allocation.
- `model.py`, `layer.py`: frozen GCN source models and target-adaptive GNN.
- `train_come.py`: target adaptation and five-run accuracy summary.
- `datasets.py`: loaders for CSBM, Twitch, and Citation.
- `configs/paper.json`: shared settings and target-specific optimizer settings.
- `pretrain/`: source checkpoints supplied with the original project archive.

## Setup

Install PyTorch, PyTorch Geometric, NumPy, SciPy, and scikit-learn in a compatible environment. Put the benchmark datasets under `data/CSBM/`, `data/Twitch/`, and `data/Citation/` using the raw-file structure expected by `datasets.py`.

Run one transfer setting across seeds 2080, 2090, 2100, 2110, and 2120:

```bash
python train_come.py --target DE --device cuda:0
```

For a single run, add `--seed 2080`. Supported targets are `CSBM-G4`, `DE`, `EN`, `DBLPv7`, `ACMv9`, and `Citationv1`. Target labels are read only to compute final accuracy.

The configuration uses GCN, 128-dimensional representations, two layers, five perturbed views, 2000 epochs, Adam, and $\lambda_{\mathrm{sta}}=1$, $\lambda_{\mathrm{con}}=0.05$. Source models are frozen during adaptation. Learning rates and weight decays are selected from the manuscript's grid. Additional implementation values, including perturbation rates and memory momentum, are recorded in `configs/paper.json`.

The included checkpoints predate this implementation. Full benchmark runs are needed to verify numerical agreement with the manuscript.
