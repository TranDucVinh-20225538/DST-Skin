# Skin Lesion OOD Detection and Triage

## Overview

This repository implements a pipeline for out-of-distribution (OOD) detection and triage in skin lesion classification.
The method is based on Mahalanobis distance applied to feature embeddings extracted from ResNet-18, ResNet-50, and EfficientNet-B3 models trained on ISIC 2018.

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the project root:

```bash
export PYTHONPATH=.
```

---

## Data Layout

Expected structure:

```text
data/
├── raw/
│   ├── isic2018/
│   └── pad_ufes20/
├── processed/
│   ├── isic2018_binary/
│   └── pad_ufes20_binary/
└── models/
    ├── resnet18_robust.pth
    ├── resnet50_robust.pth
    └── efficientnet_b3_robust.pth
```

---

## Pipeline

The pipeline consists of three steps:

1. Feature extraction
2. Score computation (OOD metrics)
3. Visualization and analysis

---

## Extract Features

```bash
PYTHONPATH=. python scripts/extract_once_resnet18.py
PYTHONPATH=. python scripts/extract_once_resnet50.py
PYTHONPATH=. python scripts/extract_once_efficientnet_b3.py
```

Outputs are saved to:

```text
outputs/features/
```

---

## Compute OOD Scores

```bash
PYTHONPATH=. python scripts/analyze_benchmark_resnet18.py
PYTHONPATH=. python scripts/analyze_benchmark_resnet50.py
PYTHONPATH=. python scripts/analyze_benchmark_efficientnet_b3.py
```

Outputs:

```text
outputs/reports/*_score_comparison.csv
outputs/reports/*_risk_coverage_*.csv
```

---

## Generate Figures

### Triage

```bash
PYTHONPATH=. python scripts/plot_triage_final_all.py
PYTHONPATH=. python scripts/plot_triage_ood_final.py
```

### Ablation

```bash
PYTHONPATH=. python scripts/plot_ablation.py
```

### UMAP

```bash
PYTHONPATH=. python scripts/plot_umap_visual.py
```

### Grad-CAM

```bash
PYTHONPATH=. python scripts/grad_cam.py
```

---

## Notes

* All paths are relative to the project root.
* Feature extraction scripts overwrite existing outputs.
* Run extraction immediately before analysis to ensure consistency.
* Checkpoints must be available under `data/models/`.
