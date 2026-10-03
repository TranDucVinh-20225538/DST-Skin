# DST-Skin — Detector Ranking Is Not Architecture-Invariant Under Medical Covariate Shift

Code, frozen results and the MIDL 2027 submission for a study of **post-hoc
out-of-distribution (OOD) detectors** on medical images.

## The question, in plain terms

A classifier trained on images from some hospitals will see images from a new
hospital that look different (stain, scanner, camera). An *OOD detector* flags
inputs that look unlike the training data, so they can be sent to a human
instead of trusted. "Post-hoc" detectors (MSP, Energy, Mahalanobis, kNN, ...)
are scores computed from an already-trained network, and papers usually rank
them on **one** backbone network.

We hold one medical shift fixed and ask: **if only the backbone changes, does the
detector ranking change too — and does that matter for deployment?**

## What we found (short version)

1. **Logit-based scores (MSP, Energy, ...) reorder a lot across backbones** on
   Camelyon17. MSP AUROC goes from 0.515 (ResNet-50) to 0.883 (DenseNet-121);
   the DenseNet jump holds on 5/5 training seeds.
2. **But they also reorder across training seeds of the same backbone.** The
   ranking agreement across seeds (Kendall's W 0.600–0.931) is about the same as
   across backbones (0.694), so "the architecture decides the ranking" is *not*
   supported for the full ranking.
3. **Feature-space scores (Mahalanobis, kNN) are the stable default.** One of
   them is the best detector on every backbone and every seed tested, on
   Camelyon and on skin.
4. **AUROC is a poor guide to deployment.** Choosing a backbone by AUROC instead
   of by *coverage@risk* (how much of the new hospital's data can be
   auto-accepted while keeping error ≤ 10%) loses 0.215 coverage on held-out
   patients.
5. The Camelyon pattern does not generalise as a law: skin images show no MSP
   jump, and on MIDOG the AUROC-vs-coverage disagreement points the other way.

No new detector is proposed. The contribution is the evaluation protocol:
default to feature-space scores, don't copy a logit-score ranking across
backbones or seeds, and pick a triage backbone from coverage@risk measured on a
validation split of the target hospital.

## Glossary

| Term | Meaning here |
|---|---|
| ID / OOD | in-distribution (training hospitals) / out-of-distribution (new hospital) |
| AUROC | how well a detector score separates ID from OOD images (0.5 = chance) |
| Kendall's W | agreement between several rankings of the same 7 detectors (1 = identical, 0 = none). Computed **per dataset, never averaged across datasets** |
| MSP jump | a backbone's MSP AUROC minus the ResNet mean on the same split; ≥ 0.15 counts as a jump |
| coverage@risk 10% | largest fraction of OOD images that can be accepted, most-confident first, while classifier error stays ≤ 10% |
| precommit | a `decision_precommit_*.md` file that fixes the experiment and how each outcome will be written up **before** running it |

## Datasets and numbers

Primary statistic: Kendall's W over 7 scores (MSP, Energy, ELogitNorm, ViM,
ReAct, Mahalanobis, kNN). Frozen in `outputs/reports/OFFICIAL_W_FREEZE.txt`.

| Dataset | Shift | Role | Backbones | Official W |
|---|---|---|---|---|
| Camelyon17 (WILDS) | hospitals 0,3,4 → hospital 2 | main result | 8 CNNs, seed 42 | **0.694** |
| Skin ISIC 2018 → PAD-UFES-20 | dermoscopy → smartphone photos | replication test | 8 CNNs | **0.791** |
| MIDOG (OpenMIBOOD 1a → 1b+1c) | scanner/tumour type | deployment readout | 3 (R18/R50/EffB3) | **0.857** |
| iWildCam (WILDS) | camera traps, non-medical | negative control | 8 CNNs | not pooled; 0/8 jump |

The 8 CNNs: ResNet-18/50, DenseNet-121, ConvNeXt-Tiny, MobileNetV3-L,
RegNetY-3.2GF, EfficientNet-B3 (300 px), EfficientNetV2-S (224 px). A ViT-B/16
was a pre-registered prediction (MSP ≥ 0.6947; observed 0.884) and is kept out
of W.

Controls added 2026-10-03 (precommits locked 2026-10-02):

| Control | Result | File |
|---|---|---|
| Cross-seed W, fixed architecture | R50 0.931, R18 0.800, DenseNet 0.789, EffB3 0.600 | `outputs/reports/camelyon_cross_seed_w.txt` |
| Coverage selection on patient-disjoint split | AUROC-pick 0.539 vs coverage-pick 0.754 held-out | `outputs/reports/camelyon_coverage_val_split.txt` |
| Bootstrap CIs on coverage | DenseNet 0.078 [0.067, 0.086], MobileNet 0.904 [0.898, 0.909] | `outputs/reports/pathology_risk_coverage/coverage_bootstrap_ci.txt` |

## Paper

The submission is **MIDL 2027** (full paper, deadline 2026-12-04), in
`manuscript/midl2027/`:

```
manuscript/midl2027/
├── main.tex, main.pdf   # 10-page main text + references + appendix
├── refs.bib, midl.cls   # bibliography, official MIDL class (needs jmlr.cls)
├── make_figures.py      # rebuilds fig/ from the frozen CSVs; never recomputes W
├── fig/
└── Makefile
```

```bash
cd manuscript/midl2027
make figures   # optional: regenerate fig/ (uses ../../.conda-env)
make main      # pdflatex + bibtex + 2x pdflatex
```

`make` sets `TEXMFHOME=./texmf`, a local TeX tree used on our HPC (not in git).
Elsewhere, a full TeX Live with `texlive-publishers` (for `jmlr.cls`) and
`texlive-science` (for `algorithm2e`) is enough.

An earlier CVPR 2027 version was dropped on 2026-10-03; its sources are in git
history (commit `18cbd2c`, `manuscript/cvpr2027/`).

## Repository layout

```
DST-Skin/
├── README.md
├── decision_precommit_*.md       # one per experiment: what runs + how each outcome is written
├── manuscript/midl2027/          # the paper
├── outputs/reports/              # tracked: every number the paper uses
│   ├── OFFICIAL_W_FREEZE.txt     # official W per dataset — the source of truth
│   ├── NARRATIVE_LOCKS.txt       # dated log of every experiment and its verdict
│   ├── architecture_invariance_ranks.csv   # 7-score AUROC + rank per backbone/dataset
│   ├── camelyon17/frac1/seed{42..46}/      # per-backbone score CSVs, Camelyon
│   ├── skin/, midog/, iwildcam/            # same for the other datasets
│   └── pathology_risk_coverage/            # coverage@risk tables
├── scripts/                      # one script per analysis step + SLURM submit_*.sh
├── src/                          # models, datasets, OOD scores (src/utils/scoring.py)
├── docs/REPOSITORY_GUIDE.md      # audit of the original skin-only codebase (predates the rest)
└── data/, outputs/features/, logs/   # local/HPC only, not in git
```

Features and checkpoints (`outputs/features/`, `data/models/`) are not in git;
they are tens of GB. Every number in the paper is read from `outputs/reports/`.

## Where each result comes from

| Paper result | Script | Output |
|---|---|---|
| Official W, ranks, MSP jump | `scripts/rebuild_architecture_invariance.py` | `architecture_invariance_*.csv`, `OFFICIAL_W_FREEZE.txt` |
| Five-seed MSP jumps | `scripts/camelyon_seed_jump.py` | `camelyon_seed_jump.txt` |
| Cross-seed W | `scripts/camelyon_cross_seed_w.py` | `camelyon_cross_seed_w.{txt,csv}` |
| Coverage@risk (full test set) | `scripts/pathology_risk_coverage.py` | `pathology_risk_coverage/` |
| Coverage held-out selection | `scripts/camelyon_coverage_val_split.py` | `camelyon_coverage_val_split*.{txt,csv}` |
| Coverage bootstrap CIs | `scripts/bootstrap_coverage_ci.py` | `pathology_risk_coverage/coverage_bootstrap_ci.txt` |
| Leave-one-backbone W | `scripts/camelyon_loo_backbone_w.py` | `architecture_invariance_loo_backbone_w.csv` |
| Permutation null for W | `scripts/kendall_w_perm.py` | `kendall_w_perm.{txt,csv}` |
| Matched-recipe (M1/M2) | `scripts/camelyon_matched_recipe_read.py` | `decision_precommit_matched_recipe.md` |
| Foundation models | `scripts/fm_ood_pilot.py` | see `NARRATIVE_LOCKS.txt` |

Training and feature extraction run on SLURM via `scripts/submit_*.sh`
(e.g. `submit_camelyon17_full.sh` for the Camelyon 8-CNN zoo,
`submit_camelyon_seed_variance*.sh` for seeds 43–46).

`outputs/reports/skin/seed43–46/` and `outputs/reports/midog/seed43–46/` hold
score CSVs from extra-seed runs of the skin and MIDOG zoos. They are not used in
the paper or in any official W.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Rules this repo follows

- Official W is never recomputed by the paper build or by follow-up scripts.
- W is reported per dataset and never averaged across datasets.
- Seed-42 artifacts are never overwritten; other seeds write to `seed{N}/`.
- Our in-house MIDOG ResNet-50 (MSP 0.512) is never mixed with OpenMIBOOD's
  public checkpoint (~0.59).
- Do not run `rebuild_architecture_invariance.py` on ad-hoc CSVs (matched-recipe,
  stain, shift-type, seed runs); see the matching precommit.

## Open items before submission

- The Datko et al. (ESWA 2026) citation is not verified against a primary source
  (flagged in `refs.bib` and in a footnote).
- The held-out coverage result uses one split of nine patients; repeated
  patient-level resampling was not run.
- Cross-seed W was computed for Camelyon only.
