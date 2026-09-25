# DST-Skin — Detector Ranking Is Not Architecture-Invariant Under Medical Covariate Shift

Research codebase and CVPR 2027 submission for a study of post-hoc out-of-distribution
(OOD) detectors under **medical covariate shift**. The question: holding one shift fixed,
how much does the ranking of OOD detectors move when only the backbone changes?

**Started as** a single-domain skin-lesion triage project (ISIC 2018 → PAD-UFES-20,
3 backbones, Mahalanobis-centric). **Now** a four-domain, eight-backbone characterization
study — the original skin pipeline is one of the four domains below, not the whole repo.
`docs/REPOSITORY_GUIDE.md` is a full audit of that original single-domain codebase and
predates the domains and backbones described here.

---

## Headline results (frozen — do not recompute without a precommit)

Primary rank-concordance statistic is **Kendall's $W$, computed per domain and never
pooled**. Numbers below are frozen in `outputs/reports/OFFICIAL_W_FREEZE.txt`.

| Domain | Role | Backbones | Official $W$ |
|---|---|---|---|
| Camelyon17 (WILDS), hospital-2 | founding cell | 8 CNN families, seed 42 | **0.694** |
| Skin ISIC → PAD-UFES-20 | tests replication | 8 CNN families | **0.791** |
| MIDOG (OpenMIBOOD 1a → 1b+1c) | deployment / utility readout | 3 (in-house R18/R50/EffB3) | **0.857** |
| iWildCam (WILDS), camera-trap | negative bound, natural-image | 8 CNN families | not pooled — 0/8 jump |

Other locked findings: a seed-robust MSP jump versus the frozen ResNet mean on
Camelyon (DenseNet-121 $5/5$ seeds, ConvNeXt-Tiny / EfficientNetV2-S@224 $4/5$,
MobileNetV3 / RegNetY $3/5$, EfficientNet-B3 $2/5$ — not seed-stable); feature-space
scores (Mahalanobis, $k$NN) hold ranks 1–2 on nearly every backbone on every domain;
the *sign* of the AUROC-vs-coverage@risk$10\%$ disagreement flips between Camelyon and
MIDOG; a pre-registered ViT-B/16 Camelyon MSP threshold ($\ge 0.6947$) was met
($0.884$, HIT), held out of the official $W$. No new detector is proposed — the
contribution is a protocol warning about copying a ranking across backbones.

---

## Manuscript

The CVPR 2027 submission lives in `manuscript/cvpr2027/`:

```
manuscript/cvpr2027/
├── main.tex        # 7-page main paper (review-mode CVPR template)
├── suppl.tex        # supplementary material (compiled separately)
├── main.pdf, suppl.pdf
├── fig/              # figures, built by make_figures.py from frozen CSVs
├── make_figures.py
└── refs.bib
```

Build:

```bash
cd manuscript/cvpr2027
pdflatex -interaction=nonstopmode main && bibtex main && pdflatex main && pdflatex main
pdflatex -interaction=nonstopmode suppl
# or: make
```

`manuscript/draft_v1.md` is the markdown twin kept in sync with `main.tex` — read it
for the same content without a LaTeX toolchain. Both are generated **from**
`outputs/reports/OFFICIAL_W_FREEZE.txt` and the other frozen CSVs below; neither
recomputes a number itself.

**Target venue:** CVPR 2027, deadline 2026-11-16 AoE (fixed, no extension).

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

`requirements.txt` covers all four domains (adds `wilds` for Camelyon17/iWildCam on
top of the original skin-pipeline dependencies).

---

## Repository layout

```
DST-Skin/
├── README.md                          # this file
├── docs/REPOSITORY_GUIDE.md           # audit of the original single-domain (skin-only) codebase
├── decision_precommit_*.md            # precommitted protocols for each robustness probe —
│                                       #   read before touching seed variance, matched-recipe,
│                                       #   shift-type, stain-cov, or the ViT/iWildCam extension
├── manuscript/
│   ├── draft_v1.md                    # markdown twin of the CVPR draft
│   └── cvpr2027/                      # LaTeX submission (see above)
├── outputs/reports/                   # tracked in git — the frozen numbers (CSVs, .txt locks)
│   ├── OFFICIAL_W_FREEZE.txt          # the single source of truth for Kendall W per domain
│   ├── NARRATIVE_LOCKS.txt            # chronological log of what was tested and closed
│   └── camelyon17/, iwildcam/, pathology_risk_coverage/, ...
├── scripts/                           # SLURM submit_*.sh drivers + one script per analysis step
├── src/                                # models, datasets, OOD scoring (src/utils/scoring.py)
└── outputs/{features,figures}/, data/, logs/   # gitignored — local/HPC only, not shipped
```

`outputs/*` is gitignored except `outputs/reports/`, which is force-tracked because it
**is** the frozen result set — everything in the manuscript is read from there, never
recomputed in the paper build.

---

## Reproducing a domain from scratch

Each domain follows the same three-step shape (feature extraction → score computation
→ aggregation); SLURM entry points are the `scripts/submit_*.sh` files. Example,
Camelyon17:

```bash
scripts/submit_camelyon17_pilot.sh        # or submit_camelyon17_full.sh for the 8-backbone zoo
python scripts/rebuild_architecture_invariance.py   # rebuilds architecture_invariance*.csv
```

Do **not** run `rebuild_architecture_invariance.py` against ad hoc CSVs (matched-recipe,
stain-cov, shift-type, seed-variance runs) — those are deliberately kept out of the
official rebuild; see the relevant `decision_precommit_*.md` for why.

The original single-domain skin pipeline (extract → analyze_benchmark → plot_*) still
works as documented in `docs/REPOSITORY_GUIDE.md`, with the caveats listed there
(two coexisting benchmark generations, a couple of scripts with stale hardcoded
checkpoint paths, some orphaned reader-study artifacts).

---

## Status

Numbers are frozen (`OFFICIAL_W_FREEZE.txt`, 2026-09-18) and the seed-variance /
shift-type / Phao-B / iWildCam-extra follow-ups are locked (`NARRATIVE_LOCKS.txt`,
2026-09-20). Draft is in submission-prep: one related-work citation
(Datko et al., ESWA 2026) is flagged unverified in `manuscript/cvpr2027/refs.bib` and
needs its original source confirmed before submission.
