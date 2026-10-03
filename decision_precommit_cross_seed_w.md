# Precommit: cross-seed Kendall's W control (fixed architecture) — locked 2026-10-02

Reviewer gap (advisor review, 2026-10): official Camelyon $W=0.694$ is a single
seed-42 cross-architecture number, but Table~\ref{tab:seeds} already shows
ResNet-18 MSP alone ranging $0.524$--$0.712$ across five seeds for one
architecture. Without knowing how much the 7-score \emph{ranking} of a single
fixed architecture reshuffles across seeds, $W=0.694$ across architectures
cannot be attributed to architecture rather than training noise. This file
specifies the missing control.

Do not reinterpret after seeing numbers. Does not reopen the coverage@risk
validation-split file, the official seed-42 8-CNN $W$ rebuild, SupCon, or the
ID-gate. The seed-variance precommit (`decision_precommit_seed_variance.md`)
logged **MSP only**, per training epoch, for cross-seed robustness of the
jump — it cannot answer this question and is not reused for anything but its
checkpoints (see below).

---

## What runs

| | |
|---|---|
| Anchors | `resnet18`, `resnet50`, `densenet121`, `efficientnet_b3` — the same four anchors already trained across 5 seeds for `decision_precommit_seed_variance.md` |
| Seeds | $\{42, 43, 44, 45, 46\}$ — identical to the seed-variance run, same checkpoints |
| New work | For each of the $4\times5=20$ existing checkpoints: extract features/logits and compute **all seven scores** (MSP, Energy, ELogitNorm, ViM, ReAct, Mahalanobis+Ledoit-Wolf, $k$NN) on Camelyon hospital-2, not MSP alone |
| Precondition | Requires the seed-variance checkpoints (`frac1/seed{N}/`) to still exist on disk. If they were deleted after the seed-variance analysis, this requires re-training the same $4\times5$ cells (identical recipe, identical seeds) before scoring — do not substitute different seeds or anchors to avoid a retrain |
| Compute | Feature extraction + 7-score analyze only — no new training if checkpoints survive. GPU, but far cheaper than the original 16 training runs |

---

## The statistic

For each anchor architecture $a \in \{$R18, R50, DenseNet-121, EffB3$\}$ separately:

\[
W^{\text{seed}}_a = \text{Kendall's } W \text{ over the 7 scores, treating each of the 5 seeds as one ranking of that architecture's own 7-score AUROC vector.}
\]

This is the same Kendall's $W$ machinery already used for the official
cross-architecture number, just with "seed" substituted for "architecture" as
the ranking unit, and a single fixed architecture instead of all eight.

Report $W^{\text{seed}}_a$ for all four anchors (mean $\pm$ sd across the
four), alongside the official cross-architecture $W=0.694$ (seed 42).

---

## Read table (locked before running)

| result | write |
|---|---|
| $W^{\text{seed}}_a$ is high (e.g.\ $\gtrsim0.85$) and consistent across all four anchors | seed noise alone does not produce incomplete concordance at this scale; $W=0.694$ across architectures is attributable to architecture, not training variance — state this explicitly with the number, do not just assert it |
| $W^{\text{seed}}_a$ is comparable to or only modestly above the cross-architecture $0.694$ | the architecture claim is confounded with seed variance and cannot be cleanly separated from it with the current data; this must become a headline limitation, not a footnote, and the abstract/intro framing needs a further hedge beyond the logit-vs-feature-space split already applied |
| $W^{\text{seed}}_a$ varies a lot across the four anchors (e.g.\ DenseNet stable, EffB3 noisy) | report per-anchor, not just a mean — this is itself informative given EffB3 is already flagged as "not seed-stable" for the MSP jump |

Do not pool the four $W^{\text{seed}}_a$ values into a single number without
also reporting the per-anchor spread. Do not drop an anchor to raise the mean.
Do not treat a single high value as sufficient — report all four even if one
looks bad.

---

## Order

1. Confirm checkpoint availability for all 20 seed-variance cells before
   writing any code; if missing, get the retrain cost signed off before
   starting (it duplicates GPU-hours already spent once).
2. Feature-extract + 7-score analyze, isolated output directory
   (`frac1/seed{N}/scores_full/`), never overwriting the MSP-only seed-variance
   artifacts.
3. Compute $W^{\text{seed}}_a$ per anchor, read against the table above.
4. Update `main.tex` Limitation 8 and the softened "representation family"
   claim (abstract, intro, discussion, conclusion) with whichever outcome
   obtains.
