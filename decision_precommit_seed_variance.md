# Precommit: Camelyon seed-variance on 4 anchors — locked 2026-09-18 11:06

Do not reinterpret after seeing numbers. Official 8-CNN W stays **seed 42**. This table is robustness of the **jump**, not a new W.

Replaces MIDOG zoo. Does not reopen SupCon / ID-gate / coverage-loss.

---

## What runs

Camelyon hospital-2, same recipe as official cells.

| | |
|---|---|
| Anchors | `resnet18`, `resnet50`, `densenet121`, `effb3` (native **300**, the official cell) |
| Seeds | **42** already exists (do not retrain, do not overwrite). New: **43, 44, 45, 46** |
| Total new GPU | 4 × 4 = **16** trains + extract + 7-score analyze |
| Dynamics | `--log-msp-epoch`: MSP AUROC each epoch (ID-val vs OOD logits only). No Maha/kNN per epoch |
| Artifacts | `frac1/seed{N}/` — isolated from seed 42 |

For each seed \(s \in \{42,43,44,45,46\}\):

\[
\mathrm{jump}^{\mathrm{Dense}}_s = \mathrm{MSP}_{\mathrm{Dense}}(s) - \tfrac12\bigl(\mathrm{MSP}_{\mathrm{R18}}(s)+\mathrm{MSP}_{\mathrm{R50}}(s)\bigr)
\]

same formula for EffB3. Seed 42 uses the frozen official CSVs. Headline: mean ± sample std over 5 seeds.

---

## Read table (locked before GPU)

Jump threshold unchanged: **0.15**. Apply **separately** to DenseNet and to EffB3.

| result | write |
|---|---|
| mean jump ≥ 0.15 **and** ≥4/5 seeds jump | not a seed-42 fluke |
| mean jump ≥ 0.15 but **2+ seeds miss** | effect present, variance high; **report, do not hide**. §4.1 “6/6 non-ResNet…” gets a variance clause (`seed-robust ±σ` or equivalent) — not a rewrite of A |
| mean jump < 0.15 | seed-fragile; official A on n=8 seed-42 **cannot** be left unqualified |

Do not drop an anchor from the paper to “get a hit.” Do not add a 5th backbone to dilute a miss. Do not retrain seed 42.

R18/R50 must stay in the no-jump band on mean MSP; if a new seed of R18/R50 itself jumps (≥0.15 vs the *other* ResNet), report it — that is also variance, not a new A.

---

## Leave-one-backbone W (CPU, not a CI)

Jackknife: drop one of the 8 Camelyon backbones from the **seed-42** rank matrix, recompute Kendall W.

This is **structural sensitivity** of the concordance statistic. It is **not** a seed confidence interval. Do not write “W = 0.694, 95% CI [·].” Call it leave-one-backbone W. Official W remains the single-seed point estimate; limitations already say that.

---

## Training dynamics (same 16 runs)

For DenseNet and EffB3, record the first epoch where \(\mathrm{jump}_s \ge 0.15\) using that seed’s R18/R50 MSP **at the same epoch** if logged, else vs the seed’s **final** ResNet mean (document which). Prefer same-epoch ResNet MSP if all four anchors log.

Read as characterization, not a kill:

- Crosses at epoch 1–2 → inductive bias / init-ish
- Crosses late → learned representation

One paragraph. No extra mechanism hunt.

---

## Order

1. CPU leave-one-backbone W (this file’s companion CSV).
2. GPU 16 runs (4 parallel jobs, one backbone × seeds 43–46).
3. Writing related work in parallel. Abstract last.

Do not start MIDOG zoo. Do not mix new seeds into official W rebuild.
