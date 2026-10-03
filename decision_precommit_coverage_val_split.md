# Precommit: Camelyon coverage@risk validation-split redo — locked 2026-10-02

Reviewer gap (advisor review, 2026-10): the main-paper coverage@risk comparison
(`main.tex` §4.5, "AUROC-best $0.078$ vs.\ coverage@risk-best $0.904$") selects
and evaluates on the **same labeled OOD test set**. That is a retrospective
diagnostic, not an executable backbone-selection procedure — nothing like it
is available at real deployment time, when OOD labels for the target hospital
do not exist. This file specifies the one fix that makes it executable:
select on a held-out validation half, evaluate on a disjoint test half.

Do not reinterpret after seeing numbers. Does not reopen seed-variance,
cross-seed $W$, SupCon, ID-gate, or the official 8-CNN $W$ rebuild — none of
those are touched by this file.

---

## What runs

No new training. This reuses the **already-extracted per-sample predictions**
for the official 8-backbone Camelyon hospital-2 zoo (seed 42) — the same
per-sample logits/features that produced `tab:cam` and the Fig.~3 coverage
numbers. This is a CPU-only resplit-and-recompute analysis, not a GPU job.

| | |
|---|---|
| Population | Camelyon17 hospital-2 OOD test set, official 8-backbone zoo, seed 42 |
| Split | Stratified **50/50** by slide/patient ID (never by individual patch — a patch-level split leaks slide-specific staining into both halves and invalidates the whole point), fixed seed **42** for the split itself (separate RNG stream from the model seed) |
| Halves | `val-half` (selection), `test-half` (evaluation) — written once, reused for every backbone/score, never resampled per-backbone |
| Metrics recomputed per half, per backbone | AUROC (all 7 scores), coverage@risk$10\%$ (MSP, matching the main-paper protocol) |

---

## The actual comparison (this is the fix)

For the 8-backbone zoo:

1. **AUROC-selection pipeline**: pick the backbone with highest MSP AUROC on `val-half`; report its coverage@risk$10\%$ on `test-half`.
2. **Coverage-selection pipeline**: pick the backbone with highest coverage@risk$10\%$ on `val-half`; report its coverage@risk$10\%$ on `test-half`.
3. **Oracle (current main-paper number, kept for contrast only, not as the headline)**: coverage@risk$10\%$ of the AUROC-best and coverage@risk-best backbones computed directly on the full labeled test set (no split) — this is `tab:cam`'s existing $0.078$ / $0.904$.

Pipelines 1 and 2 are the only two numbers allowed to be described as a
"selection procedure" anywhere in the paper. Pipeline 3 stays labeled
retrospective/oracle wherever it is quoted.

---

## Read table (locked before running)

| result | write |
|---|---|
| Pipeline 2's `test-half` coverage@risk is close to the oracle best (within ~0.05) **and** clearly beats Pipeline 1's `test-half` coverage@risk | the paper's recommendation ("estimate coverage@risk on a target-domain validation split, not from an AUROC table") is empirically supported — promote this to the headline number, demote the oracle $0.078$/$0.904$ to a footnote |
| Pipeline 2 beats Pipeline 1 but both are well below the oracle numbers | selection-from-a-validation-split helps but is noisy at $n=8$ backbones and a half-size test set — report honestly as a sample-size limitation, do not round up to "validation-split selection solves it" |
| Pipeline 2 does **not** reliably beat Pipeline 1 (comparable or worse `test-half` coverage@risk) | the paper cannot claim coverage@risk-based selection is actionable even from a validation split at this $n$; rewrite the recommendation to "AUROC is not a reliable proxy for coverage@risk; we do not yet have a validated alternative selection procedure" and keep Limitation 7 as an open problem rather than closing it |

Do not drop a backbone to improve either pipeline's result. Do not re-draw the
split after seeing which backbone each pipeline would pick. Do not run this on
MIDOG ($n=3$ is too small for a stable 50/50 split) — Camelyon $n=8$ only.

---

## Order

1. Write and freeze the val/test slide-ID split (one file, checked into
   `outputs/reports/`, same force-tracked convention as `OFFICIAL_W_FREEZE.txt`).
2. Recompute AUROC and coverage@risk$10\%$ per backbone on each half from the
   existing per-sample predictions (no re-extraction).
3. Run Pipelines 1–2, read against the table above.
4. Update `main.tex` §4.5 and Limitation 7 with whichever outcome obtains —
   do not leave both the old oracle framing and the new validation-split
   framing in the paper simultaneously.
