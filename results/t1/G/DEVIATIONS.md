# T1 item G — deviations and implementation choices (written before G1 results)

All choices below are the least-invasive reading of the order; none changes a design, threshold or seed.

- **DG-1 Labels of the heads.** The R3 caches hold no labels. Per-patch `tumor` labels are taken from the WILDS
  metadata (`load_camelyon_metadata`, `wilds_split == 0`). G0 verifies, for all five FMs, that `groups_train` equals
  the metadata train-slide sequence row by row (abort otherwise), so labels and held-out positions are aligned.
- **DG-2 Logistic probe.** sklearn `LogisticRegression(C=1, L2, lbfgs)` on features standardised with the fit-set
  mean / std; `max_iter = 1000` instead of the sklearn default 100 (convergence only; iterations are logged in
  `g_head_acc.csv`). For ViM-with-head the standardisation is folded into a raw-feature head and the binary model is
  written as 2-class logits (0, z) (W = [0; w/sd], b = [0, b − Σ w·mu/sd]).
- **DG-3 Adapter.** D → 512 → ReLU → 2 (cross-entropy), AdamW lr 1e-3, 20 epochs, seed 42 (torch + shuffle
  generator), **batch 512** (unspecified in the order), float32 training on the same standardised inputs. ViM-with-head
  for the adapter runs on the 512-d hidden layer with the final linear layer as head (the head's logit space).
- **DG-4 ViM.** Package `crossfit_ood.scorers.ViMScorer` (default principal dimension rule; raw, un-normalised features
  for the logistic head, hidden features for the adapter), fitted on the fit set at each dose.
- **DG-5 Bootstrap draws.** `default_rng(2)`, B = 2000; per replicate, in this fixed order: slide multiplicities
  resampled with replacement within U, within E₁, within E₁∪E₂ (shared by the k = 0 placebo and k = 2, which have the
  same slides), within E₁…E₄, then OOD patch counts. One set of draws is reused for every FM, head, scorer and dose, and
  for the slide-equal sensitivity (paired). This implements the order's "within the seen set and within U"; R3 item 6
  resampled the union of its id slides. 95 % percentile CI.
- **DG-6 Scorer implementations.** `mahalanobis_l2` via the float64 GPU implementation `scorer64` (validated against
  the package in A); `knn_mean_cosine` (k = 50) via exact float64 GPU top-k cosine search (validated against R3 in B).
- **DG-7 P_Δ (A1 step 6, zero-parameter additive map), `mahalanobis_l2` only, k ∈ {1, 2, 4}.** On the fit set at dose
  k: Ledoit–Wolf on L2-normalised features, ρ̂_w from `group_stats`, s_logo by LOGO refits (G_fit ≤ 40), n_g = fit
  patches of each seen slide (its fit-eligible 70 %), π_g = share of seen held-out patches, ΔQ_pred by
  `dq_pred_groups`, P_Δ = AUROC(−(Q_U − |ΔQ_pred|), −Q_OOD) − AUROC(−Q_U, −Q_OOD). Not computed at k = 0 (the
  placebo slides are not in the fit set and k = 0 never enters c3).
- **DG-8 S5.** k = 0 is computed first; if the Δ_G(0) CI of an (FM, head, scorer) excludes 0, its doses 1, 2, 4 are
  `not run: S5` (recorded in `g1_frozen.csv`).
- **DG-9 c1 implementation checks.** Mahalanobis and kNN scores are computed independently inside each head's loop and
  required to be bit-identical (`np.array_equal`) for seen, unseen and OOD; K1 identity: |AUROC of the ω-mixture −
  (ω·A_seen + (1−ω)·A_unseen)| ≤ 1e-12 for ω ∈ {.25, .5, .75}, every cell.
- **DG-10 Held-out rounding.** n_heldout = Python `round(0.3·n_s)` as written. One slide has a .5 tie
  (slide 30, n = 9555 → 2866; half-up would give 2867). Kept literal; no effect on any rule.
- **DG-11 G2 not run.** A = NO-GO, so G2 does not run and G-ii = `not run`; G ∈ {PARTIAL, NO-GO, INCONCLUSIVE}.
- **DG-12 Scheduling.** G1 is one GPU job per FM (array %1), queued after H so that at most 4 GPUs run at once.
- **DG-13 Extra report-only files.** `g_slide_shares.csv` (per-slide held-out share within each seen set and U),
  `g0.json` (WARN records, block totals), `g_checks.csv`, `g_head_acc.csv` (head accuracy on held-out U),
  `g_linfit.csv` (linear-fit R² against k/4 and against ω_patch).
