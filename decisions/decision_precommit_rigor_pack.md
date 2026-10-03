# Precommit: rigor pack for the MIDL 2027 paper — written 2026-10-03, commit BEFORE running

Paper: "Detector Ranking is Not Architecture-Invariant under Medical Covariate Shift"
(`manuscript/midl2027/main.tex`, current framing: feature-space scores (Mahalanobis/kNN)
are the stable default; Kendall W cannot separate architecture from seed; choose the
triage backbone by coverage@risk on a target validation split).

This file fixes hypotheses, metrics and pass/fail bars **before any rigor-pack result
exists**. Commit it (with the scripts) before running `run_rigor_pack.sh`, so the git
timestamp proves the order. Do not edit the bars after seeing numbers; if a bar turns
out to be ill-posed, add a dated note at the end and report both readings.

Rules carried over (unchanged): official W (Camelyon 0.694 / skin 0.791 / MIDOG n=3 0.857)
is never recomputed or replaced; seed-42 artifacts are never overwritten; W is never pooled
across domains; nothing here edits `manuscript/`.

What was run while writing this file (not results): (a) the reproduction checks of
`w_from_csv.py` against already-published numbers (0.694, 0.791, cross-seed 0.931/0.800/
0.789/0.600, 4-anchor 0.750–0.915) — all PASS; (b) the a-priori power simulation below,
which uses no study data. No new statistic of the study was looked at.

---

## Inputs (inventory done on 2026-10-03 from the repo + logs)

| Input | Where | Status |
|---|---|---|
| 7-score AUROC/FPR95 CSVs, Camelyon 8 archs x seeds 42–46 | `outputs/reports/camelyon17/frac1/seed{42..46}/` (+ seed 42 AUROC from `architecture_invariance_ranks.csv`) | in git, complete 40/40 |
| same, skin ISIC→PAD and MIDOG, 8 archs x seeds 42–46 | `outputs/reports/{skin,midog}/...` | in git, complete 40/40 each |
| per-sample features/logits, Camelyon 8 x 5 | `outputs/features/camelyon17/frac1/seed{S}/{stem}_features.pt` (HPC) | logs say written for all 40 cells (2026-09-17..19); seed 42 used 2026-10-03. Verify with `check_inputs.py` |
| checkpoints | `data/models/camelyon17/frac1/seed{S}/{stem}_best.pth` (HPC) | expected present |
| WILDS metadata | `data/raw/wilds/camelyon17_v1.0/metadata.csv` (HPC) | present (used by the val-split script) |
| train-patch slide IDs | — | **not cached**: train features were saved shuffled without indices → GPU re-extract needed only for H11 |

---

## Hypotheses, metrics, bars

Notation: "run" = one (architecture, seed) training run; "cell" = one run evaluated on one
dataset. Primary dataset Camelyon17 hospital 2. All bootstrap B=1000, permutations 10,000,
RNG seed 42 unless stated.

### H1 — Rater-count-matched W (CSV only)
Metric: cross-seed W per architecture (k=5 seeds, n=7 scores) for **all 8** architectures,
compared with the distribution of cross-architecture W over all C(8,5)=56 five-architecture
subsets at each seed 42–46 (280 values pooled). Read per architecture: percentile of its
cross-seed W within the pooled cross-arch distribution.
- **Arch separable**: cross-seed W above the pooled q95 for ≥ 6/8 architectures.
- **Not separable (current paper claim stands)**: cross-seed W inside [q05, q95] for ≥ 6/8.
- Anything else: "mixed; per-architecture", reported as a table, no headline.
Same reading, separately (never pooled), for skin and MIDOG.

### H2 — W versus chance (CSV only)
Permutation null (within-rater shuffle, 10k) and Friedman chi2 for every cross-arch W
(k=8, each seed) and every cross-seed W (k=5, each arch). Report chance-corrected
W* = (W − E0[W]) / (1 − E0[W]). Bar: each W "above chance" iff Holm-adjusted (family
"W_vs_chance", per domain) p < 0.05. Expected to pass; reported for completeness.

### H3 — Sampling uncertainty of W (needs score caches)
Paired bootstrap over test samples (ID id_val + OOD hospital-2 resampled together, same
resample for every cell), two schemes: patch-iid and patient-cluster (ID clusters = id_val
patients, OOD clusters = the 9 hospital-2 patients). Statistics: cross-arch W (k=8, each
seed), cross-seed W (each arch), mean rater-matched k=5 cross-arch W, and
D = mean cross-seed W − mean k=5 cross-arch W.
- If the cluster 95% CI of D contains 0 → "seed and architecture concordance are not
  distinguishable even before seed noise is considered"; if it excludes 0 in the direction
  of D>0 → supports H1 "arch separable" (only if H1/H4 agree).
- Sample-noise bar: patch-iid CI half-width of cross-arch W(seed 42) < 0.05 → "test-set
  sampling is not what moves W"; otherwise say W is also test-set-noise limited.

### H4 — Architecture vs seed, the direct test (CSV only)  [PRIMARY for the W claim]
(a) Arch-label permutation: statistic = mean over archs of cross-seed W; null = randomly
re-assign the 40 runs to 8 groups of 5 (10k). (b) Nested ANOVA on within-run ranks (and on
AUROC): F = MS(method×arch) / MS(method×seed(arch)), permutation p (5k label shuffles) and
variance-component share of method×arch. Run on the 4 anchors and on all 8 archs.
- **Arch effect on ranking detected**: p_perm < 0.05 for (a) AND (b, ranks) on all 8 archs.
  Paper must then say "architecture explains part of the ranking variance beyond seed
  (share = X)", replacing "cannot be attributed to architecture alone" with a quantified
  statement; the 4-anchor result is reported as underpowered if it disagrees.
- **Not detected**: p ≥ 0.05 for both on all 8 → keep "W cannot separate architecture from
  seed", and add the sensitivity statement from the power table (what share would have been
  detected).
- Split decision (one test < 0.05, the other not): report both, no headline change.
A-priori sensitivity (simulation, `power_w.py`, no data; 400 sims, alpha 0.05, 5 seeds):
| arch share rho | power, 4 archs (perm / F) | power, 8 archs (perm / F) |
|---|---|---|
| 0.0 (type-I) | 0.03–0.06 / 0.05–0.06 | 0.03–0.04 / 0.05–0.06 |
| 0.1 | 0.19–0.25 / 0.26–0.29 | 0.26–0.42 / 0.35–0.45 |
| 0.2 | 0.37–0.58 / 0.45–0.59 | 0.61–0.85 / 0.68–0.86 |
| 0.3 | 0.62–0.77 / 0.70–0.80 | 0.86–0.98 / 0.90–0.98 |
| 0.5 | 0.90–0.99 | 1.00 |
(ranges over seed-noise levels calibrated to cross-seed W 0.60 / 0.78 / 0.93). Reading
fixed now: the 4-anchor design only has ≥80% power for rho ≳ 0.35; 8 archs for rho ≳ 0.2.
A null result on 4 anchors is therefore **not evidence of no architecture effect**. The
simulation also shows that E[W] falls with the number of raters k at rho=0 (k=4: ≈0.62 vs
k=8: ≈0.56 at W_seed 0.6), so comparing cross-seed W (k=5) with cross-arch W (k=8) is biased
against the architecture W — H1 removes this bias.

### H5 — Feature-space default (CSV + score caches)
(a) Count of cells where Mahalanobis or kNN is rank 1 (AUROC), over all 40 Camelyon cells
(and 40 skin, 40 MIDOG, reported separately).
- **"every backbone and every seed" sentence allowed** only if 40/40 Camelyon.
  If 36–39/40 → sentence becomes "N/40 runs". If < 36/40 → the stability claim is
  downgraded to "usually" with the count, and the abstract changes.
(b) Paired DeLong per cell, best feature score vs best non-feature score (selected by point
AUROC). Bar: Holm (across the 40 cells) p < 0.05 and positive difference in ≥ 36/40 cells.
Mahalanobis vs kNN: report how many cells have a Holm-significant difference (rank 1 vs 2
between them is otherwise called a tie in the text).

### H6 — Coverage-based backbone selection, repeated patient splits  [PRIMARY for the deployment claim]
Score cache + metadata. Families: F1 (primary) all 252 val sets of 4–5 patients; F2 all 168
val sets of 3 or 6 patients (frozen split is in F2); F3 all 510 proper splits; LOPO (9);
DROP1 (frozen split minus one patient, 9). Pipelines: AUROC-select (max val MSP AUROC),
coverage-select (max val coverage@risk 10% MSP), oracle (max test), random (mean over
archs). Seed modes: per_seed (primary), seed_avg, cross_seed. Metric: test coverage@risk
10% of the chosen arch; gain = coverage-pick − AUROC-pick.
Primary read = F1, per_seed, MSP, 10%:
- **Supported** ("choose the backbone by validation coverage@risk"): median gain > 0 AND
  coverage-select strictly wins in ≥ 60% of split×seed cases AND median gain > 0 in ≥ 4/5
  seeds.
- **Helps on average, noisy**: median gain > 0 but one of the other two conditions fails.
- **Not supported**: median gain ≤ 0 or strict-win fraction < 50% → recommendation becomes
  "AUROC is not a reliable proxy for coverage; no validated selection rule at n=8/9 patients",
  and the 0.539 → 0.754 frozen-split number is reported as one draw from the F2 distribution
  with its percentile.
Frozen split must reproduce 0.539 (ConvNeXt) / 0.754 (EffV2-S) / 1.000 (MobileNet) at seed 42
(REPRO line) before anything is read. LOPO/DROP1 are sensitivity only (no bar).
Ranking agreement: median Kendall tau(val AUROC, test coverage) and tau(val coverage, test
coverage) over F1; reported, no bar.

### H7 — Metric robustness of the ranking conclusions (CSV + score caches)
Recompute cross-arch W, cross-seed W and the feature-rank-1 count with FPR@95TPR (CSV,
seeds 43–46 complete; seed 42 only where all 7 FPR95 exist), AUPR-In and AUPR-Out (caches).
Bar: conclusions are "not AUROC-specific" iff for each metric |W_metric − W_AUROC| ≤ 0.10
for the seed-42 cross-arch W and the mean cross-seed W, AND feature rank-1 holds in ≥ 36/40
cells. Otherwise list which conclusion changes under which metric.

### H8 — Is the MSP jump an accuracy/temperature artifact? (score caches)
(a) Temperature: for 2-class softmax, MSP = sigmoid(|z1−z0|/T) is monotone in |z1−z0| for
any T>0, so MSP AUROC is mathematically T-invariant. Check: max |ΔAUROC| over T ∈ {0.5, 1, 2,
T*} < 1e-6 in all cells. Consequence fixed now: the paper's "temperature is not the
mechanism (gap 0.308→0.308)" must be re-worded as a mathematical identity, not an
experiment. Energy AUROC at T* is reported (it does change).
(b) Accuracy: OLS over the 40 runs, MSP AUROC ~ arch (+ hospital-2 accuracy, + id_val
accuracy). Read DenseNet − ResNet-mean adjusted jump with run-bootstrap 95% CI.
- **Jump not explained by accuracy**: adjusted DenseNet jump ≥ 0.15 and CI excludes 0 with
  both covariates. - **Partly explained**: CI excludes 0 but estimate < 0.15. - **Explained**:
  CI includes 0 → the DenseNet-jump sentence must add "after adjusting for hospital-2
  accuracy the jump is not distinguishable from zero".
Accuracy-matched pairs (|Δ hospital-2 acc| ≤ 0.02) reported, no bar.

### H9 — Coverage beyond MSP@10% (score caches)
(a) coverage@risk 5/10/20% with MSP, Energy, Mahalanobis, kNN as triage score, per cell, with
patch-iid and patient-cluster bootstrap CIs. (b) H6 repeated for every (score, risk).
Bar: the deployment recommendation is "risk-level robust" iff the H6 verdict for MSP is the
same at 5%, 10%, 20%; otherwise the paper states the risk level it holds at. "Triage-score
robust" likewise across the 4 scores. No new headline from (a).

### H10 — Patient-clustered uncertainty (score caches + metadata)
Every AUROC cell (Table 2/3 cells and all others) gets DeLong, patch-iid bootstrap and
patient-cluster bootstrap 95% CIs. The retrospective DenseNet-vs-MobileNet coverage gap
(0.826) gets a patient-cluster CI.
- If the cluster CI of the gap excludes 0 → keep the sentence, replace the patch CI
  [0.817, 0.837] by the cluster CI. If it includes 0 → the sentence must say the gap is not
  resolved at the patient level (9 clusters).
- The DenseNet MSP jump per seed: cluster CI of (DenseNet MSP − ResNet mean); "robust" needs
  the lower bound ≥ 0 in ≥ 4/5 seeds (reported next to the 5/5 point count).

### H11 — Leakage (metadata; GPU re-extract for (c))
(a) Patients of hospital 2 vs hospitals 0/3/4: overlap must be 0. (b) slides: overlap must
be 0. Either > 0 → stop, everything OOD-related is invalid until fixed.
(c) id_val shares slides with train (WILDS design). Slide-excluded kNN and slide-disjoint
2-fold Maha/kNN (seeds 42; all 8 archs). Bar: if any arch loses > 0.02 AUROC for kNN or
Maha under the slide-disjoint protocol → the paper reports slide-disjoint feature AUROCs in
the appendix and re-checks H5(a) with them; if every |Δ| ≤ 0.02 → one sentence "ID-side slide
sharing changes feature AUROCs by ≤ 0.02".

### H12 — Cross-seed W for skin and MIDOG (CSV only)
Same as H1/H4 per domain. Bars identical to H1/H4. These CSVs already exist (8 archs x
seeds 43–46); the paper's Limitation 8 ("not computed for skin or MIDOG") is closed either way.

### H13 — ViT-B/16 over seeds (opt-in GPU, ~27 GPU-h)
Locked prediction (MSP ≥ 0.6947) re-tested on seeds 43–46 with the same recipe.
- ≥ 4/5 seeds (incl. 42) ≥ 0.6947 → "seed-robust locked prediction". 3/5 → "seed-dependent".
  ≤ 2/5 → the ViT paragraph must say the seed-42 hit did not replicate.

### H14 — Multiple testing
Every p-value lands in a `pvalues_family.csv`; `multiple_testing.py` applies Holm (FWER) and
BH within each declared family (W_vs_chance, arch_vs_seed, msp_jump_gt0, msp_jump_gt0.15,
delong_best_feature_vs_best_other, delong_maha_vs_knn) per domain, and globally. Only
within-family Holm-significant results may be called "significant". The MSP-jump "≥0.15"
counts stay descriptive; inferential language needs the Holm-adjusted t-test (family
msp_jump_gt0.15).
Sensitivity (added 2026-10-03 during the reviewer audit, before any result was read): `msp_jump_tests.csv` also reports each arch's mean MSP minus the pooled 10 ResNet cells and a one-sided Welch p (`p_welch_vs_pooled_resnet_sens`). It is outside every family and is descriptive only: if it disagrees with the per-seed-baseline test, the paper says the jump count depends on the baseline definition.

### H15 — Matched-recipe results over seeds (opt-in GPU, ~65–70 GPU-h)  [added 2026-10-03, before any result]
M1 (ConvNeXt, MobileNetV3, RegNetY, EffV2-S@224 trained with ResNet-Adam) and M2 (ResNet-18/50
with ConvNeXt-AdamW) re-run on seeds 43–46 with the seed-42 commands. Jump = MSP − official
ResNet mean of the same seed; bar 0.15. Seed-42 values must reproduce the paper (REPRO).
- M1, per arch: ≥ 4/5 seeds jump → "recipe mismatch does not explain the jump" stays, with the
  count; 3/5 → "seed-dependent"; ≤ 2/5 → the M1 paragraph must say the seed-42 hit did not
  replicate for that arch.
- M2, ResNet-18: ≥ 4/5 seeds jump → "ResNet can jump under AdamW" becomes seed-robust; ≤ 2/5 →
  the sentence is downgraded to "one seed".

### H16 — Transfer regret over all pairs (CSV only)  [added 2026-10-03, before any result]
regret(src→tgt) = best AUROC on tgt within a score family − AUROC on tgt of the score that was
best on src. All ordered arch pairs at every seed (cross_arch) and all ordered seed pairs per
arch (cross_seed); families nonfeature (MSP, Energy, ELogitNorm, ViM, ReAct), feature (Maha,
kNN), all 7. REPRO: ResNet-50→ConvNeXt seed 42 nonfeature = 0.185.
- The 0.185 sentence stays only together with its percentile among the seed-42 cross-arch
  nonfeature regrets and the median; if its percentile is ≥ 0.95, the text must call it "the
  worst case of 56 pairs", not a typical cost.
- "Feature-space transfer regret ≤ 0.05" is allowed for all seeds only if the max feature-family
  cross_arch regret over all 5 seeds is ≤ 0.05; otherwise report the fraction > 0.05.
- Cross-seed vs cross-arch nonfeature median regret is reported side by side (no bar); it is
  the regret version of H1.

### H17 — W with near-ties (CSV only)  [added 2026-10-03, before any result]
AUROCs closer than eps ∈ {0.005, 0.01, 0.02} share a mid-rank (single-linkage chain), then the
repo's tie-corrected W. eps = 0 must reproduce 0.694 (REPRO). Read at eps = 0.01:
D = mean cross-seed W (8 archs) − mean cross-arch W (5 seeds).
- |D| ≤ 0.10 → "seed reorders about as much as architecture" is tie-robust (one sentence).
- D > 0.10 → add: "treating AUROC differences below 0.01 as ties, seeds agree more than
  architectures (D = …); much of the seed disagreement is among near-tied scores".
- D < −0.10 → add the mirror sentence for architecture.
(The k = 5 vs 8 rater mismatch remains here; H1 is the matched comparison.)

---

## Order
1. Commit this file + `scripts/rigor/` + `run_rigor_pack.sh` + docs (no results).
2. `bash run_rigor_pack.sh` (CPU stages; GPU stages only with explicit flags).
3. Read results strictly against H1–H17; write `outputs/reports/rigor_pack/VERDICTS.md`
   with one line per H (verdict row + number), then propose manuscript edits.
4. Only after 3, edit the manuscript (separate commit).

---

## Dated notes

**2026-10-03, disclosure (before any intended read of results).** While building the smoke
test, one run of `smoke_test.py` also ran `w_from_csv.py` (200 permutations, not 10k) on the
real tracked CSVs and then `multiple_testing.py` over the shared report tree, which printed
aggregate "number of Holm/BH-significant tests per family" counts for the CSV-only families
(W_vs_chance, arch_vs_seed, msp_jump_*). No W value, p-value, verdict row or table was read.
The bars above were all written before that run and were not changed after it. The smoke test
now writes real-CSV output to a separate, non-aggregated folder. Later additions, all made during the reviewer audit (`docs/REVIEWER_AUDIT.md`) and before
any of their numbers existed or were read: the descriptive, family-free pooled-baseline
sensitivity under H14, and H15/H16/H17. Only their REPRO lines (seed-42 values already in the
paper) were checked; all PASS.
