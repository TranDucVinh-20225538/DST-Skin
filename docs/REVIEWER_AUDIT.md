# Reviewer audit: MIDL 2027 paper (tough-reviewer pass)

Paper: `manuscript/midl2027/main.tex` ("Detector Ranking is Not Architecture-Invariant under
Medical Covariate Shift"), as of branch `main` on 2026-10-03. The older CVPR draft
(`manuscript/cvpr2027/`, deleted in commit `4ae2123`, last version at `4ae2123^`) is covered
in section E.

How to read this file:
- Each item says what a strict MIDL reviewer would write, why the point is fair (with file and
  line evidence), and what to do about it.
- **Fix** is `PACK Hn` (a computation in the rigor pack; bars in
  `decisions/decision_precommit_rigor_pack.md`), `TEXT` (a wording change, proposed below), or
  `OUT` (needs new data or a new study, so it can only be stated as a limitation).
- Severity: **M** = major (could cause a reject or a strong "weak reject"), **m** = minor,
  **c** = cosmetic.
- No manuscript file was edited. Every proposed edit is written out in section C (edits you
  can make now, whatever the results) and section D (edits that depend on a rigor-pack
  verdict; pick the version that matches the verdict).
- No rigor-pack result was read while writing this file. Numbers quoted here are either
  already in the paper or are dataset facts (patient and slide counts from the tracked
  `outputs/reports/camelyon_coverage_val_split_patients.csv`).

---

## A. Summary table

| # | Weakness (one line) | Sev | Fix | Pack stage |
|---|---|---|---|---|
| A1 | Cross-seed W (k=5 raters) is compared with cross-arch W (k=8); E[W] depends on k, so the comparison is biased | M | PACK H1 | csv |
| A2 | "Seed reorders about as much as architecture" is eyeballed; there is no test of arch vs seed | M | PACK H4 (+H17 ties) | csv, ties |
| A3 | Only 4 anchors with 5 seeds; there is no power statement, so a null result is uninterpretable | M | PACK H4 power table + TEXT | power (done) |
| A4 | W has no sampling uncertainty; leave-one-backbone is not a CI (the paper says so, but gives no CI) | M | PACK H3 | persample |
| A5 | No CI on any AUROC in Tables 2 and 3 | M | PACK H10 (DeLong + iid + patient-cluster bootstrap, every cell) | persample |
| A6 | Patches are not independent: the patch bootstrap CI [0.817, 0.837] has 9 OOD patients behind it | M | PACK H10 (cluster bootstrap) + TEXT | persample |
| A7 | Many tests, no multiplicity control (counts like 5/5, 4/5, 4/4 used inferentially) | M | PACK H14 (Holm/BH per family) | mtest |
| A8 | The jump is defined against a noisy per-seed ResNet baseline (0.520–0.667); the "≥0.15 in k/5 seeds" counts are dichotomised | m | PACK H14 t-tests + pooled-baseline sensitivity; H10 cluster CI of the jump | csv, persample |
| A9 | Seed is nested in arch (seed 43 of DenseNet has nothing to do with seed 43 of ResNet) | m | PACK H4b nested ANOVA (+crossed sensitivity) | csv |
| A10 | "Rank 1 on every backbone and every seed tested" rests on 20 anchor cells + 8 seed-42 cells; Maha vs kNN rank 1/2 could be a tie | M | PACK H5 (all 40 cells; DeLong best-feature vs best-other; Maha vs kNN) | csv, persample |
| A11 | AUROC only; ranking conclusions may be AUROC-specific | M | PACK H7 (FPR@95TPR, AUPR-In/Out) | csv, persample |
| A12 | Near-tied scores (Energy/ELogitNorm/ReAct, Maha/kNN) flip rank by noise and deflate W | m | PACK H17 | ties |
| A13 | The 0.185 "transfer cost" is one hand-picked pair (R50→ConvNeXt, seed 42) | m | PACK H16 (all pairs, all seeds) | regret |
| B1 | The +0.215 held-out gain comes from one split of 9 patients | M | PACK H6 (252/168/510 splits, LOPO, DROP1; 3 seed modes) | splits |
| B2 | "Stratified on the tumor label" is not true in effect: tumor fraction 0.339 (val) vs 0.651 (test); one patient is 73% of the test half | M | TEXT + PACK H6 | – |
| B3 | "Frozen before any model was scored" cannot be verified: the split file and its results are in the same commit (`18cbd2c`, 2026-10-03 10:35 +07), long after the scores existed | M | TEXT (+H6 makes the frozen split one of 168) | – |
| B4 | Coverage only with MSP and only at 10% risk; why MSP when the paper recommends feature scores? | M | PACK H9 (5/10/20% × MSP/Energy/Maha/kNN, with CIs; H6 repeated for each) | persample, splits |
| B5 | "AUROC rank ≠ coverage rank" is shown for single tables, not quantified | m | PACK H6 Kendall tau (val AUROC vs test coverage; val coverage vs test coverage) over splits | splits |
| B6 | The MIDOG "sign flip" rests on 3 backbones with MSP AUROC 0.438–0.512 (at or below chance) and a coverage CI [0.004, 0.369] | M | TEXT; `--skin-midog` gives per-cell MIDOG CIs | skin-midog (opt-in) |
| B7 | Coverage@risk on hospital-2 is selective classification on the target domain; AUROC is ID-vs-OOD separability. Disagreement is expected and partly driven by hospital-2 accuracy | M | TEXT + PACK H8b (accuracy per run) | persample |
| B8 | The Camelyon n=4 utility table mixes EffB3@224 (not a zoo member) with the zoo | m | TEXT | – |
| C1 | ID test set = WILDS `id_val`, which shares slides (and patients) with train → near-duplicate ID patches favour Maha/kNN | M | PACK H11c (slide-excluded kNN, slide-disjoint 2-fold Maha/kNN) | extract + leak (GPU, opt-in) |
| C2 | Is any hospital-2 patient also in ID train/test? | m | PACK H11a,b | patients |
| C3 | Maha/kNN reference set = train features extracted *with train augmentation, shuffled* | m | TEXT + optional re-extract with `--train-transform eval` | extract (opt-in) |
| C4 | Checkpoint "best" is selected on `id_val`, which is also the ID side of every AUROC | m | TEXT (disclose); not cheaply fixable | – |
| C5 | Is the MSP jump just accuracy (DenseNet more accurate on hospital 2)? | M | PACK H8b (OLS + run bootstrap, matched pairs) | persample |
| C6 | The temperature experiment is vacuous: binary MSP AUROC is invariant to T by construction | M | TEXT now; PACK H8a verifies numerically | persample |
| C7 | M1/M2 recipe controls are seed 42 only | m | PACK H15 (GPU opt-in, ~65–70 GPU-h) | recipe-seeds |
| C8 | ViT "locked prediction" is one seed, and the threshold was easy (all 6 non-ResNets at seed 42 cleared it) | M | PACK H13 (GPU opt-in, ~27 GPU-h) + TEXT | vit-seeds |
| C9 | EffV2-S at 224 instead of native 384 looks like a post-hoc choice | m | TEXT (say when and why) | – |
| C10 | Skin/MIDOG have no cross-seed control (Limitation 8) although the CSVs exist | m | PACK H12 | csv |
| C11 | Seed-42 ViM/ReAct/ELogitNorm were "filled from features" (see `scripts/bootstrap_invariance_spread.py`), so not every cell went through one pipeline | m | PACK: score caches recompute all 7 scores uniformly; REPRO vs CSV ≤ 2e-3 | scores |
| D1 | Title and abstract claim more than the results support (seed reorders as much as arch) | M | TEXT | – |
| D2 | Discussion says "architecture is the dominant axis", which contradicts the cross-seed result | M | TEXT | – |
| D3 | "Eight CNN families": ResNet-18/50 and EffB3/EffV2-S are the same families | m | TEXT | – |
| D4 | Datko et al. citation carries a footnote saying it could not be verified | M | TEXT (verify or remove before submission) | – |
| D5 | "Pre-registered" / "precommitted" without verifiable timestamps | m | TEXT (cite anonymised commit hashes) | – |
| D6 | "May be biomedical-specific" from one non-medical dataset (iWildCam) | m | TEXT | – |
| D7 | MIDOG W: paper says n=3 W=0.857; released files `architecture_invariance_kendall_w.csv` / `kendall_w_perm.txt` show MIDOG n=8 W=0.866 | m | TEXT/repo (label the n=8 file as the in-house zoo) | – |
| D8 | Training details (epochs, augmentations, selection rule, recipes per arch) are not in the paper | m | TEXT (appendix table) | – |
| D9 | Anonymity: the code repo is public, under the author's name, with the same title; old scripts contain an absolute path with a username | M | TEXT/repo (anonymous mirror for review) | – |
| D10 | Unclear sentence: "MSP AUROC span 0.515–0.883 matches the Maha−MSP gap on ResNet-50" | c | TEXT | – |
| OUT1 | 9 OOD patients, one OOD hospital: everything patient-level is low-n | M | OUT (limitation; H10/H6 quantify it) | – |
| OUT2 | No mechanism for the DenseNet MSP jump | m | OUT | – |
| OUT3 | No reader study / clinical utility | m | OUT (already a limitation) | – |

---

## B. Details, item by item

### A1 — Rater-count bias in the W comparison (M) → PACK H1
*Reviewer:* "Cross-seed W uses 5 raters, cross-architecture W uses 8. Kendall's W is not
comparable across k: under no agreement, E[W] = 1/k, and with partial agreement it also falls
with k. The headline comparison (0.600–0.931 vs 0.694) is apples to oranges."
*Evidence:* `main.tex` l.233–235. The a priori simulation in the precommit (no data) gives,
for the same true agreement, E[W] ≈ 0.62 for 4 raters vs ≈ 0.56 for 8.
*Pack:* `w_from_csv.py` computes cross-seed W for **all 8** archs (k=5) and the full
distribution of cross-arch W over all C(8,5)=56 five-arch subsets at each seed (k=5, the same
k). Verdict rule: H1.

### A2 — No test of architecture vs seed (M) → PACK H4, H17
*Reviewer:* "The key conclusion ('cannot be attributed to architecture alone') is a visual
comparison of ranges. Run a test: does knowing the architecture explain ranking variance
beyond seed?"
*Pack:* arch-label permutation test (re-assign the 40 runs to 8 groups of 5, 10k
permutations) and a nested ANOVA (method × arch vs method × seed(arch)) on ranks and on
AUROC, with permutation p and a variance-component share. Run on the 4 anchors and on all 8
archs (the CSVs for all 8 × 5 already exist). H17 adds a near-tie sensitivity.

### A3 — Power for 4 anchors (M) → precommit power table + TEXT
*Reviewer:* "With 4 architectures × 5 seeds, what effect could you have detected at all?"
*Pack:* `power_w.py` (done, no data; table in precommit H4): 4 archs reach 80% power only if
architecture explains ≳35% of ranking-relevant variance; 8 archs reach it at ≳20%.
*Text (now):* see C-6.

### A4 — No CI for W (M) → PACK H3
*Reviewer:* "W = 0.694 has no uncertainty. Leave-one-backbone is not a CI (you say so), so
what is the CI?"
*Pack:* `persample.py` gives paired bootstrap CIs of W (patch-iid and patient-cluster), of the
cross-seed W, and of D = mean cross-seed W − mean k=5 cross-arch W.

### A5, A6 — No AUROC CIs; patch bootstrap ignores patients (M) → PACK H10
*Reviewer:* "Tables 2 and 3 report 3-decimal AUROCs without any interval. The only CI
(coverage gap [0.817, 0.837]) treats 85k patches as independent, but they come from 10 slides
of 9 patients. The effective sample size is closer to 9 than to 85k."
*Evidence:* `main.tex` l.315, l.385 (the paper admits the clustering issue).
*Pack:* for **every** cell (8 archs × 5 seeds × 7 scores; this includes every Table 2/3
cell): DeLong CI, patch-iid bootstrap CI, patient-cluster bootstrap CI. Cluster CI of the
DenseNet−MobileNet coverage gap, and of the DenseNet jump per seed.
*Note:* with 9 OOD clusters a cluster bootstrap is itself rough (few distinct resamples,
skewed). The pack reports it as a percentile interval and the text should say "9 patients".

### A7 — Multiple testing (M) → PACK H14
*Reviewer:* "The paper runs many precommitted checks (jumps per arch, W vs chance, DeLong
comparisons, M1/M2, temperature, gates). None is corrected."
*Pack:* every p-value goes to `pvalues_family.csv`; `multiple_testing.py` applies Holm and BH
within declared families (and globally). Only Holm-significant results may be called
significant.

### A8 — Jump baseline and dichotomised counts (m) → PACK H14, H10
*Reviewer:* "The jump threshold is relative to that seed's ResNet mean, which itself moves
from 0.520 to 0.667. 'DenseNet 5/5, EffB3 2/5' are counts of a dichotomised noisy quantity."
*Pack:* one-sided t-tests of jump > 0 and jump > 0.15 per arch (Holm); a family-free
sensitivity against the pooled 10 ResNet cells (Welch); a cluster CI of the jump per seed (H10).

### A9 — Seed nested in arch (m) → PACK H4b
Nested ANOVA treats seed as nested; a crossed-seed version is reported as a sensitivity only.

### A10 — "Rank 1 on every backbone and every seed" (M) → PACK H5
*Reviewer:* "Abstract l.29 says 'every backbone and every seed tested', but the text (l.236)
checks 20 anchor × seed cells plus the 8 seed-42 cells. Also, Maha vs kNN at rank 1 vs 2 may be
a tie."
*Pack:* count over all 40 Camelyon cells (and 40 skin, 40 MIDOG); paired DeLong, best feature
vs best non-feature score, per cell (Holm over 40); Maha vs kNN differences.

### A11 — AUROC-only (M) → PACK H7
FPR@95TPR from the CSVs (W, cross-seed W, rank-1 count) and AUPR-In/Out from the score
caches. Bar: |ΔW| ≤ 0.10 and rank-1 ≥ 36/40.

### A12 — Near-ties (m) → PACK H17
W recomputed with AUROCs closer than 0.005/0.01/0.02 treated as ties.

### A13 — Hand-picked transfer cost (m) → PACK H16
*Reviewer:* "0.185 is one pair. Is it typical or the worst case?"
*Pack:* `transfer_regret.py` gives the regret for all 56 ordered arch pairs at each seed, and
all seed pairs within each arch, for the non-feature, feature, and all-7 families. It also gives
the percentile of 0.185. REPRO of 0.185 passes.

### B1 — One split (M) → PACK H6
*Reviewer:* "0.539 → 0.754 is one split of 9 patients; you admit it (Limitation 7). Run all
splits."
*Pack:* `coverage_splits.py`: all 252 splits with 4–5 validation patients (primary), all 168
3/6 splits (the frozen one is among them, so its percentile is reported), all 510 proper
splits, leave-one-patient-out, frozen-minus-one. Per seed, seed-averaged, cross-seed. Four
triage scores × three risk levels. Kendall tau between val AUROC rank and test coverage rank.

### B2 — The split is not tumor-balanced (M) → TEXT
*Facts* (tracked `camelyon_coverage_val_split_patients.csv`): hospital 2 has **9 patients,
10 slides** (slides 20–29; patient 046 has two), 85,054 patches, tumor fraction 0.500 overall.
The frozen split: 6 val patients, 41,217 patches (48.5%), tumor fraction **0.339**; 3 test
patients (040, 046, 051), 43,837 patches, tumor fraction **0.651**. Patient 051 (slide 28)
alone has 31,878 patches (85% tumor): 37.5% of hospital 2 and ≈73% of the test half.
"Stratified on the tumor label, grouped by patient" (l.319) therefore balanced the patch
*count*, not the tumor fraction. A reviewer will compute this from WILDS metadata in minutes.
*Text:* see C-3.

### B3 — "Frozen before any model was scored" is not verifiable (M) → TEXT
*Evidence:* `git log` shows `camelyon_coverage_val_split_patients.csv`, the script, and the
coverage results were all added in one commit, `18cbd2c` (2026-10-03 10:35 +07). The score
CSVs and feature caches are from mid-September. The split rule is deterministic (seed 42), but
git cannot prove it was fixed before the scores were looked at.
*Text:* see C-3 (claim only what can be shown). H6 makes the point moot by reporting every
split.

### B4 — MSP-only, 10%-only coverage (M) → PACK H9
*Reviewer:* "You recommend feature-space scores for detection, but triage with MSP. Does the
conclusion hold with Maha/kNN as the triage score, and at 5% / 20% risk?"
*Pack:* coverage@risk 5/10/20% × MSP/Energy/Maha/kNN for every cell with CIs (H9a); the
whole H6 selection experiment for every (score, risk) (H9b).

### B5 — AUROC vs coverage rank, quantified (m) → PACK H6
Kendall tau(val AUROC, test coverage) and tau(val coverage, test coverage) across splits.

### B6 — MIDOG sign flip on 3 near-chance detectors (M) → TEXT (+ optional CIs)
*Reviewer:* "On MIDOG the three MSP AUROCs are 0.512 / 0.486 / 0.438. Two are at or below
chance. Ranking near-chance detectors and then calling the disagreement with coverage a
'sign flip' over-reads 3 points. The R50 coverage CI is [0.004, 0.369]."
*Pack:* `--skin-midog` gives per-cell MIDOG coverage and AUROC CIs (patch-level; MIDOG has no
patient grouping in the pack). There is no computation that turns n=3 into a law.
*Text:* see C-8.

### B7 — Coverage@risk measures something else (M) → TEXT + PACK H8b
*Reviewer:* "Coverage@risk on hospital-2 is selective classification inside the target domain.
It depends mostly on the classifier's hospital-2 accuracy. AUROC is ID-vs-OOD separability.
That they disagree is expected (you say so at l.151), so 'AUROC is not a proxy for coverage' is
weak as a contribution unless you show the disagreement is not just accuracy."
*Pack:* `persample.py` writes hospital-2 and id_val accuracy for every run
(`accuracy_by_run.csv`). This lets the text say how much of the coverage ranking is accuracy.
*Text:* see C-7.

### B8 — Mixed n=4 utility table (m) → TEXT
l.296 uses EffB3@300, EffB3@224, R18, R50; EffB3@224 is not an official cell. Say it is the
locked pre-zoo utility table, or replace it with the 8-arch numbers (H9 tables give them).

### C1 — ID-side slide sharing (M) → PACK H11c (GPU, opt-in)
*Reviewer:* "WILDS `id_val` is drawn from the same slides as train. Maha/kNN fitted on train
features see near-duplicate patches of the ID test patches, which inflates their ID-vs-OOD
AUROC. The logit scores do not get this advantage. Your main stable finding (feature scores
rank 1) could be partly this."
*Why the pack needs GPU:* the cached train features were saved shuffled and without indices,
so no slide ID can be attached to them. `extract_features_indexed.py` re-extracts with indices
and slide IDs (≈13 min/cell on A100); `leakfree_knn.py` then runs slide-excluded kNN and
slide-disjoint 2-fold Maha/kNN. Bar: any |ΔAUROC| > 0.02 → appendix table and re-check of H5.
*Text (now, whatever the result):* see C-9.

### C2 — Patient overlap across hospitals (m) → PACK H11a,b
`hospital2_patients.py` checks that no hospital-2 patient or slide appears in hospitals
0/3/4. Expected 0 by WILDS design, but a reviewer may ask, and it costs nothing.

### C3 — Augmented reference features (m) → TEXT (+ optional eval-transform re-extract)
Train features were extracted with the training transform (`RandomResizedCrop(scale 0.7–1.0)`
+ horizontal flip, `src/datasets/camelyon_ood.py`), not the eval transform (resize + center crop). Maha/kNN therefore use an augmented reference set. This is unusual and
should be stated. `extract_features_indexed.py --train-transform eval` writes a deterministic
version (`rigor_indexed_evaltf/`). It is not wired into `run_rigor_pack.sh` by default; run it
by hand if a reviewer insists (same cost as `--extract`).

### C4 — Model selection on the ID test set (m) → TEXT
`*_best.pth` is chosen on `id_val`, which is also the ID side of every AUROC. With id_val
accuracy ≈ 99.5% the effect is likely small, but it should be disclosed (C-9).

### C5 — Accuracy confound for the MSP jump (M) → PACK H8b
OLS over the 40 runs: MSP AUROC ~ arch + hospital-2 accuracy (+ id_val accuracy), run
bootstrap CI of the adjusted DenseNet jump, plus accuracy-matched pairs.

### C6 — Temperature check is vacuous (M) → TEXT now (PACK H8a verifies)
*Reviewer:* "For a 2-class softmax, MSP = σ(|z₁ − z₀| / T), a monotone function of the logit
margin for every T > 0. AUROC depends only on the order. So 'gap 0.308 → 0.308 after T*' is an
identity, not evidence." The correlation of T* with the jump (r = 0.268, p = 0.52, n = 8) is
also underpowered.
*Text:* see C-5.

### C7 — M1/M2 seed 42 only (m) → PACK H15 (GPU, opt-in)
`--recipe-seeds` re-runs the 6 matched-recipe runs on seeds 43–46 (24 runs, ≈65–70 GPU-h).
`recipe_seeds.py` reproduces the six seed-42 MSPs in the paper (REPRO PASS).

### C8 — ViT: one seed, easy threshold (M) → PACK H13 + TEXT
*Reviewer:* "All six non-ResNet CNNs cleared 0.6947 at seed 42, so predicting that a seventh
non-ResNet clears it is a low bar. And it is one seed."
*Pack:* `--vit-seeds` (4 runs, ≈27 GPU-h); `vit_seeds.py` applies the H13 rule.
*Text:* see C-10.

### C9 — EffV2-S resolution choice (m) → TEXT
Say whether 224 was chosen before or after the 384 result; the paper reports both, which helps.

### C10 — Skin/MIDOG cross-seed (m) → PACK H12
The CSVs for 8 archs × seeds 43–46 already exist for skin and MIDOG. `w_from_csv.py` computes
everything for them too. Limitation 8 can be closed.

### C11 — Mixed computation pipelines (m) → PACK (score caches)
`build_score_cache.py` recomputes all 7 scores from the feature caches with the repo's
`OODScorer` and checks them against the CSVs (≤ 2e-3). Per-sample results (H3, H7, H10) all use
this uniform pipeline.

### D1–D10, OUT1–3
See the proposed text in sections C and D.

---

## C. Proposed text edits you can make now (independent of rigor-pack results)

Line numbers refer to `manuscript/midl2027/main.tex` on 2026-10-03.

**C-1 Title (l.16), D1.** The current title states the hypothesis the paper ends up *not*
supporting in its strong form. Options:
- "Detector Rankings Move with Backbone and Seed under Medical Covariate Shift; Feature-Space Scores Do Not"
- "How Stable Are OOD Detector Rankings across Backbones and Seeds under Medical Covariate Shift?"
(Final choice after H1/H4: if H4 detects an architecture effect, the current title becomes
defensible again, with a quantified share.)

**C-2 Abstract l.29.** Replace "Mahalanobis or $k$NN is rank~1 on every backbone and every
seed tested" with the H5 wording (section D). Until then, the honest version is: "Mahalanobis
or $k$NN is rank~1 on all eight backbones at seed 42 and on all 20 anchor×seed runs."

**C-3 Held-out split (l.319–323), B2/B3.** Replace
"we split the nine hospital-2 patients into two halves (stratified on the tumor label, grouped
by patient, seed~42, frozen before any model was scored): six patients ($41{,}217$ patches) for
selection, three ($43{,}837$) for evaluation."
with
"we split the nine hospital-2 patients (ten slides) by patient into a selection set of six
patients ($41{,}217$ patches, $34\%$ tumor) and an evaluation set of three ($43{,}837$ patches,
$65\%$ tumor), using a fixed rule (seed~42) that was not tuned. The two sets are not
tumor-balanced; one evaluation patient contributes $31{,}878$ patches ($73\%$ of the evaluation
set, $85\%$ tumor)."
Drop "frozen before any model was scored" unless you can point to a commit that predates the
scores. (H6 then reports this split as one of 168 and gives its percentile; see D-6.)

**C-4 Cross-seed paragraph (l.233–235), A1.** Add after the W values: "These two kinds of $W$
are not directly comparable, because the expected $W$ falls with the number of raters ($5$
seeds vs.\ $8$ backbones); \appendixref{app:suppl} compares them at equal rater count." (Then
fill from H1.)

**C-5 Temperature (l.340–343, and l.154), C6.** Replace the paragraph with:
"\textbf{Temperature $T^\ast$.} For a two-class softmax, MSP $=\sigma(|z_1-z_0|/T)$ is a
monotone function of the logit margin for every $T>0$, so MSP AUROC cannot change under
temperature scaling; the unchanged DenseNet${-}$ResNet-18 gap ($0.308$) is this identity, not
an experiment. Temperature can matter only for scores that are not monotone in the margin
(e.g.\ Energy at $T^\ast$; \appendixref{app:suppl})."
Delete "$r{=}0.268$, $p{=}0.52$" or keep it with "$n{=}8$, underpowered". In l.154, remove
"temperature scaling" from the list of tested artefacts, or mark it "(ruled out analytically)".

**C-6 Power (Discussion or Limitation 8), A3.** Add: "With four anchors and five seeds, a
simulation calibrated to the observed seed noise gives ${\ge}80\%$ power only if architecture
explains roughly a third or more of the ranking variance (eight backbones: about a fifth); a
null at four anchors is therefore not evidence of no architecture effect."

**C-7 Coverage framing (l.31, l.81–83, l.149–151), B7.** Replace "AUROC is not a proxy for
deployment coverage" by "ID-vs-OOD detection AUROC does not predict target-domain selective
classification (coverage@risk)". Add one sentence after l.151: "Coverage@risk on hospital~2 is
bounded by each backbone's hospital-2 accuracy (\appendixref{app:suppl} lists it per run), so
part of the coverage ranking is a classifier-accuracy ranking."

**C-8 MIDOG sign flip (l.31, l.82, l.299–302, l.308–309, l.314), B6.** Soften from a finding
to an illustration: "On MIDOG (in-house $n{=}3$), all three MSP AUROCs are near chance
($0.44$--$0.51$) and the AUROC-highest backbone is the coverage-lowest (ResNet-50 coverage
$0.166$, $95\%$ CI $[0.004, 0.369]$); the direction of the disagreement differs from Camelyon,
but three near-chance detectors cannot establish a sign rule." Remove "flips" from the
abstract, or keep it with "on three backbones".

**C-9 Setup: ID side (after l.127), C1/C3/C4.** Add: "The ID test set is WILDS
\texttt{id\_val}, which is drawn from the same slides as the training patches; model selection
(best checkpoint) also uses \texttt{id\_val}. Mahalanobis and $k$NN are fit on training-set
features extracted with the training augmentation." (Then add the H11c result, see D-10.)

**C-10 ViT (l.285–289, l.32, l.84–85, l.404), C8.** Add: "The threshold was easy to meet: all
six non-ResNet CNNs exceeded it at seed~42. The prediction is therefore a consistency check, not
a strong test." Keep "a pre-registered ViT hit shows the Camelyon MSP jump is not a CNN-zoo
accident" (l.404) only if H13 passes; otherwise see D-12.

**C-11 Discussion l.364, D2.** "architecture is the dominant axis, not a clean one" contradicts
l.362. Replace with "architecture moves the MSP level (DenseNet, $5/5$ seeds), but it is not a
clean variable: optimizer and seed also move it." (Revisit after H4/H15.)

**C-12 "Eight CNN families" (l.27, l.135), D3.** "eight CNNs from six families".

**C-13 Datko et al. (l.58, l.102, l.116), D4.** A footnote saying a citation could not be
verified is a red flag for any reviewer. Before submission: find the DOI and read the paper,
or remove the citation and its row in Table 1. Do not submit with the footnote.

**C-14 "Pre-registered" (l.32, l.68, l.285, l.334, l.404), D5.** Add one sentence in Setup:
"Each precommitted test was written to a dated file in the (anonymised) code repository before
it was run; commit hashes are listed in the supplement." Then list them. Without this, use
"locked in advance" only where git shows it.

**C-15 iWildCam (l.376), D6.** "may be biomedical-specific" → "is absent on this one
non-medical shift; we cannot say whether it is specific to medical data."

**C-16 MIDOG W provenance, D7.** In the paper nothing changes (0.857, n=3 is the official
value). In the repo, label `architecture_invariance_kendall_w.csv` / `kendall_w_perm.txt`
MIDOG rows "in-house n=8 zoo (0.866), not official W", or a reviewer reading the released code
will think the paper misreports W.

**C-17 Training details, D8.** Appendix table: per architecture, optimizer, lr, wd, batch,
epochs (10), input size, augmentation, selection rule (best id_val accuracy), train-frac 1.0.

**C-18 Unclear sentence (l.75), D10.** "MSP AUROC span $0.515$--$0.883$ matches the
Maha${-}$MSP gap on ResNet-50" → "the MSP AUROC range across backbones ($0.37$) is as large as
the gap between Mahalanobis and MSP on ResNet-50 ($0.38$)."

**C-19 Leave-one-backbone (l.164).** Fine as is. Once H3 exists, add "the patient-cluster
bootstrap $95\%$ CI of $W$ is $[\dots]$".

**C-20 Anonymity, D9 (not a manuscript edit).** The GitHub repo is public, under the author's
name, with the paper's title and README. For double-blind MIDL review, give reviewers an
anonymous mirror (e.g. anonymous.4open.science) and keep the public repo's link out of the
paper. Old `scripts/submit_*.sh` contain `/data2/cmdir/home/<username>/...`. The rigor pack
does not, but the old scripts would need cleaning in the anonymous mirror.

---

## D. Edits that depend on rigor-pack verdicts (fill after `VERDICTS.md`)

Each block gives the replacement for each verdict defined in the precommit.

**D-1 (H1 + H4) Cross-seed paragraph, abstract l.28, Discussion l.362, Conclusion l.400.**
- *Arch effect detected (H4 both tests p<0.05 on 8 archs; H1 "arch separable"):* "At equal
  rater count, seeds of one architecture agree more than architectures do (H1), and an
  arch-label permutation test and a nested ANOVA attribute a share of X of the ranking variance
  to architecture beyond seed (p=…). The cross-architecture $W$ thus reflects a real but partial
  architecture effect."
- *Not detected:* keep the current claim and add C-6 (power) and the H1 percentile: "At equal
  rater count, cross-seed $W$ falls inside the range of cross-architecture $W$ for N/8
  architectures; neither a permutation test nor a nested ANOVA detects an architecture effect
  beyond seed (p=…, power ≥80% only for shares ≳0.2)."
- *Split / mixed:* report both numbers, no headline change.

**D-2 (H2).** Add "(all $W$ above chance, Holm $p<0.05$)" or name the exceptions.

**D-3 (H3).** Put the cluster CI after every quoted $W$ (0.694 [a, b]). If the CI of D
contains 0: "seed and architecture concordance cannot be told apart even before seed noise".

**D-4 (H5).** Abstract l.29 and l.236, l.363:
40/40 → keep "every backbone and every seed"; 36–39 → "in N/40 runs"; <36 → "usually (N/40)"
and the abstract changes. Add "Mahalanobis vs.\ $k$NN differences are significant in M/40
cells; otherwise we call them tied."

**D-5 (H7).** One sentence: "The conclusions are unchanged under FPR@95TPR and AUPR" or name
which conclusion changes under which metric.

**D-6 (H6).** l.31, l.54, l.83, l.320–323, l.369, l.401:
- *Supported:* "Across all 252 patient splits (4–5 validation patients) and five seeds,
  selecting by validation coverage@risk beats selecting by validation AUROC in X% of cases
  (median gain G); the frozen split's $+0.215$ is at the P-th percentile."
- *Helps on average, noisy:* same numbers, plus "but the gain is not consistent across splits
  or seeds."
- *Not supported:* recommendation becomes "AUROC is not a reliable proxy for coverage; at
  eight backbones and nine patients we find no validated selection rule"; report $+0.215$ as one
  draw with its percentile. Remove "raises held-out coverage from 0.539 to 0.754" from the abstract.

**D-7 (H8b).** DenseNet-jump sentences (l.27, l.77, l.220, l.236): add "after adjusting for
hospital-2 accuracy the jump is J [CI]" (or "is not distinguishable from zero").

**D-8 (H9).** State the risk levels and triage scores at which the D-6 verdict holds.

**D-9 (H10).** Replace the patch CI [0.817, 0.837] (l.315) by the patient-cluster CI; replace
Limitation 5's "but ignore patient clustering" with the cluster CIs. Table 2/3: add a CI column
or a sentence "patient-cluster $95\%$ CI half-widths are a–b".

**D-10 (H11c).** If every |Δ| ≤ 0.02: "Excluding training patches from the same slide changes
Mahalanobis/$k$NN AUROC by at most $0.02$." Otherwise: appendix table, and re-state H5 with the
slide-disjoint values.

**D-11 (H12).** Limitation 8: replace "it was not computed for skin or MIDOG" with the skin and
MIDOG cross-seed $W$ values (and their H1/H4 readings).

**D-12 (H13).** ≥4/5: "The locked prediction held on N/5 seeds." 3/5: "seed-dependent (3/5)".
≤2/5: "The seed-42 ViT hit did not replicate (N/5)"; remove the "not a CNN-zoo accident" claim.

**D-13 (H14).** Use "significant" only for Holm-significant results; elsewhere "descriptive".

**D-14 (H15).** M1/M2 paragraphs: add the seed counts; downgrade per the precommit rule.

**D-15 (H16).** l.76, l.227–228, l.368: "Copying ResNet-50's best logit onto ConvNeXt costs
$0.185$ AUROC (the P-th percentile of the 56 seed-42 backbone pairs; median M)." If P ≥ 0.95:
"in the worst of 56 pairs". Keep "feature-space transfer regret ${\le}0.05$" only if H16 allows it.

**D-16 (H17).** One sentence with the near-tie reading.

---

## E. The CVPR draft (`manuscript/cvpr2027/`, removed in `4ae2123`)

The last CVPR version (`git show 4ae2123^:manuscript/cvpr2027/main.tex`) predates the
cross-seed control and the held-out split. It still has the stronger claims: the abstract
headline "logit-based scores reorder substantially with the backbone" with no seed caveat; the
retrospective 0.83 coverage gap as a main result; the same temperature and Datko issues. If
that draft is ever revived or posted (e.g. arXiv), all of sections C/D apply to it. It must
also not be under review at the same time as the MIDL paper (dual-submission rules).

---

## F. What cannot be fixed by computation (state as limitations)

- **OUT1** One OOD hospital, 9 patients, 10 slides. Every patient-level interval is wide by
  construction; H6/H10 quantify this but cannot remove it. A second pathology covariate
  shift (e.g. Camelyon16 or a TCGA site split) would be the real fix.
- **OUT2** No mechanism for the DenseNet MSP jump (H8 rules accuracy in or out; nothing more).
- **OUT3** No reader study; coverage@risk is a proxy for clinical utility.
- **C4** Re-training with a separate selection set would remove the id_val selection issue.
  It costs a full zoo retrain and is not in the pack.

---

## G. Map: precommit hypothesis → script → output

| H | script | output folder (`outputs/reports/rigor_pack/`) |
|---|---|---|
| H1, H2, H4, H5a, H7 (FPR95), H12, H14 jump tests | `w_from_csv.py` | `w_csv/` |
| H3, H5b, H7 (AUPR), H8, H9a, H10 | `persample.py` | `persample_camelyon17/` (+ `_skin_isic_pad`, `_midog` with `--skin-midog`) |
| H6, H9b, tau | `coverage_splits.py` | `coverage_splits/` |
| H11a,b | `hospital2_patients.py` | `hospital2/` |
| H11c | `extract_features_indexed.py` → `leakfree_knn.py` | `leakage/` |
| H13 | `camelyon17_pilot.py` (ViT seeds) → `vit_seeds.py` | `vit_seeds/` |
| H14 | `multiple_testing.py` | `multiple_testing/` |
| H15 | `camelyon17_pilot.py` (M1/M2 seeds) → `recipe_seeds.py` | `recipe_seeds/` |
| H16 | `transfer_regret.py` | `transfer_regret/` |
| H17 | `w_ties.py` | `w_ties/` |
| power note | `power_w.py` | precommit H4 table (`power/` if re-run) |
| tables, Fig 1b | `make_tables.py` | `tables/`, `fig/` |
