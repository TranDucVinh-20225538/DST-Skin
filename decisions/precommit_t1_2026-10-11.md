# Precommit T1 (2026-10-11, Asia/Saigon) - WORK ORDER T1 v2.5

Pre-registration under rule 1 of the order. Committed on branch `t1` before any computation other than item 0 and the inventory.
Nothing has been run.

- Order: `WORK_ORDER_T1_v2_5.md`, sha256 `4dec41ec184ce07c811c9ff9c6434bbcd382228b49a192bde7bb00c5cd5c3096` (full text appended below, verbatim; it is the binding text).
- Bundle: `t1_theory_bundle.tgz`, sha256 `035454a4c13a16e17179aa9a656e89252b2f2de866f5a97a7ab0e2425b5d1f06`; all 27 members verified against `PRECOMMIT_T1_hashes.txt`.
- Input zip: `crossfit-ood_v3.zip`, sha256 `c925933300e29dc76ff599cfc16696901aaa483eabc20b8fce3ad14ae2f9dce1`.
- Structured summary: `results/t1/PRECOMMIT_T1.json` (the order wins where they differ).
- Work lists recorded now (rule 16): G list 0 (G1, 40 units), G list 1 (G2, 24 units).
  Lists that depend on the item I registry or on sampled designs are recorded in precommit addenda before their item launches.
- Defaults fixed now (the order does not define them): `stable_hash(x) = int.from_bytes(sha256(s).digest()[:4], 'big')`,
  s = x for strings, else `json.dumps(x, sort_keys=True, separators=(',', ':'))`; G work-list ID formats as in the JSON.

---

# WORK ORDER T1 (v2.5) — validate or kill the theory program (HPC)

**v2** supersedes `WORK_ORDER_T1.md` (v1, kept untouched). v2 changes only: the Camelyon17 manifest facts and
the Item G slide design (explicit hospitals, counts and a start-of-item verification), the hospital-stratified
designs that depend on them (A2, E3, E4, Item I), the separation of two cluster-size regimes in Item E, and
three wording clarifications (A, C, FINAL). No other threshold, seed or grid changed. See `CHANGELOG_v2.md`.

**v2.1** supersedes `WORK_ORDER_T1_v2.md` (v2, kept untouched). v2.1 makes **only** pre-registration clarifications
to Item G and to the matching rows of the falsification table (H4) and the consolidation (FINAL): (1) the primary
estimand of Δ_G is stated and a report-only slide-equal-weight sensitivity analysis is defined; (2) every dose is
reported in both slide units and patch units; (3) a binding interpretation sentence on what G-ii does and does not
measure; (4) a note that block patch totals set each slide's weight in the patch-level metric. **The slide
allocation algorithm, seeds, doses, every numeric threshold, every criterion and every verdict rule are unchanged,
and no fixed-evaluation-set design is added.** See `CHANGELOG_v2_1.md`.

**v2.2** supersedes `WORK_ORDER_T1_v2_1.md` (v2.1, kept untouched). v2.2 makes **only** three pre-registration
clarifications and a short list of mechanical corrections (typos, cross-references, one missing file name; each listed
in `CHANGELOG_v2_2.md`): (1) row F-19r is moved out of the H4 claim table into a separate report-only block that is
not counted in claim totals or status counts; (2) binding wording on what the Item G bootstrap confidence intervals
do and do not cover (OOD patches from 9 patients; no inference about new OOD patients or hospitals); (3) the sign()
convention of the sign-disagreement flag. **The bootstrap, the slide allocation, seeds, doses, every numeric threshold,
every criterion and every verdict rule are unchanged.**

**v2.3** supersedes `WORK_ORDER_T1_v2_2.md` (v2.2, kept untouched). v2.3 makes **only** four decided edits (each listed in
`CHANGELOG_v2_3.md`): (1) the scorer set of G-i is Mahalanobis and kNN only, and the exact cells that enter the median
relative error of P_Δ in G-i are traced to the predictors registered in items A and B and listed (ViM-with-head is run and
reported but is not part of G-i; no predictor is extended to another scorer); (2) G-ii keeps its threshold of ≥ 5 of 6
model instances, and a G2 reduction that leaves fewer than 5 instances makes G-ii `not evaluable under the pre-registered
criterion` (never a pass; G = PARTIAL if G-i passes); (3) the rule-16 work-list ordering seed namespace is changed to
`SeedSequence([20261010, item_code, 16, 0])` with a published item-code table — **this changes execution order (which
cells run first before a cap), not the statistical design**; the slide-assignment seed `[20261010, 7, 0]` and every other
registered seed are unchanged; (4) when `MANIFEST_OK` is false, H2 is `not run: manifest mismatch (S11)`, H-ii is `not
evaluable` and H cannot be GO. **No numeric threshold, dose, slide-allocation seed, reduction order or other verdict rule
is changed.**

**v2.4** supersedes `WORK_ORDER_T1_v2_3.md` (v2.3, kept untouched). v2.4 is the precommit-ready text. It makes **only**
wording and status / verdict-handling clarifications (each listed in `CHANGELOG_v2_4.md`): (1) the relative error of the
G-i median is defined exactly as e_rel = |P_Δ/Δ_G − 1| over the registered cell set; (2) an empty G-i median makes G-i
`not evaluable` and G = INCONCLUSIVE (never PARTIAL, never GO; INCONCLUSIVE is never a pass); (3) the wording "G2 not run if
A ≠ GO" is clarified (G0 must pass, G1 runs and G-i is evaluated, G-ii is `not run`); (4) the remaining open points of
`CHANGELOG_v2_2.md` and `CHANGELOG_v2_3.md` are resolved by wording only, in particular a general rule 18 that fixes the
verdict whenever a sub-criterion is `not evaluable` / `not run`, the G-ii instance-count cases, the E / H / A0 / H2 / D4 cases,
the work-list sort key and per-list child seeds (rule 16), the file names (A, D5, I) and the H4 / FINAL coverage lines.
**No design element, dose, numeric threshold, reduction order, slide-allocation seed or locked criterion is changed.** The
changelog lists the few places where an ambiguity needed a default (flagged DEFAULT-IF-NOT-DECIDED) and the one execution-order
effect (work-list ordering, rule 16).

**v2.5** supersedes `WORK_ORDER_T1_v2_4.md` (v2.4, kept untouched). v2.5 makes **only one** edit (listed in `CHANGELOG_v2_5.md`):
when A-iii is `not evaluable` under S11 it is no longer counted as "not passed" in a way that can produce A = NO-GO. Rule 18
applies to it like to every other sub-criterion: A-iii `not evaluable` is never a pass (A cannot be GO) and never a failure
(it is never evidence against the hypothesis); A = NO-GO only if NO-GO still holds with A-iii treated as passed, otherwise A =
INCONCLUSIVE (or PARTIAL where the PARTIAL row is met without A-iii). **No design element, dose, numeric threshold, seed,
reduction order, slide-allocation seed or other criterion or verdict rule is changed**, and the other defaults listed in
`CHANGELOG_v2_4.md` (section E, items 2–9) are unchanged.

No venue. Nothing here goes into a manuscript until this order's consolidation report exists.
This order does **not** touch the AISTATS or MIDL submissions, `results/r3/`, or any earlier output.

## What this order is for

`theory_scoping/REPORT.md` (bundle, below) scopes three ideas. Everything in it comes from a
**Gaussian random-effects toy** simulated on a CPU, plus derivations that were not written out as
proofs. **Nothing is proven. Every formula, cap and rule below may fail on real features. The job of
this order is to find out, with decisions fixed before any run, and to stop early when the answer is no.**

| Idea | Claim under test | Items |
|---|---|---|
| 2 — certify the evidence | The inflation Δ of a reported OOD AUROC under unknown group overlap is predicted by a closed form in ρ (group ICC), s = tr S_λ⁻¹, n, N, G, λ (Mahalanobis), by ρ√d (kNN), and is capped for every scorer by an information bound; a label-free fingerprint T detects overlap | 0, I, A, B, C, D, G, H |
| 1 — enough independent evidence | A group-level time-uniform confidence sequence is valid at any ICC for the group-mean estimand θ_group; the sample-level one is not (when cluster size is non-informative; when it is informative the sample-level CS targets a different, size-weighted estimand and is judged against that, separately); groups can be recovered label-free only above a separability threshold | E |
| 3 — blindness classes | For quadratic scorers the sign of AUROC−½ follows the sign of ⟨A,ΔM⟩; the exact blind surface is shifted by an Edgeworth term; blind-but-detectable shifts exist | F |

## Bundle (upload before pasting; Vinh does this)

- `~/handoff/crossfit-ood_v3.zip` from `/Users/cubo/Downloads/handoff_bundle/crossfit-ood_v3.zip`.
  Expected sha256 `c925933300e29dc76ff599cfc16696901aaa483eabc20b8fce3ad14ae2f9dce1` (the value in the
  R3 item-0 report).
- `~/handoff/t1/t1_theory_bundle.tgz`, expected sha256
  `035454a4c13a16e17179aa9a656e89252b2f2de866f5a97a7ab0e2425b5d1f06`. It holds `REPORT.md`, `sims/*.py`
  and the `*_out.txt` of the CPU toy runs. **These are reference outputs for reproduction in A0/C0/F1 and
  for reading only. They are not results of this order and must never be quoted as measurements.**
- If a file is missing or a hash differs: STOP (S1) and report.

---

## Standing rules (apply to every item; no exceptions without a DEVIATIONS entry)

1. **Pre-registration first.** Before any computation other than item 0 and the inventory, write
   `decisions/precommit_t1_<commit-date>.md` and `results/t1/PRECOMMIT_T1.json` containing every
   criterion, threshold, seed and grid in this order, commit and push them, and write the commit hash
   and a `precommit_t1_<date>.DEPOSIT.txt` (same convention as the existing `*.DEPOSIT.txt`).
   Every result file records that hash. A criterion edited after any item has produced output is a
   deviation (rule 12), not an edit.
2. **Report failures as findings. Report and stop: do not tune the method.** If a pre-registered
   criterion fails, write the failure, mark the item NO-GO, and stop that item. No new shrinkage
   choices, alternative mappings, clusterers, thresholds or "one more variant". Only the items'
   "Permitted post-hoc follow-up" lists are allowed afterwards, labelled post-hoc.
3. **Training splits only.** Every statistic (means, covariances, shrinkage, ρ̂, s, thresholds τ,
   correction coefficients, calibrations a+b·z) is fit on the training side of the split being
   evaluated. Calibrations across backbones use leave-one-backbone-out (LOBO). Corrected formulas in
   item C are fit on the training grid only and scored on held-out grids; held-out blocks are never used
   for any choice.
4. **float64** for every covariance, precision, LOGO refit, ICC and statistic
   (`torch.float64`, `torch.backends.cuda.matmul.allow_tf32 = False`,
   `torch.backends.cudnn.allow_tf32 = False`). kNN distance matmuls may use the R3-validated GPU kNN
   implementation (`scripts/r3/recompute_gpu_knn.py`) only after its equality check against float64 CPU
   has been re-run in item 0 on 2 cells (k = 1 and k = 50) and the maximum |ΔΔ| is recorded; otherwise
   use float64.
5. **Chance floors match the metric.** AUROC 0.5; Δ null 0; Spearman null 0 *by pairing permutation*;
   R² floor = constant predictor from the training backbones (R²_oos, can be negative); sign
   accuracy floor = majority-class fraction, **not** 0.5; ARI floor 0; coverage floor = nominal;
   "inversion detected" is judged against "never predict inversion".
6. **Every test run is listed** in `results/t1/<item>/tests.csv` (columns: `test_id, item, quantity,
   n_units, statistic, null, p_raw, p_holm, ci_lo, ci_hi, status{prereg|post-hoc}, pass{true|false|na}`).
   Holm adjustment is applied within each item's family. Go/no-go decisions use only the single
   pre-specified criterion written for them (an intersection–union of the listed sub-tests where stated);
   they never use "the best of several tests".
7. **No multiplicity-free cherry-picking.** No selecting a backbone, scorer, k, λ, seed, dataset, ω or
   subsample after seeing results. Grids are fixed here. Where a grid is truncated for budget it is
   truncated by the pre-declared reduction rule (rule 16), never by choice.
8. **Post-hoc is labelled post-hoc** in the filename (`*_posthoc.csv`), the table header and the report.
9. **Every Δ table carries** `n_groups_fit` (groups actually used to fit the scorer), `K`, `d`, `N_fit`,
   the group-size distribution summary (min/median/max), the ID accuracy of the model, the scorer name
   and preprocessing, and the seed. Scorers are the Track A definitions used in R3 item 1:
   `mahalanobis_l2` = Ledoit–Wolf on L2-normalised features, `knn_mean_cosine` k = 50, ViM
   (residual-only unless the cell has a head file). Do not change them.
10. **Seeds are stated.** Master seed `20261010`. Every random draw uses
    `numpy.random.SeedSequence([20261010, item_code, stable_hash(config), replicate])` (`item_code` from the table in rule 16). Fixed
    resampling seeds: A 101 (bootstrap) / 102 (permutation); B 201 / 202; C 301; D 401; E 501; F 601;
    G 701; H 801 — all `numpy.random.default_rng(seed)`. Bootstrap B = 10,000 (cross-backbone) or
    2,000 (AUROC-level, resampling groups and OOD images as in R3 item 6, `default_rng(2)` convention
    kept for the AUROC-level CI).
11. **Resumable.** Every job writes `<name>.tmp` then renames; skips if a non-empty final file with the
    matching config hash exists (`[ -s out ] && exit 0`, as in `run_item*.sh`); sweeps are sharded into
    ≤ 30-minute shards with a `.done` marker per shard; SLURM jobs use `--requeue`. Re-submitting the
    same command line must be safe at any time.
12. **Deviations.** Anything done differently from this order, or impossible as written, is written
    to `results/t1/<item>/DEVIATIONS.md` (id, what, why, effect on which criterion, who decided = agent)
    and summarised in the final report. A cell that cannot be run is marked `not run: <reason>`; it is
    never silently replaced.
13. **No deletion, no overwrite.** Never delete or modify anything under `results/r3/`, `outputs/`,
    `decisions/` (except adding the T1 precommit), `~/r3work/`, or earlier `results/t1/` output.
    Superseded T1 outputs are moved to `results/t1/<item>/_superseded_<timestamp>/`.
14. **Parallelism.** CPU work: SLURM arrays (one task per cell/shard), BLAS threads = CPUs per task
    (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`), `--exclude=node002` as in the earlier
    scripts. GPU work: one process per GPU, shards assigned `shard_id % n_gpu`. The GPUs are shared with
    other jobs: hold at most 2 GPUs at once unless `squeue` shows the GPU partition idle; never pre-empt
    others.
15. **One REPORT.md per item** under `results/t1/<item>/`: tables, commit hash, precommit hash, the
    one-line verdict read off that item's decision table (GO / PARTIAL / NO-GO / INCONCLUSIVE / `not run`; items 0 and I: PASS / FAIL),
    estimated vs actual GPU-hours, wall clock, and only the caveats this order requires. No other prose.
    **A sub-criterion or verdict component whose status is `not evaluable` or `not run` is shown in every REPORT.md,
    verdict table and falsification table under that status word; it is never shown as, or counted as, a pass (in
    `tests.csv` its `pass` is `na`).** **A verdict of INCONCLUSIVE or `not run` is likewise never counted as a pass anywhere (item tables, H4, FINAL survival rules). Verdicts in the presence of `not evaluable` / `not run` sub-criteria are computed by rule 18.**
16. **Budget discipline.** Each item has a planning estimate and a hard cap (table below). The
    estimates are planning guesses, **not measurements**; item 0 produces measured throughputs and
    `results/t1/0/BUDGET.md` replaces the estimates before items C and G launch. Work lists are
    processed in a fixed pseudo-random order drawn with
    `numpy.random.default_rng(child_j)`, `child_j = SeedSequence([20261010, item_code, 16, 0]).spawn(n_lists)[j]` (list index j and `n_lists` as defined in "Work-list order" below; **namespace changed in v2.3**; v2.2 used
    `[20261010, item, 0]`, which for item G was the same SeedSequence as the registered slide-assignment seed); when the
    cap is reached the run stops and reports the fraction completed. Never exceed a cap by more than 20 %.
    **Item-code table (used by rule 10 and rule 16; published here):**

    | item | 0 | A | B | C | D | E | F | G | H | I |
    |---|---|---|---|---|---|---|---|---|---|---|
    | `item_code` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

    The codes A = 1, B = 2, F = 6, G = 7 are the ones already written in items A, B, F and G; C, D, E, H follow the
    same letter order (3, 4, 5, 8); items 0 (code 0) and I (code 9) draw no random numbers and the codes are only
    reserved. **No other registered seed changes:** the slide-assignment seed `[20261010, 7, 0]`, the held-out draw
    `[20261010, 7, 1, stable_hash(slide)]`, A `[20261010, 1, hash, rep]`, the B MTS stream `[20261010, 2, …]`, F4
    `[20261010, 6, …]` and the fixed resampling seeds of rule 10 are as before. The third entry 16 is reserved for the
    ordering streams, so no ordering stream (for G: the children of `[20261010, 7, 16, 0]`) equals any of the other registered
    streams. **Scope: this seed controls only the order in which an item's work list is executed and therefore which
    cells are done when a cap truncates the run; it does not control the slide allocation, any data draw, the doses, the
    scorers, any threshold or any criterion.**

    **Work-list order (binding; v2.4 wording that fixes the sort key and the seed use; the namespace above is unchanged).** A
    *work list* is the set of units of one item that are executed under one cap and can be enumerated before any computation.
    Every item has exactly one work list (index j = 0), except **E** (regime (a) list j = 0, regime (b) list j = 1; regime (b)
    is processed after regime (a), as stated in E) and **G** (G1 cell list j = 0, G2 fine-tune list j = 1); `n_lists` is 2 for E
    and G and 1 for every other item. A unit is identified by its `cell_id` / `config_id` string, exactly as written in the first
    column of the item's raw output. Procedure: (1) sort the ID strings ascending by Python string comparison (`sorted`,
    code-point order); (2) `perm = numpy.random.default_rng(child_j).permutation(len(list))`; (3) execute in the order
    `sorted_ids[perm[0]], sorted_ids[perm[1]], …`. Each list uses its own spawn child, so no list consumes another list's
    stream. `PRECOMMIT_T1.json` records, for every list that can be enumerated at precommit time, the number of units and the
    sha256 of the newline-joined sorted ID list. The G2 reduction ladder is **not** an ordering: it removes instances in the
    fixed order (seed 44, seed 43, convnext_tiny), never by this permutation.
17. **Wording.** Say "consistent with" / "not consistent with" / "within the tested region". Never say
    "proved", "confirmed", "explains" or "mechanism established". Numbers in this order from the
    bundle are toy-simulation references only.
18. **Verdicts when a sub-criterion is `not evaluable` or `not run` (binding; every item's decision table; v2.4).**
    (a) Such a sub-criterion is **not a pass**: GO is awarded only if every sub-criterion in the GO row was evaluated and
    passed, and a PARTIAL row worded "only X passes", "exactly one passes" or "at least one passes" counts it as not passed.
    (b) It is **never a failure**: a NO-GO row, or a PARTIAL row that needs a sub-criterion to have *failed* (D: "D-ii finds a
    falsifier"), is awarded only if its condition would still hold with every `not evaluable` / `not run` sub-criterion treated
    as passed; a verdict that depends on counting `not evaluable` as failed is not awarded. (c) If no row of the item's table is
    awarded under (a)–(b), the verdict is **INCONCLUSIVE**. (d) Where an item's table states its own consequence for a named
    case (E rows, F-ii / F-iii, G, H, the A `not run` row), that statement is applied as written. A-iii under S11 is **not** an
    exception (v2.5): it follows (a)–(c) through the "A-iii `not evaluable` under S11" clause of the A verdict table.
    (e) INCONCLUSIVE and `not run` are never counted as a pass or as support in any item table, in H4 or in the FINAL survival
    rules. (f) No `not evaluable` / `not run` outcome is converted into a pass or a fail by any report, filter, sensitivity
    analysis or post-hoc analysis. Readings that follow from (a)–(c) for tables without a stated case: **B and C** — one
    criterion passed and the other `not evaluable` → PARTIAL; one failed and the other `not evaluable` → INCONCLUSIVE (not
    NO-GO); both `not evaluable` → INCONCLUSIVE. **D** — D-i or D-iii failed → NO-GO (independent of the others); otherwise D-i
    or D-iii `not evaluable` → INCONCLUSIVE; D-i and D-iii passed with D-ii `not evaluable` → INCONCLUSIVE (PARTIAL needs an
    evaluated falsifier). **F** — F-i and F-iv passed with F-ii / F-iii failed or `not evaluable` → lemma-level (as written);
    F-iv failed → dropped; F-iv `not evaluable`, or F-i failed or `not evaluable` with F-iv not failed → INCONCLUSIVE.
    (For the vocabulary of rule 15 and FINAL, F's "standalone survives" ≙ GO, "lemma-level only" ≙ PARTIAL, "dropped" ≙ NO-GO.)
    **C when S8 stops C** before C-i can be evaluated: C-ii `fail` is reported as a finding and C is INCONCLUSIVE (C-i `not
    evaluable`, rule 18(b)), unless C-i was already evaluated on the completed cells.

---

## Reference: the claims under test (notation; none is proven)

Groups g = 1..G with n_g fit points, N = Σ n_g, features d-dimensional after the scorer's
preprocessing. **Toy model M(G,n,ρ,d):** x_gi = u_g + e_gi, u_g ~ N(0, ρ I_d), e_gi ~ N(0,(1−ρ) I_d)
(total covariance I, ICC ρ). The eval-ID sample is, with probability ω, a new draw from a uniformly
chosen **fit** group ("seen"), otherwise a draw from a fresh group ("unseen"). Higher score = more
anomalous in the toy; Δ = AUROC_ov − AUROC_disjoint (positive = flattering). In the real cells use the
R3 convention (higher score = more ID; Δ = AUROC_seen − AUROC_unseen).

- **K1 (identity, elementary):** Δ(ω) = ω · (AUROC_in − AUROC_out), linear in ω for a fixed scorer.
  Therefore a dose–response in the mixing fraction ω with a fixed scorer is not evidence of anything
  (R3 item 6 already said so). K1 is used only as an implementation check.
- **K2 (closed form, derived + simulated at G ≥ 40, d ≥ 32 only):** S_λ = X'X/N + λI,
  s = tr S_λ⁻¹, w₁ = 1+(n−1)ρ, Q = squared Mahalanobis distance.
  E Q_unseen = s ; **ΔQ := E Q_seen − E Q_unseen = − n ρ² s² / (N + w₁ s)** ; |ΔQ| < ρ s.
  Deterministic equivalent for s: s = Nδ with
  δ = d / ( Nλ + G w₁/(1+w₁δ) + (N−G)(1−ρ)/(1+(1−ρ)δ) ).
  Gaussian map: AUROC ≈ Φ((μ_o−μ_id)/√(σ_o²+σ_id²)), which under-predicted the toy Δ by 15–25 %.
- **K3 (universal cap, proof sketch in REPORT, not independently checked):** for **every** scorer,
  E_u[Δ] ≤ (ω/2)·√( ((1−ρ)^{−d} − 1) / G ). Non-vacuous only for G ≳ (1−ρ)^{−d}.
  Covariance-general version: d → d_ρ = Σ_j −ln(1−ρ_j).
- **K4 (kNN, conjecture, empirical in toy):** the kNN leak is a function of ρ√d, nearly independent of n,
  for k ≤ n−1 (k ≤ fit points per group).
- **K5 (label-free fingerprint T):** T = (mean NN²(eval→fit) − mean NN²(fit→fit, leave-one-out)) /
  √(Var_e/m + Var_f/N), squared Euclidean nearest-neighbour distances. Toy: T ≈ 0 when eval overlaps
  fit; T ≫ 0 when disjoint and ρ√(d/2) ≳ 1. **Caveat:** the toy SE ignores dependence between
  fit→fit distances (mutual neighbours, shared groups), so it is probably anti-conservative on real
  features; the order treats T as a statistic with frozen thresholds, not as a z-score.
  The earlier midpoint-threshold estimator (`s2c`) failed and is not to be used.
- **K6 (confidence sequences):** DEFF = 1+(n−1)ρ. Two estimands: **θ_group = E[m_g]** (mean over the
  population of future groups; unit = group) and **θ_size = E[n_g m_g]/E[n_g]** (size-weighted mean; unit =
  sample). A sample-level CS (N1, N2) targets θ_size; a group-level CS (N3, G1; unit = group mean) targets
  θ_group. They coincide when group size is independent of m_g (**non-informative** cluster size, incl. equal
  sizes) and differ when it is not (**informative** cluster size). In the non-informative regime a sample-level
  CS under-covers when DEFF > 1; a group-level CS is valid for any ρ and costs ≈ DEFF× more groups. In the
  informative regime each method is judged against the estimand it targets and the cross-estimand coverage is
  reported separately (never pooled into the dependence conclusion). Label-free group recovery is claimed only
  when feature-ICC·√(d/2) ≳ 3 (REPORT's falsifier threshold; the underlying lemma is a conjecture).
- **K7 (sign law, first order):** P = N(0,Σ), Q = N(m,C), ΔM = C + mm' − Σ, scorer s_A(x) = x'Ax.
  AUROC(Q vs P) − ½ ≈ Φ(z_A) − ½ with
  z_A = ⟨A,ΔM⟩ / √(2 tr((AΣ)²) + 2 tr((AC)²) + 4 m'ACAm).
  On {⟨A,ΔM⟩ = 0}: AUROC − ½ ≈ −φ(0) κ₃/(6σ³), κ₃ = 8[tr((AC)³) + 3 m'A(CA)²m − tr((AΣ)³)], σ the
  denominator of z_A. (Derived from Gaussian quadratic-form cumulants; the rate is not looked up.)
  **For a fitted A on real features ⟨A,ΔM⟩ equals the difference of mean scores, so on real data K7's
  first-order sign is "sign of the mean-score gap"; its content on real data is where the mean-gap sign
  and the AUROC sign disagree (skew), and the Edgeworth term.**
- **K8:** trace-balanced covariance shifts are blind for the quadratic scorer (AUROC within 0.01–0.02 of ½)
  yet detectable by another monitor, with KL(Q‖P) unbounded in d.

Known caveats from the toy: s2a validated K2 at d ∈ {32,128}, G ∈ {40,100,200} (5–8 % low in magnitude
at d = 32); **the Camelyon regime (G_fit = 15, n_g in the thousands, d up to 2560) is outside what was
simulated**; s2e/s2f used the unregularised inverse of a covariance with N < d at d ≥ 256 (NaN or
meaningless Mahalanobis values) — do not reproduce those cells with a plain inverse; `s2e` T(ω=0) at
d = 1024 is also in that regime.

---

## Reference: Camelyon17 manifest facts used by this order (read from Vinh's repo files; re-verified on HPC by item I and G0)

Everything below is **verified** from the committed repo / docs unless marked *inferred*; the HPC checks in item I
(`MANIFEST_OK`) and G0 re-read the real metadata and stop (S11) if they disagree. Sources in `CHANGELOG_v2.md`.

- WILDS Camelyon17 v1.0 **train split = 3 hospitals {0, 3, 4} × 10 slides = 30 slides**; 302,436 train patches
  (per-slide 1,430–55,149; hospitals 3 and 4 each have one slide > 50,000 patches); `id_val` = 33,560 patches
  **of the same 30 slides** (WILDS design; id_val shares slides with train).
- **OOD = hospital 2 (`test` split), 85,054 patches, 9 patients; hospital 1 (`val`) is not used by this order.**
  Hospitals 1 and 2 are never in any fit, fine-tune, checkpoint-selection or scorer-calibration set.
- The `paper_2fold` / H11c folds: per hospital the 10 slides are shuffled (`default_rng(1000 + seed)`) and
  dealt alternately, so each fold holds **5 slides per hospital = 15 slides** (`n_groups_fit = 15/15` in the R3
  item-5 tables; seed-42 fold sizes 97,099 / 205,337 train patches).
- **Not verified here (HPC must read them):** the slide IDs, the slide → hospital map, per-slide and
  per-hospital patch counts, and the H11a/b overlap outcomes. No design below depends on a per-slide count
  except as reported by item I / G0.

---

## Order, priority and early stops

Stages run in this order. **A decides how much of the rest runs.**

| Stage | Items | Runs when |
|---|---|---|
| 0 | 0 setup, I inventory | always, first |
| 1 | **A** (A0 → A1 → A2 → A3) | always; the decisive step; ≈ 1–2 days |
| 2 | B, C-core (C0–C4), D-synthetic (D1–D2) | after A0 passes; in parallel with A1–A3 if GPUs idle |
| 3 | C-extended (C5), D-real (D3–D5), E, F, H-cheap (H1–H3) | after stage 1 report |
| 4 | G (G0 manifest check, G1 CPU; G2 full fine-tune), F4 corruption extraction, H4 | G2 **only if A = GO** |
| 5 | FINAL consolidation | always |

Early stops (decided now):
- **A0a fails or A0c fails → STOP everything except item E and F1** (S3; A1–A3 and every other item are `not run: S3`). **A0b fails → S2** (A, B, D3–D5, E3–E4, F2–F4, G and H stop; C, E1–E2 and F1 may continue).
- **A = NO-GO:** skip G2; run C-core and D at their listed scope only; skip C5's real-cell validity map
  (C5 on synthetic remains); the consolidation states that the Gaussian formula is not supported on real
  features in the tested set.
- **A = PARTIAL:** run everything except G2.
- **A ≠ GO in general (A = PARTIAL, NO-GO or INCONCLUSIVE):** G2 does not run and G-ii is `not run`; G0 and G1 still run (unless S2 or S3 stops G) and G-i is evaluated; see "Run conditions of G".
- **A = GO:** run all, including G2.
- Items E and F never depend on A.
- Any item's cap reached → stop that item (rule 16).

### Budget table (GPU-hours; planning guesses, not measurements — replaced from item 0 calibration)

| Item | Content | Planning estimate | Hard cap |
|---|---|---|---|
| 0 | setup, tests, microbenchmarks | 0.5 | 1 |
| I | inventory | 0 (CPU 1 h) | 0 |
| A | A0–A3: LOGO refits, ρ̂/s on every cell, subsampling designs | 4–8 | 12 |
| B | kNN on real features, per-k, matched toy simulator | 6–12 | 18 |
| C | GPU simulation sweep C0–C5 | 80–130 | 160 |
| D | fingerprint T: synthetic search + real | 8–15 | 24 |
| E | confidence sequences + clustering | 1–3 (CPU ≈ 200 core-h) | 6 |
| F | sign law, constructed shifts, corruption extraction | 6–12 | 20 |
| G | G1 frozen (≤ 5) + G2 full fine-tune (24 runs) | 45–70 | 80 |
| H | controls, estimator coverage, falsification table | 3–6 | 10 |
| **Total** | | **≈ 155–260** | **≈ 330** |

The REPORT's own rough figure for controlled-split fine-tuning (30–60 GPU-h for ~60 ViT-B-scale runs)
and for embedding extraction (≈ 1–2 GPU-h for ~100k images × 20 backbones, A100-class) is the only
sourced figure; everything else is a guess. Record actual GPU-hours per item from `sacct`
(`AllocTRES`, `Elapsed`) in each REPORT.

---

## Hard stop list (the whole campaign or the named item stops and reports; no workaround)

**Status after a stop (v2.4 wording; rule 18 applies).** What a stop prevents from running is `not run: S<k>`, with the item
verdict `not run` or the sub-criterion status `not run`; this is never NO-GO and never a pass. A stop that is itself an evaluated
failure is a failure of the criterion that names it: S8 → C-ii `fail`; S9 → H-i `fail`; S5 → see S5. S1 stops the campaign: every
item not yet completed is `not run: S1`.

- **S1** A required bundle file is missing or has a wrong sha256.
- **S2** crossfit-ood v3 tests are not 44/44, or Track A Δ cannot be reproduced to 1e-6 (A0b). Items A, B, D3–D5, E3–E4, F2–F4, G and H stop (everything that reads R3 cells); C, E1–E2 and F1 may continue.
- **S3** A0a fails (or A0c fails, see Early stops): the toy closed form or the sim reference numbers cannot be reproduced within the
  stated tolerance. Everything except E and F1 stops: the implementation is wrong or the claim is.
- **S4** Fewer than 8 Camelyon backbones have both cached features and R3 item-1 Δ for `mahalanobis_l2`:
  item A is INCONCLUSIVE; stop A1/A2 and report the inventory.
- **S5** A sanity check Δ(0) CI excludes 0 (G, A2 at G' = full vs identical designs): stop that
  model × scorer. Consequences (v2.4 wording): in G1 a stopped combination is an evaluated failure of G-i check c2 (G-i `fail`); in
  G2 it is an evaluated failure of the G-ii sanity condition (G-ii `fail`, see "G-ii under the reduction ladder"); in A2 the
  backbone's A2 is `not run: S5` and counts as incomplete for the A-iii / INCONCLUSIVE rule ("fewer than 8 backbones with complete A2").
- **S6** Any item exceeds its cap by 20 %, or the campaign exceeds 330 GPU-h.
- **S7** Any job would write under `results/r3/`, `outputs/`, or overwrite a non-T1 file.
- **S8** Item C4: any non-vacuous configuration in which the 99.9 % upper bound of E[Δ] exceeds the cap
  K3. Stop C, report with the configuration and seeds; do not "fix" the cap.
- **S9** Item H1: a negative control with an expected null shows |Δ| > 0.02 with a CI excluding 0 in more
  than 5 % of cells: stop H, report; do not reinterpret the pipeline.
- **S10** A decision criterion is changed after output exists (rule 1): the agent stops and reports
  instead of editing.
- **S11** Camelyon manifest mismatch. Item I's manifest check or G0 finds that the 30 training slides are not
  exactly 10 per hospital over hospitals {0, 3, 4}, that hospital 1 or 2 appears in any fit / fine-tune /
  selection set or among the 30 slides, or that the item-G design table (below) cannot be reproduced with the
  stated counts. G (G0 aborts; G1 and G2 `not run: manifest mismatch`) stops and reports the real vs expected
  counts in `results/t1/G/MANIFEST_MISMATCH.md`; if item I's flag `MANIFEST_OK` is false, the hospital-stratified
  designs A2, E3, E4 and H2 are `not run` as well (E3 and E4 `not run`: E-i and E-iii are then `not evaluable` unless a component already evaluated has failed, see item E; A-iii is then recorded `not evaluable`: never a pass (A cannot be GO) and never a failure, and the A verdict is set by
  the "A-iii `not evaluable` under S11" clause of the A verdict table (rule 18; NO-GO only if it still holds with A-iii treated as passed, otherwise
  INCONCLUSIVE); H2 (the whole item, including its medbench placebo) is recorded `not run: manifest mismatch (S11)` and so is D4's permuted-label check, H-ii is then `not evaluable` and counts as not
  passed, and H cannot be GO, see item H). **No redesign, no re-count, no "nearest design" is allowed on the HPC** (rule 2).

## Things NOT to do

- No new scoring method, detector, estimator, regulariser or "improved formula" outside the
  pre-specified correction family of item C.
- No benchmark sweep of detectors without a discriminating comparison (a comparison whose outcome can
  differ between the hypotheses). Seen-vs-unseen mixing dose–response for a fixed scorer is **not**
  discriminating (K1) and is run only as an implementation check.
- No retrospective tuning of k, λ, shrinkage, thresholds, ω, grids, or exclusion of "bad" backbones.
- No re-deriving or re-running of the published numbers; no touching the AISTATS/MIDL manuscripts;
  no changes to crossfit-ood v3, `scripts/r3/`, or `results/r3/`.
- No use of OOD data or labels in a predictor labelled Tier-L (fit-set-only). No use of held-out
  backbones, grids or datasets in any fit.
- No re-design of the Camelyon slide table on the HPC if the manifest differs from the table (S11), and no use
  of hospital 1 or 2 in any fit, fine-tune, checkpoint selection or calibration.
- No causal language and no "proved" language (rule 17). No emails, no pushes to remote branches other
  than `t1`.

---

# ITEMS

Outputs: `results/t1/<item>/` in the DST-Skin clone on branch `t1`; scripts in `scripts/t1/`
(shared helpers in `scripts/t1/common.py`: cell registry, fold reconstruction through the package's own
fold code, scorer-precision extraction, ICC/ρ̂/s estimators, LOBO machinery, bootstrap).

## 0. Setup (CPU, ≤ 1 h; GPU ≤ 0.5 h for microbenchmarks)

1. `cd ~/DST-Skin && git status` clean; `git checkout -b t1`; record HEAD hash.
2. Verify `~/handoff/crossfit-ood_v3.zip` sha256 (expected above). If `~/handoff/crossfit-ood_v3/` and
   `~/.venvs/crossfit-r3` from R3 item 0 exist and `pip show crossfit-ood` reports 0.0.3.dev0, reuse
   them read-only; otherwise unzip to `~/handoff/crossfit-ood_v3/` and `pip install --no-deps -e` in
   `~/.venvs/crossfit-t1` (same isolation as R3 item 0). Do not modify the package.
3. `pytest -q` in the package: record pass count (R3 recorded 44/44). Not 44/44 → S2.
4. Unpack the bundle to `~/handoff/t1/bundle/`; verify sha256.
5. Record python / numpy / scipy / scikit-learn / torch / CUDA versions, `nvidia-smi -L`, `sinfo`
   GPU partitions, and a 60-second microbenchmark per GPU type: float64 matmul 4096×4096 throughput,
   float64 `torch.linalg.inv`/`cholesky` at d = 2560, kNN (GPU) 50,000 × 100,000 × 2560 time. Write
   `results/t1/0/throughput.json` and `results/t1/0/BUDGET.md` (planning estimates re-derived from these).
6. Re-run the GPU-kNN equality check (rule 4) on 2 cells; record max |Δ(AUROC)|.
7. Write and commit the precommit (rule 1). Record the hash in `results/t1/0/REPORT.md`
   (versions, hashes, test count, microbenchmarks, kNN equality, one-line verdict PASS/FAIL).

Do not start any other item until the precommit commit exists.

## I. Inventory (CPU, ≤ 1 h)

Locate and list, with absolute path, size, mtime and sha256, every input T1 needs; write
`results/t1/I/registry.csv` (one row per **cell** = dataset × backbone × seed) and `inventory.md`:

- R3 item-1 inputs `~/r3work/item1/inputs/<cell>.npz` (keys `features_train, groups_train,
  features_id_eval, groups_id_eval, features_ood[, labels_train, groups_ood]`, `<cell>_head.npz` if any)
  and outputs `~/r3work/item1/out*/<cell>/paper_ci.csv`: Camelyon17 8 CNNs and 5 FMs (every seed),
  Medbench DermaMNIST, ISIC 2019, Kermany v2, BreakHis (`results/r3/1/` if present).
- The fold assignment each cell used (`paper_2fold`, fold seed = model seed): obtain it through the
  package's own function, not by re-implementation; record `n_groups_fit`, `N_fit` and the group-size
  distribution per fold.
- `outputs/rigor_pack/foundation_gate/feats/camelyon_<fm>.npz`, `outputs/features/...` CNN caches,
  `outputs/rigor_pack/miccai_campaign/scores/camelyon_<fm>[_s43|_s44].npz`,
  `outputs/rigor_pack/miccai_campaign/trackA_foundation/slide_identity.csv`,
  R3 item-3 whitening caches `outputs/rigor_pack/r3/item3/*.json`, R3 item-6 `results/r3/6/dose.csv`.
- Paper-3 caches (`*_z.npz` with `z_train, y_train, z_id, y_id, z_ood, y_ood`; 15 files expected,
  5 seen in the local copy: runA_grl_s42/s62, runB_orth1_s42, runB_s42/s62).
- **Camelyon17 manifest** (CPU): read the real metadata with the R3 helper `load_camelyon_metadata` (as used in
  `scripts/r3/item6_dose.py`) and the `groups_train` of every Camelyon cache; write
  `results/t1/I/camelyon_manifest.csv` (columns `slide, hospital, n_train_patches, n_idval_patches`; one row per
  training slide) and `camelyon_manifest.md` (per-hospital slide counts and patch totals; hospital of the OOD
  split; per-fold slides per hospital of the `paper_2fold` assignment of every Camelyon cell). Check and record
  `MANIFEST_OK` = true iff: the hospitals of the 30 training slides are exactly {0, 3, 4} with exactly 10 slides
  each; the OOD patches are all hospital 2 and no OOD patient or slide is among the training slides; every
  Camelyon cache's slide set equals the metadata's; every fold of every Camelyon cell holds exactly 5 slides per
  hospital. Patch totals are compared with the documented 302,436 / 33,560 / 85,054 and any difference is
  recorded (WARN), not a stop. `MANIFEST_OK` is written to the file `results/t1/I/MANIFEST_OK` (content `true` or
  `false`), which is the file G0 reads. Used by A2, E3, E4, H2 and G0; false → S11.
- Retrain-arm code and logs for GPU-hour accounting: `scripts/r3/item5_b_isbi_patch2.py`,
  `decisions/precommit_isbi_patch2_2026-10-06.md`, `logs/` of the isbi_patch2 and p2d training jobs.

Columns: `cell_id, dataset, backbone, family{CNN|FM}, seed, d, input_npz, paper_ci_csv, id_acc,
n_groups_fit_f0, n_groups_fit_f1, N_fit_f0, N_fit_f1, has_labels, has_head, has_delta_mahalanobis_l2,
has_delta_knn_mean_cosine, notes`. A missing item is `NA`, never imputed. Verdict: counts per dataset
and family; S4 if < 8 Camelyon backbones have `has_delta_mahalanobis_l2`; S11 if `MANIFEST_OK` is false.

Outputs `results/t1/I/`: `registry.csv`, `inventory.md`, `camelyon_manifest.csv`, `camelyon_manifest.md`, `MANIFEST_OK`,
`DEVIATIONS.md`, and `REPORT.md` (rule 15: the count tables, commit and precommit hash, wall clock, the `MANIFEST_OK` value and
the S4 / S11 flags; the one-line status is **PASS** if S4 is not triggered and `MANIFEST_OK` is true, otherwise **FAIL (S4)** /
**FAIL (S11)**; item I has no GO / NO-GO verdict and is not an input of the FINAL survival rules).

---

## A. THE DECISIVE STEP — does the closed form predict real leakage better than ICC alone? (stage 1)

**Question.** On real cached features, does the Gaussian-toy prediction of ΔQ and of Δ explain the
measured random-vs-group-split Δ out of sample, and better than ICC alone, the number of groups alone,
and d/N?

**Compute:** CPU array + GPU for LOGO refits and subsampling. Estimate 4–8 GPU-h, cap 12.

### A0. Unit tests before any real-data claim (≤ 0.5 GPU-h)

- **A0a Toy reproduction.** Implement the closed form (K2, with the fixed-point equation) and a
  float64 GPU simulator; reproduce these bundle cells (`s2a_out.txt`, sim ± SE / theory, 24 replicates):
  (d = 128, G = 40, n = 10, ρ = .6, λ = 0): ΔQ sim −170.27 ± 0.66, theory −168.93;
  (d = 128, G = 100, n = 5, ρ = .3, λ = .1): sim −12.73 ± 0.23, theory −12.99;
  (d = 32, G = 40, n = 10, ρ = .6, λ = 0): sim −10.89 ± 0.20, theory −10.43.
  **Pass:** the new simulation within 3 combined SE of the bundle simulation, and the new theory
  function equal to the bundle theory to 1 % relative. Fail → S3.
- **A0b Track A reproduction.** For 3 cells (resnet50 s42, uni, dinov2_vitb14) recompute Δ for
  `mahalanobis_l2` with your own fit/score code using the package fold assignment and compare with
  the R3 item-1 `paper_ci.csv` Δ. **Pass:** |difference| ≤ 1e-6. Fail → S2.
- **A0c Unit conversion.** In float64 on one cell verify the scorer's precision
  P_scorer = (1/(1−δ̂)) · (Ŝ_n + λ_eff I)⁻¹ / μ̂ to 1e-9 relative, where Ŝ is the covariance the
  scorer centred and fitted, μ̂ = tr Ŝ/d, δ̂ the Ledoit–Wolf shrinkage, Ŝ_n = Ŝ/μ̂, λ_eff = δ̂/(1−δ̂).
  All K2 quantities are computed in the normalised units (features divided by √μ̂) and converted back
  with the factor 1/(1−δ̂). If class labels are used by the scorer (class-conditional means), Ŝ is the
  covariance of class-centred residuals and ρ̂ is computed on those residuals; document it.

### A1. Cross-backbone prediction (CPU + GPU)

For each cell and fold f ∈ {0,1} of `paper_2fold` (fit set F_f; **seen** = ID-eval samples of groups in
F_f; **unseen** = ID-eval samples of the other groups), compute with the scorer's own preprocessing,
fit on F_f only:

*Fit-set descriptors (Tier-L: no OOD data, no eval data):*
1. N, G, group sizes n_g; n̄ = N/G; n_w = Σ n_g² / N; d; ID accuracy; LW shrinkage δ̂.
2. **ρ̂_w** = 1 − tr(P Ŵ)/tr(P T̂) — P the scorer's precision, Ŵ the pooled within-group covariance
   (about group means, divisor N−G), T̂ the total covariance (divisor N−1). **ρ̂_raw** = same with
   P = I. **ρ̂_anova** = mean over coordinates of (MSB−MSW)/(MSB+(n₀−1)MSW),
   n₀ = (N − Σn_g²/N)/(G−1). Also the existing R3-item-3 whitened ICC tr(PB)/tr(PT) for continuity
   (report only).
3. **s_logo** (primary): leave-one-group-out refit of the scorer (G refits, via covariance downdating
   is allowed; K = 10 group folds if G > 40): s_logo = (1−δ̂) · mean over left-out-group points of
   Q_scorer. **s_tr** (secondary): tr S_λ⁻¹ in normalised units. Report s_logo/s_tr.
4. **Saturation ratio** r_sat = |ΔQ_pred| / (ρ̂_w s_logo).

*Predictions:*
5. **P_ΔQ (primary):** ΔQ_pred = (1/(1−δ̂)) · Σ_g π_g · [ − n_g ρ̂_w² s_logo² / (N + w₁,g s_logo) ],
   w₁,g = 1+(n_g−1)ρ̂_w, π_g = share of seen-eval points in group g. (Variants, reported, not primary:
   n_w plug-in; n̄ plug-in; s_tr instead of s_logo; ρ̂_raw / ρ̂_anova instead of ρ̂_w.)
6. **P_Δ (primary, additive map; uses unseen and OOD scores, so it is Tier-S):** with scores oriented
   higher = more ID, Δ_pred = AUROC(score_unseen + |ΔQ_pred|, score_OOD) − AUROC(score_unseen, score_OOD),
   fold-averaged. Secondary: multiplicative map Q_seen = Q_unseen (1+ΔQ_pred/s) ; Gaussian CLT map.

*Measured (post-hoc relative to the formula; this is the target):*
7. Δ_meas = R3 item-1 Δ for `mahalanobis_l2` (verified in A0b); AUROC_seen, AUROC_unseen;
   **ΔQ_meas** = mean Q over seen-eval − mean Q over unseen-eval (scorer units, same fit);
   s_unseen = (1−δ̂) · mean Q over unseen-eval, as a check on s_logo.
   Caveat to write in the report: seen vs unseen eval sets differ also in hospital/class composition;
   Camelyon is stratified by hospital in the fold construction (hospitals {0, 3, 4}, 5 slides per hospital per
   fold; other datasets may not be).

*Baselines (each an equal-capacity LOBO calibration y = a + b·z, fit on training backbones only):*
B0 constant; **B1 ICC-only** z = ρ̂_raw; **B1w** z = ρ̂_w; B2 number of groups only z = G (degenerate on
Camelyon: all cells share G; evaluated only in A2/medbench); B3 z = d/N; B4 z = ρ̂_w²; B5 bound-only
z = ρ̂_w · s_logo. No descriptor is chosen after seeing results.
The predictor under test: z = Δ_pred (and, at the ΔQ level, ΔQ_pred) with the same (a,b) calibration.
A zero-parameter version (a = 0, b = 1) is reported beside it.

*Evaluation units and splits:*
- Unit = backbone (seeds and folds averaged within backbone; seed-level tables in the appendix).
- **LOBO** over the 13 Camelyon backbones (primary); leave-family-out (train CNN → test FM; train FM →
  test CNN); **within-FM** LOBO with n = 5 (report; no power, say so); within-CNN n = 8.
- Medbench: leave-one-dataset-out over the 4 datasets (report only; n = 4).
- Metrics per held-out backbone b: absolute error e_b = |Δ_meas − Δ_hat|; pooled MAE; Spearman between
  held-out predictions and Δ_meas (permutation null by pairing, `default_rng(102)`, 10,000);
  R²_oos vs B0; and at the ΔQ level the relative error |ΔQ_pred/ΔQ_meas − 1| (zero-parameter).

### A2. Within-backbone subsampling designs (predicts the dependence on G, n, N, λ that A1 cannot probe)

Camelyon has G_fit = 15 and the same N for every backbone, so A1 cannot separate K2 from the bound
ρ s (check r_sat). A2 varies the drivers on the same cached features, CPU/GPU, no new extraction:
- Cells: 13 Camelyon backbones, seed 42 (FM features do not depend on the seed; CNN cell = seed 42).
- Per fold: fit groups subsampled to G' ∈ {3, 6, 9, 12, 15} (stratified by hospital over the **3 source
  hospitals {0, 3, 4}**: exactly G'/3 ∈ {1, 2, 3, 4, 5} slides per hospital, drawn without replacement from that
  hospital's 5 slides in the fold; `SeedSequence`; G' = 15 keeps all 15 slides), patches per retained group n' ∈ {10, 30, 100, 300, 1000, all}; seen-eval restricted
  to retained groups; unseen-eval unchanged; **10 repeats** per (G', n'); Δ_meas by the package scorer
  on the subsampled fit set (`mahalanobis_l2`; LW δ̂ recomputed per fit — it varies with N/d, and that is
  part of what is tested); kNN is **not** run here (item B).
- Sanity (S5): the design (G'=15, n'=all) must reproduce the A1 fold Δ to 1e-6. A2 starts only if item I's
  `MANIFEST_OK` is true and the fold in use holds 5 slides per hospital (else S11).
- Descriptors, P_ΔQ and P_Δ recomputed on each subsampled fit set exactly as in A1 (Tier-L descriptors
  from the subsampled fit only).
- Baselines (LOBO over backbones, equal-or-larger capacity): the A1 baselines that vary within a
  backbone (B3, B4, B5) and **B6-scaling** — OLS of Δ_meas on (ρ̂_w, log G', log n', d/N') fit on training
  backbones (5 parameters; the strong mechanism-free baseline).
- Metrics: pooled MAE over all (design, repeat) points of held-out backbones; within-backbone Spearman of
  prediction vs Δ_meas over the 30 designs, averaged over backbones; the LOBO paired difference
  D_b = e_b(baseline) − e_b(P_Δ) with cluster bootstrap over backbones.

### A3. Estimator audit (CPU)

For ρ̂_w, ρ̂_raw, ρ̂_anova, s_logo, ΔQ_pred: group-jackknife SE per cell; sensitivity of ρ̂_w and ΔQ_pred
to group-size imbalance (n_w/n̄), to class-conditional vs class-agnostic centring, and to the A0c
conversion. Report which cells have r_sat ≥ 0.9, δ̂ ≥ 0.5, or N < 5d (formula regime unverified).

### Outputs (`results/t1/A/`) (there is no per-item `PRECOMMIT.json`: the single precommit file is `results/t1/PRECOMMIT_T1.json`, rule 1, and A's `REPORT.md` records its hash)

`cells.csv` (every descriptor, prediction and measurement per cell × fold),
`a1_lobo.csv`, `a1_family.csv`, `a2_designs.csv`, `a2_lobo.csv`, `a3_audit.csv`, `tests.csv`,
`scatter_dq.png`, `scatter_delta.png`, `a2_dose.png`, `DEVIATIONS.md`, `REPORT.md` (which must state the A-i … A-iv status words, and when A-iii is `not evaluable` under S11 the sentence of the A verdict table's S11 clause, and must contain the
"Interpretation limits" paragraph below, verbatim in substance: 13 backbones are few; B = 10,000 over backbones
does not create independent backbones; within-FM n = 5 is descriptive only).

### Pre-registered decision (fixed now)

Let D_b(B) = e_b(baseline B) − e_b(P_Δ) averaged over held-out backbones; CI = percentile cluster
bootstrap over backbones (B = 10,000, `default_rng(101)`).

| Sub-criterion | Pass when |
|---|---|
| **A-i** (A1 vs ICC, Camelyon LOBO, n = 13) | the lower 95 % CI bound of mean D_b is > 0 against **each** of B1 and B1w (intersection–union; both must pass) |
| **A-ii** (A1 vs the other single-descriptor baselines) | same, against B3, B4, and — **only if** r_sat < 0.9 in at least 20 % of cells — B5. If r_sat ≥ 0.9 in ≥ 80 % of cells, A1 cannot discriminate P from the bound ρ s; state it and move the P-vs-B5 comparison to A-iii |
| **A-iii** (A2 within-backbone) | lower 95 % CI bound of mean D_b > 0 against B6-scaling **and** against B5, pooled over designs; and mean within-backbone Spearman > 0 with lower CI bound > 0. If `MANIFEST_OK` is false (S11), A2 is `not run` and A-iii is `not evaluable` (`pass = na`): never a pass, never a failure (v2.5; see the S11 clause below the verdict table) |
| **A-iv** (mechanism level, ΔQ) | median zero-parameter relative error abs(ΔQ_pred/ΔQ_meas − 1) ≤ 0.35 over cells **and** pooled R²_oos of the LOBO-calibrated ΔQ_pred vs B0 > 0 |

| Verdict | Rule |
|---|---|
| **GO** | A-i and A-ii and A-iii and A-iv all pass |
| **PARTIAL** | A-i passes and at least one of A-iii, A-iv passes, or A-iii and A-iv pass without A-i |
| **NO-GO** | otherwise (including A-i fails); **except** that when A-iii is `not evaluable` under S11 the S11 clause below decides (NO-GO only if A-i and A-iv both failed). Consequence: the Gaussian closed form is not supported as the mechanism of real-feature leakage in this set; Idea 2 is downgraded to toy theory; G2 is skipped; report and stop |
| **INCONCLUSIVE** | S4 triggered, or r_sat ≥ 0.9 in ≥ 80 % of cells **and** A-iii not evaluable (fewer than 8 backbones with complete A2), or A-iii `not evaluable` under S11 and the S11 clause below awards INCONCLUSIVE |
| **not run** | A0a or A0c failed (S3), or A0b failed (S2): A1–A3 are `not run`, every A sub-criterion is `not run` (never NO-GO, never a pass); G2 is `not run` |

**A-iii `not evaluable` under S11 (v2.5; rule 18 (a)–(c); no threshold or criterion changed).** A-iii `not evaluable` is never a pass, so
A cannot be GO; it is never a failure and never evidence against the hypothesis. A-ii does not enter the PARTIAL / NO-GO rows and is not
used here. The verdict is read from the PARTIAL / NO-GO rows above, where NO-GO is awarded only if it still holds with A-iii treated as
passed, which gives:

| A-i | A-iv | A (A-iii `not evaluable` under S11) |
|---|---|---|
| pass | pass | PARTIAL (A-i passes and A-iv passes; A-iii is not counted as a pass) |
| fail (evaluated) | fail (evaluated) | NO-GO (holds with A-iii treated as passed) |
| any other combination (one pass and the other fail; either `not evaluable` or `not run`) | | INCONCLUSIVE (never NO-GO: with A-iii treated as passed, the row would be PARTIAL or not NO-GO) |

(S4 still gives INCONCLUSIVE, and A0a / A0b / A0c failing still gives `not run`, as in the table above. `A ≠ GO` anywhere in this order means
A is PARTIAL, NO-GO, INCONCLUSIVE or `not run`.) `REPORT.md` of A states the A-i … A-iv status words and, in this case, the sentence
"A-iii is not evaluable (S11: manifest mismatch); it is not counted as a pass or as a failure; the A verdict follows rule 18."

Within-FM, across-family and medbench results are reported in full and **do not enter the verdict**.
Anything not listed above is post-hoc.

**Interpretation limits of the A verdict (binding wording in `results/t1/A/REPORT.md`, stated next to the
verdict, not in a footnote):** (1) 13 backbones is a small number of independent units; the LOBO result is a
screening result and its confidence intervals may be unstable. (2) The cluster bootstrap with B = 10,000 over
backbones resamples the same 13 backbones; it does **not** create independent backbones and cannot make the
interval narrower than 13 units allow. (3) The within-FM analysis has n = 5 units (and within-CNN n = 8): it is
descriptive only, has no power, and must not be used to claim generalisation within the foundation-model family
or within the CNN family. (4) A GO means "the formula predicted held-out backbones better than the stated
baselines on these 13", not that the Gaussian mechanism is established; the saturation ratio r_sat and the
A-ii/A-iii rules above stay as pre-registered.

**Permitted post-hoc follow-up (label `_posthoc`):** the same analysis with ρ̂_raw/ρ̂_anova substituted;
excluding cells with δ̂ ≥ 0.5; per-scorer ΔQ residuals vs d, N/d, class imbalance. No new predictors.

**GPU-hours:** estimate 4–8, cap 12. Seeds: 101/102 + `SeedSequence([20261010, 1, hash, rep])`.

---

## B. kNN leakage on real features (stage 2)

**Question.** Is the real kNN Δ a function of ρ√d_eff (K4) and predicted by the matched toy simulator,
better than ICC alone? How does it depend on k?

**Compute:** GPU kNN on cached features. Estimate 6–12 GPU-h, cap 18.

**Inputs:** the same cells and `paper_2fold` folds as A1; scorer `knn_mean_cosine` with
k ∈ {1, 5, 10, 20, 50, 100, 200} (primary k = 50, as Track A); also the k-th-neighbour-distance variant
for the same k (report only). One neighbour search with k_max = 200 per (fit, eval-set) pair; the
mean-over-k and k-th-distance scores are derived from it. Δ_meas(k) = Track A definition.

**Descriptors (Tier-L, fit set only; L2-normalised features):** ρ̂_cos (ANOVA-corrected ICC on the
normalised features, as ρ̂_raw), PR = (tr T̂)²/tr(T̂²) (participation ratio), ambient d, TwoNN
intrinsic dimension (Facco et al.; fit set, 20,000-point subsample), n_g distribution, own-group
neighbour rate r_NN(k) = fraction of the k nearest fit-neighbours of a **seen**-eval point that belong
to its own group (label-dependent mechanistic indicator; reported, not a Tier-L predictor), and
T_unseen (item D definition, on unseen eval; label-free).

**Predictors:**
- **x₁ = ρ̂_cos √PR** (primary); x₂ = ρ̂_cos √d; x₃ = ρ̂_cos √TwoNN.
- **MTS (matched toy simulator, primary zero-real-fit predictor):** for each cell × fold simulate the toy
  with d_toy = round(PR) (capped at 1024), ρ = ρ̂_cos, the **actual** group-size vector n_g, N, and an
  isotropic-scale OOD whose scale is set (bisection, on the toy only) so that the toy AUROC_unseen of
  kNN-k equals the cell's measured AUROC_unseen for that k; read Δ_MTS(k). 16 replicates per cell.
  The OOD parameter uses the measured unseen AUROC only (no leakage information).
- Baselines (LOBO equal capacity, as A1): B1 ICC-only (ρ̂_cos), B0, B3 d/N, T_unseen-only; and, as a
  mechanistic upper reference not entering the verdict, r_NN(k).

**Analyses (all in `tests.csv`):**
1. **Collapse:** across all cells × k with k ≤ n_g, monotone (isotonic) fit of Δ_meas on x₁, x₂, x₃ vs
   on ρ̂_cos alone and on d alone (LOBO R²_oos for each).
2. **Per-k sensitivity:** Δ_meas(k) curves per cell; prediction (K4) that for k ≤ n_g the leak is flat in
   k to first order; on cells with n_g < 100 (expected: some medbench datasets — if none exist, write
   "not testable") the prediction that Δ_meas(k) falls once k exceeds n_g.
3. **MTS vs measured:** MAE, Spearman, LOBO difference vs B1.
4. **A2-style design variation for kNN** is **not** run (cost; covered by C3 in simulation).

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| B-i (collapse) | at k = 50 on Camelyon LOBO, the LOBO R²_oos of the isotonic map from x₁ is > 0 and the paired LOBO error difference vs B1 (ICC-only) has lower 95 % CI bound > 0 |
| B-ii (MTS) | MTS (zero real-fit parameters) has pooled MAE below B1's, lower 95 % CI bound of mean D_b > 0 |
| Verdict | **GO** both; **PARTIAL** exactly one; **NO-GO** neither (consequence: no ρ√d statement about kNN on real features is made; kNN stays empirical) |

Per-k results, medbench, TwoNN/ambient variants: report only. **Permitted post-hoc:** residuals of x₁
vs intrinsic-dimension estimators; k_max sensitivity. Seeds 201/202, MTS `SeedSequence([20261010, 2, …])`.

Outputs `results/t1/B/`: `cells_knn.csv`, `perk.csv`, `collapse_lobo.csv`, `mts.csv`, `tests.csv`,
`collapse.png`, `perk.png`, `DEVIATIONS.md`, `REPORT.md`.

---

## C. GPU simulation sweep of the random-effects model (stage 2 core, stage 3 extended)

**Question.** Where in (d, ρ, G, n, ω, λ, scorer, tails, anisotropy, mixture) does K2 hold, how does
the kNN leak scale (K4), is the cap K3 ever violated, and where does the isotropic formula fail? Corrected
formulas are fit **only** on a training grid and judged on held-out grids.

**Compute:** float64 GPU simulator (one process per GPU, config shards). Estimate 80–130 GPU-h, cap 160.
Replace the estimate after item 0 throughput; truncation by rule 16 (fixed random order of configs).

**Common simulator spec (`scripts/t1/sim_re.py`).** Per config and replicate: draw group effects and fit
data per the variant; fit scorers on the fit set only; draw m = 4,000 seen points (new draws from uniformly
chosen fit groups), 4,000 unseen (fresh groups, new draws), 4,000 OOD; report ΔQ_sim (Mahalanobis),
AUROC_seen, AUROC_unseen, Δ = AUROC_seen − AUROC_unseen, orientation higher = more ID,
and Δ(ω) for ω ∈ {0, .25, .5, 1} by mixing scores (K1 check: |Δ(ω) − ω Δ(1)| ≤ 1e-12 — implementation
identity, not a result). Replicates: adaptive, until SE(ΔQ_sim) ≤ 2 % of |ΔQ_th| and SE(Δ) ≤ 0.004, with
R_min = 8, R_max = 64. OOD families (each config uses all three): (O1) isotropic scale
a² = 1 + c√(2/d), c ∈ {1, 2, 4}; (O2) mean shift |m|² = c′√(2d), c′ ∈ {0.5, 1, 2}; (O3) none (null OOD =
fresh unseen draw, for the leak itself). Scorers: **Mahalanobis** with S_λ = X'X/N + λI for
λ ∈ {0, 0.01, 0.1, 1} (λ = 0 only when N ≥ 2d, else skipped and logged) and with the scorer's own
Ledoit–Wolf; **kNN-k** (Euclidean, k ∈ {1, 5, 20}, k ≤ n; and the cosine/mean-k Track A form);
**ViM-residual** (residual norm outside the top-p principal subspace of the fit set, p = the package's
`_default_vim_dim(d)`); centred-mean variants (mean estimated from the fit set), as in real use.

### C0. Reproduction and throughput (≤ 2 GPU-h)
Reproduce, in the new simulator, the bundle cells of `s2a` (as A0a), `s2d` (exact-TV cap table d = 1, 2),
`s2f` (kNN-1 null-OOD leak: ρ√d = 1.2 → +0.025, 1.6 → +0.042, 2.0 → +0.111, 2.4 → +0.164,
4.0 → +0.476, 4.8 → +0.489 at G = 100, n = 5, 8 replicates) and the `s2b` Δ_sim lines for Maha at
d = 64/128/256 (G = 100–200, n = 5–10, ρ = .2/.4, λ = 0/.1). Tolerance: 3 combined SE; the REPORT's
numbers are 8–24-replicate toy runs, so use ≥ 64 replicates. Fail → S3. Record seconds per replicate
by (d, N) for the budget.

### C1. Closed-form validation grid (Mahalanobis ΔQ and s)
- Design: Sobol/Latin-hypercube sample of **6,000 configs** (seed 301) over: d ∈ {16, 32, 64, 128, 256,
  512, 1024, 2048}; ρ ∈ {0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7}; G ∈ {6, 12, 24, 48, 100, 200, 400};
  n ∈ {5, 10, 25, 50, 100, 500} with N = Gn ≤ 2·10⁵; λ as above. G ∈ {6, 12} (Camelyon-like) and
  n ≥ 500 are deliberately included.
- **Split fixed now:** TRAIN d ∈ {16, 32, 128, 512}; TEST-interp d ∈ {64, 256, 1024}; TEST-extrap
  d = 2048 **or** G = 6 **or** ρ = 0.7 (any of the three). Configs are assigned by these rules, not by
  random draw. The correction family is fit on TRAIN only.
- **Pre-specified correction family:** log(ΔQ_sim/ΔQ_th) = β₀ + β₁/d + β₂/G + β₃/N + β₄ρ + β₅(d/N) + β₆λ
  (ridge, penalty chosen by 5-fold CV **within TRAIN grouped by d**). No other family. Also report the
  uncorrected formula.
- Declared validity region R₀ = {d ≥ 64, G ≥ 12, N ≥ 2d} (stated now, tested, not tuned).
- Metrics on each test block: median and 90th-percentile |ΔQ_sim/ΔQ_th − 1| (uncorrected and corrected),
  relative error of s, and Δ-level error of the additive and Gaussian maps (K2 → AUROC).

### C2. AUROC-level and ω structure (inside C1 configs with O1/O2)
Δ_sim vs Δ_pred (additive map and Gaussian map) per OOD family; ω-linearity as an implementation
check; the bound |ΔQ| < ρ s (count violations).

### C3. kNN/ViM structure
- **Collapse test of K4:** for kNN-1 and kNN-5, null OOD, G ∈ {50, 100, 200}, n ∈ {5, 10, 25}, d ∈ {16, 32,
  64, 128, 256, 512, 1024}, ρ ∈ {0.03, 0.05, 0.1, 0.2, 0.3, 0.5}: collapse score = R² of a 1-D isotonic
  fit of Δ on x = ρ√d divided by R² of a 2-D smoother (kernel regression, bandwidth by CV) on (ρ, d), in
  the interior region 0.02 < Δ < 0.45.
- **n-independence** for k ≤ n: Δ at n ∈ {5, 10, 25} at equal x; and the **k > n** collapse prediction.
- ViM-residual Δ and Mahalanobis(LW) Δ on the same configs (reported).

### C4. Cap check (K3) — try to falsify
- **Exact-TV grid** d ∈ {1, 2, 3}, ρ ∈ {0.3, 0.5, 0.7, 0.9, 0.99}, G ∈ {3, 10, 30, 100, 300}: exact
  numerical TV(P_in, P_out) on a fine grid (GPU; d = 3 on a 256³ grid), ≥ 40 draws of group means each;
  compare E TV and E[Δ] of an oracle scorer (the likelihood ratio that knows the fit groups) with the cap.
- **Scorer grid** d ∈ {2, 4, 8, 16}, same ρ, G: kNN-1, Mahalanobis, oracle; the configurations where the
  cap is non-vacuous (cap < 0.45). ≥ 64 replicates each; 99.9 % upper bound of the mean Δ.
- Heavy-tailed and anisotropic repeats of the scorer grid (cap is stated for the Gaussian model; whether
  it survives is reported, not claimed).
- **Falsification rule (S8):** any non-vacuous Gaussian configuration whose 99.9 % upper bound exceeds the
  cap → stop C and report. Also report tightness = E[Δ]/cap, and the smallest G at which cap < 0.1.

### C5. Misspecification map (stage 3)
Variants (each over a 1,500-config sub-design of C1, same TRAIN/TEST rules; seed 301):
- (a) **Heavy tails:** multivariate Student-t, ν ∈ {3, 5, 10, 30}, applied to e and, separately, to u;
- (b) **Anisotropy, proportional group covariance** (B = ρΣ, Σ ≠ I): spiked (r ∈ {1, 5, 20} spikes of
  size κ ∈ {5, 20, 100}) and power-law spectra (λ_j ∝ j^{−α}, α ∈ {0.5, 1, 1.5}); ridge λ applied in raw
  coordinates (the realistic case) **and** in whitened coordinates (K2 should then be exact if
  affine-invariant);
- (c) **Anisotropy, non-proportional:** group effect in the top-k or bottom-k eigendirections (k = d/10),
  magnitude matched so a linear group probe has equal accuracy (as R3 item 4, calibrated on the TRAIN
  grid only);
- (d) **Class mixtures:** K ∈ {2, 5, 10} classes with class means separated by m ∈ {0, 1, 3, 6}σ,
  class-conditional Mahalanobis (labels known) and class-agnostic;
- (e) **Small-G, unequal-size** designs: n_g ~ lognormal (CV = 1), G ∈ {6, 12, 24}.
For each variant: fraction of configs inside |ΔQ_sim/ΔQ_th − 1| ≤ 0.25 and ≤ 0.10, and Δ-level absolute
error ≤ 0.02. **Failure boundary:** a logistic regression of "fail" (ΔQ error > 0.25) on diagnostics
that are computable on real features — log condition number of T̂, PR/d, excess kurtosis (mean over
whitened coordinates), δ̂_LW, N/d, G, class-separation index — fit on the TRAIN family of variants
(a, b, d) and scored on held-out variants (c, e), reporting AUROC (floor 0.5) and then **applied to the
A1 cells** to say which real cells fall in the predicted-valid region (feeds the A verdict
explanation; does not change it).

### Outputs (`results/t1/C/`)
`raw/shard_*.jsonl` (every config, seed, replicate count, all outputs), `c1_blocks.csv`, `c1_correction.json`
(coefficients, TRAIN CV), `c2_auroc.csv`, `c3_collapse.csv`, `c4_cap.csv`, `c5_variants.csv`,
`c5_failure_boundary.json`, `tests.csv`, figures (error heat-maps per block; collapse plot; cap-tightness
plot), `DEVIATIONS.md`, `REPORT.md`.

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| C-i (closed form, in R₀) | **uncorrected** K2: median abs(ΔQ_sim/ΔQ_th − 1) ≤ 0.10 and 90th percentile ≤ 0.25 on TEST-interp **and** on TEST-extrap restricted to R₀ |
| C-ii (cap) | no S8 violation; tightness reported |
| C-iii (collapse) | collapse score ≥ 0.90 in the interior region |
| C-iv (robustness) | variants (b)-proportional with whitened ridge: ≥ 80 % of configs within 0.25; stated separately for (a), (c), (d), (e) |
| Verdict | **GO**: C-i and C-ii (core); **PARTIAL**: exactly one of C-i/C-ii; **NO-GO**: neither — then Idea 2 has no theoretical backbone beyond K1 and the consolidation says so. C-iii and C-iv are reported as "supported / not supported" per claim and never promoted |

**Interpretation limits of the cap K3 (binding wording in `results/t1/C/REPORT.md`):** (1) Failing to find a
violation of K3 is **not** evidence that K3 is useful or true: C-ii "pass" means only "no counterexample found
in the configurations tried". (2) Always report the tightness E[Δ]/cap per configuration and the region
{cap < 0.45} in which the cap is non-vacuous (and the smallest G at which cap < 0.1); every statement about K3
is restricted to that region, and if the cap is several times larger than the observed leakage there, say that
it is loose. (3) Simulation can only find counterexamples or measure tightness; it cannot prove K3. Any
universal statement about K3 needs a proof written out independently of the simulations, and none is claimed
by this order. (4) K3 is stated for the Gaussian model; the heavy-tailed and anisotropic repeats are reported,
not claimed.

Corrected formulas that improve the held-out error are reported as **post-hoc** (the family is
pre-specified but the claim "corrected formula" requires TEST-block improvement with lower CI bound of
the error reduction > 0; otherwise it is not claimed). **Permitted post-hoc:** one residual plot per
variant; no refit of the family.

---

## D. Label-free overlap fingerprint T (stage 2 synthetic, stage 3 real)

**Question.** Does T separate "eval overlaps fit groups" from "eval disjoint" whenever overlap would
matter, and can the claim be falsified?

**Compute:** GPU nearest-neighbour. Estimate 8–15 GPU-h, cap 24.

**Definition.** T as K5, computed on L2-normalised features (primary) and raw Euclidean (secondary);
m = min(5,000, |eval|) eval points, fit→fit leave-one-out distances for a uniform subsample of 20,000 fit
points against the full fit set. **Primary decision statistic is the raw T with a frozen threshold τ**
(not a p-value). "Flag" := T < τ.

### D1. Synthetic sensitivity/specificity
Configs: 1,000 from the C1 design (seed 401) with d ∈ {32, 128, 512, 1024, 2048} (Mahalanobis with
shrinkage only), ρ ∈ {0.03 … 0.5}, G ∈ {12, 24, 100, 400}, n ∈ {5, 25, 100}; Gaussian and Student-t
(ν = 5) variants. For each config: replicates with ω = 0 (disjoint) and ω ∈ {0.25, 0.5, 1}.
**Δ_full** = true leak at ω = 1 for Mahalanobis (LW) and kNN-k50, OOD O1 c = 2. "Material" := Δ_full ≥ 0.02 for
either scorer (the paper's 0.02 bar). τ is chosen on the TRAIN configs (d ∈ {32, 512, 2048}) as the value
maximising Youden's J for ω = 0 vs ω = 1, then **frozen**; TEST configs d ∈ {128, 1024}. Report on TEST:
sensitivity (flag | ω ≥ 0.25, material), specificity (no flag | ω = 0), AUC of T (ω = 0 vs ω = 1), and the
**calibration of the naive z-score SE** (size when ρ = 0).

### D2. Try to falsify (adversarial search)
Search for a configuration with Δ_full ≥ 0.02 **and** AUC(T: ω = 0 vs ω = 1) < 0.8. Algorithm fixed now:
(i) 3,000 random configs from an extended space (d ∈ [16, 2048] log-uniform, ρ ∈ [0.005, 0.7] log-uniform,
G ∈ [6, 500], n ∈ [3, 300], Student-t ν ∈ [3, ∞), spiked/power-law Σ, class mixtures, unequal n_g,
low-rank u), seed 401; (ii) CMA-ES (population 32, 40 generations, seed 401) minimising AUC(T) subject to
Δ_full ≥ 0.02, started from the 20 worst random configs; 16 replicates per evaluation. Report the 10
worst configs with all parameters. A found falsifier is a finding (restrict the claim to the complement
region); the absence of one is **not** a proof.

### D3. Real features, known groups
Cells: all Camelyon backbones and medbench cells. Per cell × fold: T(seen-eval vs fit), T(unseen-eval vs
fit), and mixtures of seen and unseen eval points at ω ∈ {0, .25, .5, .75, 1} (T only — Δ is linear by K1,
so no Δ dose-response is claimed); 50 random subsamples of 5,000 eval points each. AUC(T_seen vs T_unseen)
per backbone; correlation of T_unseen with Δ_meas(Maha), Δ_meas(kNN) (LOBO as A1 vs ICC-only baseline).

### D4. Null on real features
Random split of the fit set **ignoring groups** into pseudo-fit/pseudo-eval (eval is exchangeable with fit
including group overlap): T distribution (reference for "overlap present"); and **permuted group labels**
(Item H2) do not change T (label-free) — recorded as an implementation check. If H2 is `not run` (S11, or A1 not available) this check is `not run`; it is an implementation check, not a D sub-criterion, and has no effect on D-i … D-iii or on the D verdict.

### D5. Dependence-aware SE (diagnostic, post-hoc)
With known groups, group-cluster bootstrap SE of the mean NN² difference vs the naive SE; ratio per cell.
Used only to say how anti-conservative the z-score form is. Output file `d5_se_posthoc.csv` (rule 8); its rows in `tests.csv` have status `post-hoc`.

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| D-i (synthetic TEST) | sensitivity ≥ 0.90 and specificity ≥ 0.90 on material configs |
| D-ii (falsification) | D2 finds no configuration with Δ_full ≥ 0.02 and AUC(T) < 0.8 |
| D-iii (real) | AUC(T_seen vs T_unseen) ≥ 0.90 in at least 11 of 13 Camelyon backbones |
| Verdict | **GO** all three; **PARTIAL** D-i and D-iii but D-ii finds a falsifier (claim restricted to the complement region, listed); **NO-GO** D-i or D-iii fails |

Outputs `results/t1/D/`: `d1_configs.csv`, `d1_tau.json` (frozen τ and the training sets used), `d2_search.csv`,
`d2_worst10.csv`, `d3_real.csv`, `d4_null.csv`, `d5_se_posthoc.csv`, `tests.csv`, `DEVIATIONS.md`, `REPORT.md`.
**Permitted post-hoc:** ROC curves of T per backbone; T with PR-matched distances. Seeds 401.

---

## E. Idea 1 — clustered time-uniform confidence sequences and label-free group recovery (stage 3; CPU, GPU optional)

**Question.** Is a non-asymptotic group-level betting CS valid at every ICC, is the sample-level CS invalid, what
does certification cost, and when can groups be recovered without labels? **Two regimes are kept apart
throughout:** (a) **non-informative cluster size** — group size n_g independent of the group's mean m_g — which is
the clean test of the within-group dependence effect; (b) **informative cluster size** — n_g correlated with m_g —
in which the sample-level and group-level methods target different estimands and each is judged against its own.
Results of regime (b) are reported separately and are never mixed into the dependence conclusion (E-i is the only
criterion that reads regime (b), and only for G1 against θ_group).

**Compute:** vectorised torch (GPU) or CPU. Estimate 1–3 GPU-h (or ≈ 200 CPU core-h), cap 6. Regime (b) adds
≈ 25 % more E1 cells than v1 and is processed **after** every regime-(a) cell (rule 16 ordering is applied
within each regime; the cap can never truncate regime (a) in favour of (b)).

**Implementation.** Predictable-plug-in empirical-Bernstein betting CS of Waudby-Smith & Ramdas
(arXiv 2010.09686; use `confseq` if installed, else implement from the paper over a 1,001-point grid for
the mean in [0,1]; record which). **Unit test before use:** i.i.d. Bernoulli(0.3) stream, 10,000
replicates, horizon 2,000: ever-miscover ≤ α + 3·SE (SE = √(α(1−α)/10,000)). Fail → fix the implementation
(not the criterion) and report.

**Estimands (stated per method, fixed now).** Let each group g have a true mean m_g and size n_g.
- **θ_group = E[m_g]** — the mean over the population of future groups (unweighted by size). Targeted by the
  group-level methods **N3** and **G1** (unit = group mean m̂_g, t = number of groups; E[m̂_g] = θ_group for any
  size mechanism because E[m̂_g | m_g, n_g] = m_g).
- **θ_size = E[n_g m_g] / E[n_g]** — the size-weighted mean (the population mean of a randomly chosen
  *sample*). Targeted by the sample-level methods **N1** and **N2** (pooled stream of samples).
- θ_size = θ_group **exactly when n_g is independent of m_g** (regime (a), including equal sizes, where they are
  equal for every ρ and every size distribution). They differ in regime (b); the gap |θ_size − θ_group| and its
  ratio to ε are recorded per cell.
- True values: regime (a) analytic (= θ); regime (b) θ_group analytic, θ_size by a 10⁷-draw Monte-Carlo of
  (m_g, n_g) pairs (seed 501; its SE is recorded and must be < 10⁻⁴, else the cell is `not run`).

### E1. Simulated streams (seed 501; 10,000 replicates per cell; α = 0.05; horizon 2,000 groups)
Common: group mean m_g ~ Beta with ICC ρ, flags Bernoulli(m_g) (θ ∈ {0.3, 0.05}) or a continuous Beta score;
ε ∈ {0.05, 0.10}; methods **N1** sample-level betting CS; **N2** N1 with radius × √(1+(n̄−1)ρ̂_t) (running ANOVA ρ̂,
a practical design-effect baseline); **N3** group-level asymptotic CS (as `s1b`); **G1** group-level betting CS
on group means (unit = group, t = number of groups). DEFF = 1+(n̄−1)ρ with n̄ the nominal mean group size.

**Regime (a) — non-informative cluster size (v1 grid, unchanged).** n ∈ {5, 20, 50, 200}; ρ ∈ {0, 0.02, 0.05,
0.1, 0.2, 0.4}; flags (θ ∈ {0.3, 0.05}) and a continuous Beta score; group size constant / 1+Poisson(n−1) /
lognormal (CV = 1), **drawn independently of m_g**. Here θ_size = θ_group = θ; each method is judged against θ
(which is also the estimand it targets). The constant-size cells are the cleanest dependence test and are
tabulated separately (report only); the unequal-size cells are included in every pooled (a) count.

**Regime (b) — informative cluster size (new cells).** n ∈ {5, 20, 50} (nominal E[n_g]); ρ ∈ {0.05, 0.2};
outcomes: flags θ ∈ {0.3, 0.05} and the continuous Beta score; three size mechanisms:
- (b1) **shared latent**, Gaussian copula between (m_g, n_g) with correlation κ = +0.6; (b2) same with κ = −0.6;
  size margin lognormal (CV = 1) with mean n, m_g margin as above;
- (b3) **size depends on m_g**: n_g = clip(round(n · m_g/θ_m), 1, 10 n), θ_m = E[m_g] (size ∝ group mean).
(54 cells.) Every method is run, and **both estimands are scored for every method**.

**Metrics (every cell, every method, both estimands).** Ever-miscover rate for θ_group **and** for θ_size
(columns `miscover_theta_group`, `miscover_theta_size`, plus `targeted` ∈ {θ_group, θ_size} naming the method's own
estimand); P(false certificate) = P(a certificate is issued and |mean − θ| > ε) and its conditional version, for
both estimands; expected groups to certify (first t with radius ≤ ε) and its ratio to the i.i.d. requirement vs
DEFF; average width at 100 groups. Regime (a): the two estimand columns are identical by construction and
are both written. Regime (b): the estimand gap |θ_size − θ_group|/ε is written.

### E2. Label-free group recovery (simulation)
G = 60 groups, n = 15 (and n = 5, 50), feature ICC ρ_f ∈ {0.05, 0.1, 0.2, 0.3, 0.6}, d ∈ {4, 16, 64, 256,
1024}, score ICC 0.2, 500 replicates (equal group sizes, so the estimands coincide; regime (a) by
construction). Clusterers fixed now: **M1** average-linkage cut at the midpoint of
the median nearest-neighbour and median pairwise distances (as `s1c`); **M2** k-means with k by maximum
silhouette over k ∈ {2…N/3}; **M3** oracle labels. Metrics: pair precision/recall, ARI (floor 0), k̂/k,
fixed-time 95 % CI coverage of the score mean with cluster-robust SE on estimated clusters (CR1) vs naive
vs oracle, and **silent-failure rate** := k̂ ≥ 2k and coverage < 0.90. Lemma prediction tested: recovery
(coverage ≥ 0.90) iff ρ_f√(d/2) ≥ 3, G ≥ 60.

### E3. Real features, known groups
Camelyon (13 backbones; fit-set patches, 4,000 patches per fold stratified over the 15 slides of the fold =
5 slides per hospital × hospitals {0, 3, 4}; requires `MANIFEST_OK`) and medbench
cells: M1/M2 pair precision/recall, ARI, k̂ vs true; ρ_f (ANOVA) and ρ_f√(PR/2), ρ_f√(d/2); agreement
between the lemma's predicted recoverability and the observed one.

### E4. Real-ICC-matched streams (semi-synthetic; requires `MANIFEST_OK`)
Per Camelyon backbone, per-slide flag rates m_g = fraction of unseen-eval patches of slide g below the
5th percentile of the fit-LOGO scores (slide g is "unseen" in the one `paper_2fold` fold whose fit set excludes it;
the 30 slides = 10 per hospital × hospitals {0, 3, 4}; their empirical m_g distribution is the superpopulation).
Groups are drawn i.i.d. with replacement from the 30 slides; patch flags in a drawn group are Bernoulli(m_g).
Methods **N1, N2, G1**. Three size regimes, both estimands scored for each method as in E1:
- **(a) equal sizes:** n_g ∈ {20, 100} for every group. θ_size = θ_group = the unweighted mean of the 30 m_g.
- **(a′) non-informative unequal sizes:** n_g drawn i.i.d. from the empirical distribution of the 30 slides'
  real unseen-eval patch counts N_g (capped at 1,000), **independently of the slide drawn** (sizes decoupled from
  m_g). θ_size = θ_group in expectation.
- **(b) informative sizes:** n_g = min(N_g, 1,000) of the **drawn slide itself** (per-slide flag rate and
  patch-count dependence on the slide both retained). θ_group = (1/30) Σ_g m_g; θ_size = Σ_g n_g m_g / Σ_g n_g
  over the 30 slides (exact). Also report, per backbone, the Spearman correlation of (m_g, N_g) over the 30 slides
  with a slide-bootstrap CI (B = 2,000, `default_rng(501)`) — descriptive, never used as a gate.
Regime (b) is reported separately from (a)/(a′) in every table and in the REPORT.

### Outputs `results/t1/E/`
`unit_test.json`, `e1_cs.csv` (columns include `regime`, `size_mechanism`, `theta_group`, `theta_size`,
`theta_size_se`, `targeted`, `miscover_theta_group`, `miscover_theta_size`), `e1_estimands.csv`, `e2_recovery.csv`,
`e3_real_recovery.csv`, `e4_semisynth.csv` (same estimand columns, `regime ∈ {a, a', b}`), `tests.csv`,
`DEVIATIONS.md`, `REPORT.md`.

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| E-i (validity of G1 for its own estimand) | G1 ever-miscover **of θ_group** ≤ 0.05 + 3·SE in **every** E1 cell of regime (a) **and** every E1 cell of regime (b), and in every E4 backbone **in each of the regimes (a), (a′), (b)**. (G1's estimand does not depend on the size mechanism, so regime (b) is a validity check of G1, not a dependence test; cells of the two regimes are listed separately in `tests.csv`) |
| E-ii (naive fails; judged in regime (a) only) | N1 ever-miscover > 0.05 + 3·SE, **judged against the estimand N1 targets (θ_size, which equals θ_group = θ in regime (a))**, in at least 90 % of **regime-(a)** E1 cells with DEFF ≥ 2. Regime (b) cells and regime-(b) E4 are never counted here. Report separately (report-only, same 90 % reading): the equal-size subset of regime (a); N2 (if N2 is valid it is the practical baseline and Idea 1's added value is stated against it) |
| E-iii (label-free) | in E2, no configuration with ρ_f√(d/2) ≥ 3 and G ≥ 60 has M1 cluster-robust coverage < 0.90; and in E3 the lemma's predicted recoverability matches observed in ≥ 80 % of cells |
| Verdict | **GO**: E-i and E-iii; **PARTIAL**: E-i only (Idea 1 = the design-effect corollary, no label-free claim); **NO-GO**: E-i fails (implementation or claim wrong; report) |

**E status words and verdict when a component is `not evaluable` / `not run` (binding; v2.4; rule 18).** *E-i* is `fail` if any
component that was evaluated failed (any E1 regime-(a) or regime-(b) cell, any E4 backbone × regime); otherwise it is `not
evaluable` if any required component is `not run` (E4 under S11 or S2, or an E1 / E4 cell not run for any reason, including cap
truncation), and `pass` only when every component was evaluated and passed. *E-iii* is `fail` if the E2 condition is violated or E3
was evaluated and the match is < 80 %; otherwise `not evaluable` if E3 is `not run` (S11 or S2) or the E2 part is incomplete; `pass`
only when both parts were evaluated and passed. **Verdict:** GO = E-i and E-iii both passed; PARTIAL = E-i passed and E-iii failed or
`not evaluable`; NO-GO = E-i failed (an evaluated failure, whatever E-iii is); **INCONCLUSIVE** = E-i `not evaluable` (whatever
E-iii is). In particular, when `MANIFEST_OK` is false (S11), E3 and E4 are `not run`, E-i and E-iii are `not evaluable` (unless an
E1 cell has already failed, which makes E-i `fail` and E NO-GO), E is INCONCLUSIVE and cannot be GO or PARTIAL; INCONCLUSIVE is
never a pass.

E-ii decides only the wording ("naive fails" supported or not), and only for the within-group dependence effect
under non-informative sizes. **Regime (b) results (coverage of θ_group and of θ_size for N1/N2/N3/G1, the
estimand-mismatch effect) are reported as a separate table and never feed E-ii or the dependence conclusion;** if
N1 under-covers θ_group in regime (b), the report says that this mixes dependence with estimand mismatch.
**Permitted post-hoc:** certification-cost vs DEFF regressions; recovery vs PR. Seeds 501.

---

## F. Idea 3 — blindness lemmas on real frozen features (stage 3; CPU + corruption extraction on GPU)

**Question.** On real features, how often do the first-order sign and the Edgeworth-shifted blind surface
agree with the observed AUROC sign, and do blind-but-detectable shifts exist?

**Compute:** CPU for F1–F3; GPU for F4. Estimate 6–12 GPU-h, cap 20.

### F1. Reproduction of the Gaussian reference (CPU)
Re-implement the quadratic-scorer sign law: 240 random Gaussian shifts at d = 8 and 32, 96 at d = 128
(random covariance spectrum and mean), scorers Mahalanobis (A = Σ⁻¹), Euclid (A = I), ViM-residual projector.
Reference (bundle `s3_out.txt`, `s3_d128_out.txt`): sign accuracy .96–.99 overall, corr(AUROC, Φ(z)) ≥ .995.
Reproduce Edgeworth (`s3c`: mean AUROC on trace-balanced shifts .489/.491/.495 at d = 8/32/128 vs
CF-predicted .486/.490/.495) and the two-block family (`s3b`). Tolerance ±0.02 on sign accuracy, ±0.005 on
the balanced means. Fail → the F items stop (not S3).

### F2. Real cells, natural shifts
ID/OOD feature pairs (disjoint-group ID eval vs OOD; OOD positive): all paper-3 caches (including
`runA_grl_*` and `runB_orth1_*`, the adversarially trained cells), every Camelyon backbone (hospital 2) and
every medbench cell, plus **class-split shifts** (leave-one-class-out on DermaMNIST, ISIC 2019, Kermany v2,
BreakHis, and the paper-3 label sets; fit on the remaining classes by group-disjoint folds, or random folds
labelled as such if no groups). Scorers: A = fitted Σ̂⁻¹ (MLE if N ≥ 20d, else LW), I, ViM-residual
projector. Per cell: ⟨A,ΔM⟩, z_A (Gaussian moment formula with empirical Σ̂, C, m), measured AUROC,
mean-score gap, median-score gap, skewness of ID and OOD scores.
Unit test: sign⟨A,ΔM⟩ equals the sign of the mean-score gap for A = fitted Σ̂⁻¹, I (identity to 1e-9) —
if not, the implementation is wrong.
Metrics (units = cells; inversions counted per independent shift unit, where cells sharing ID data and
OOD set count as one unit): sensitivity and specificity of "predicts inverted/blind" vs the baseline
"never predict inversion" (floor), disagreement rate between sign(mean gap) and sign(AUROC−½) as a
function of score skewness, and agreement of Φ(z_A) with the measured AUROC (MAE).

### F3. Constructed trace-balanced shifts on real features (CPU)
For each cell: split the group-disjoint ID eval features into halves (E₁ = ID, E₂ → shifted). Eigenbasis of
the fit covariance; scale the top-half-variance directions by a ∈ {1.2, 1.5, 2.0} and the bottom half by
b solved (bisection) so that the mean scorer-A score of E₂′ equals that of E₁ (trace-balanced for A =
Σ̂⁻¹; separately for A = I). Measure: AUROC of Mahalanobis (E₂′ vs E₁), of the half-variance-contrast
statistic log(‖P_top x‖²/‖P_bot x‖²) (fixed, fit-free given the eigenbasis), and of a cross-validated
logistic regression on the second-order features (supervised upper reference). Prediction fixed now: the
sign of AUROC_Maha − ½ equals the sign of −κ₃ (K7, empirical moments) — evaluated against two baselines:
the coin flip (exact binomial test, floor 0.5) and sign of the difference in empirical score skewness.
"Blind but detectable": Maha AUROC ∈ [0.45, 0.55] **and** contrast AUROC ≥ 0.90.

### F4. Corruption shifts (GPU; stage 4)
Backbones: resnet50 (trained, seed 42), uni, dinov2_vitb14 (Camelyon ID val patches, 20,000 patches
sampled with `SeedSequence([20261010, 6, …])`, hospital-matched). Corruptions: Gaussian noise, Gaussian
blur, JPEG, brightness, each at 3 severities (ImageNet-C-style parameters taken from `imagecorruptions` if
installed; record the parameter table). Extract features of corrupted patches, then repeat F2 with
corrupted vs clean ID eval. Cap 15 GPU-h; if the measured extraction rate gives more, reduce patches
(rule 16 order: 20,000 → 10,000 → 5,000), never the corruption list.

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| F-i (reference) | F1 reproduced within tolerance |
| F-ii (sign law, real) | **power:** ≥ 30 independent inversion units exist in F2 (else the criterion is "not evaluable" and F2 is reported descriptively). If evaluable: sensitivity of the first-order sign ≥ 0.80 with lower 95 % Wilson bound > 0.65, and specificity ≥ 0.90 |
| F-iii (Edgeworth + balanced) | F3: sign accuracy of the Edgeworth prediction ≥ 0.70 with lower Wilson bound > 0.5 on ≥ 30 non-degenerate balanced cells (abs(AUROC−½) ≥ 0.002), **and** better than the skewness-sign baseline by a paired sign test p < 0.05 (single pre-specified test) |
| F-iv (blind-but-detectable) | ≥ 80 % of backbones have at least one (a, scorer) cell that is blind-but-detectable |
| Verdict | **Idea 3 standalone survives** only if F-i, F-ii, F-iii, F-iv all pass. **Lemma-level only** if F-i and F-iv pass and F-ii/F-iii fail or are not evaluable. **Dropped** if F-iv fails. The prior recommendation (Idea 3 as lemmas, not a paper) stands unless all four pass |

**Permitted post-hoc:** the relation between disagreements and score skew/kurtosis; Gaussian-moment vs
empirical-skew Edgeworth. Seeds 601. Outputs `results/t1/F/`: `f1_ref.csv`, `f2_real.csv`, `f3_balanced.csv`,
`f4_corruption.csv`, `tests.csv`, `DEVIATIONS.md`, `REPORT.md`.

---

## G. Controlled group overlap with real fine-tuning (stage 4; G0 and G1 unless G0 aborts or S2 / S3 stops G; G2 only if A = GO)

**Question.** Do the scorer-channel predictions hold when the features themselves are produced with a
controlled amount of group overlap, and how large is the separate backbone channel? (R3 item 5b noted that
cross-fitting removes only the scorer channel.)

**Warning stated once:** a dose–response in the *mixing* fraction ω of seen vs unseen eval points for a
fixed scorer is linear by K1 and is not run as evidence. The dose here is the **number of eval-group
blocks included in the scorer fit set (ω_s) and in the backbone fine-tuning set (ω_b)**, which changes G,
N, ρ̂ and the features — a non-trivial prediction of K2 for the scorer channel.

**Slide design (Camelyon17: the 30 training slides = 3 source hospitals {0, 3, 4} × 10 slides).** Hospital 2 is
the OOD split and hospital 1 is unused; **neither appears in any slide set below, in any fit, in any fine-tune,
in any checkpoint selection or in any calibration** (G0 enforces it). The 30 slides are partitioned into three
disjoint sets, with these exact counts **per hospital**:

| hospital | train slides | U (never fit) | T (always fit) | E₁ | E₂ | E₃ | E₄ | check |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 2 | 4 | 1 | 1 | 1 | 1 | 2 + 4 + 4 = 10 |
| 3 | 10 | 2 | 4 | 1 | 1 | 1 | 1 | 2 + 4 + 4 = 10 |
| 4 | 10 | 2 | 4 | 1 | 1 | 1 | 1 | 2 + 4 + 4 = 10 |
| **total** | **30** | **6** | **12** | **3** | **3** | **3** | **3** | 6 + 12 + 12 = 30 |
| 1 (`val`) | — | not used | | | | | | |
| 2 (`test`) | — | OOD only (never in any slide set) | | | | | | |

E = E₁ ∪ E₂ ∪ E₃ ∪ E₄ = 12 slides; **each block E_j holds exactly 3 slides, one from each of hospitals 0, 3, 4**,
so the four blocks are equal in slide count and balanced over the source hospitals. U, T, E are pairwise disjoint
and cover the 30 slides.

**Assignment (the algorithm and seed are pre-registered; the realised slide lists are written at G0 and hashed).**
`rng = numpy.random.default_rng(SeedSequence([20261010, 7, 0]))`; for h in (0, 3, 4): sort the hospital's 10
slide IDs ascending as strings, `p = rng.permutation(slides)`; U ← p[0:2]; T ← p[2:6]; E_j ← p[5+j] for j = 1..4
(i.e. p[6], p[7], p[8], p[9]). Slides are never moved after G0.

**Held-out ID-eval patches.** For each slide of E and U a fixed 30 % of its training patches (round(0.3·n_s),
drawn without replacement with `SeedSequence([20261010, 7, 1, stable_hash(slide)])`) is **held out as ID-eval**
(never in any fit or fine-tune); the remaining 70 % of an E slide is "fit-eligible". T slides contribute all
patches to every fit and no ID-eval patches.

**Doses** (dose k ∈ {0, 1, 2, 4} blocks ↔ ω ∈ {0, .25, .5, 1}; ω_s for the scorer-fit set, ω_b for the backbone
fine-tuning set). Fit set at dose k = all patches of T ∪ fit-eligible patches of E₁…E_k.

| k | ω | blocks in fit | slides in fit (hospitals 0 / 3 / 4) | G_fit | "seen" eval = held-out of | "unseen" eval = held-out of |
|---|---|---|---|---|---|---|
| 0 | 0 | none | 4 / 4 / 4 | 12 | placebo: E₁ ∪ E₂ (both unseen; 6 slides) | U (6 slides) |
| 1 | .25 | E₁ | 5 / 5 / 5 | 15 | E₁ (3 slides) | U |
| 2 | .5 | E₁, E₂ | 6 / 6 / 6 | 18 | E₁ ∪ E₂ (6 slides) | U |
| 4 | 1 | E₁–E₄ | 8 / 8 / 8 | 24 | E₁ ∪ … ∪ E₄ (12 slides) | U |

(The placebo set of k = 0 and the seen set of k = 2 are the same 6 slides, so their difference is purely a change
in fit membership. **This is the only pair of doses that shares its evaluation slides.**) The slide-count dose k/4
differs from the patch-weighted fraction of E held-out patches in the seen blocks because slides have very unequal
patch counts (documented 1,430–55,149); G0 records the realised patch-weighted ω per block set and the per-block
patch totals (report-only; no criterion uses them).

**Primary estimand (fixed now).** OOD = hospital 2, as in R3. Measure
**Δ_G(k) = AUROC(held-out of E₁…E_k vs OOD) − AUROC(held-out of U vs OOD)**; the placebo for k = 0 is E₁∪E₂
(both unseen). Each AUROC is a **patch-level pooled AUROC**: all held-out ID-eval patches of the seen slides (resp.
of the unseen slides U) are pooled into one sample and compared with all OOD (hospital-2) patches, one patch one
vote. This is the estimand kept for comparability with R3. **Large slides weigh more:** a slide's weight in Δ_G is
proportional to its number of held-out patches, so slides with tens of thousands of patches dominate the pooled
AUROC of their set. CI: bootstrap over eval slides and OOD images (B = 2,000, `default_rng(2)`; "OOD images" are the OOD patches, exactly as
in the R3 helper). Per R3 item 6, **Δ_G(0) must have a CI containing 0** (S5: else stop for that model × scorer).

**Scope of the Δ_G confidence intervals (binding; the paragraph is copied verbatim into `REPORT.md` of G, next to every
table or figure that shows a CI of Δ_G or Δ_G^sw).** The bootstrap resamples the eval slides (within the seen set and
within U) and the individual OOD patches, as in the R3 helper; **the bootstrap is not changed in this order.** The
85,054 OOD patches come from only 9 patients, and resampling at the patch level does not model the dependence between
patches of the same patient. Therefore these CIs **must not be interpreted as uncertainty about generalising to new OOD
patients or new hospitals**; they describe resampling variability of the eval slides and of these OOD patches only. No
statement in any G report may be stronger than this design allows. Uncertainty at the patient level (or hospital level)
would be a separate inference design and is **not added here**.

**Block patch totals and the primary allocation (note).** Because the primary metric is patch-level, the patch total
of each block E_j (G0, report-only) and of U determines how much each slide counts in Δ_G(k); a block that contains a
very large slide can dominate the seen set of every dose that includes it. The primary allocation stays **random by
the registered seed**; it is not balanced, re-drawn or otherwise changed before or after any result (any other
allocation would be a new order).

**Slide-equal-weight sensitivity analysis (pre-defined; REPORT-ONLY).** Computed for **every cell** (every
backbone × head × scorer × dose k ∈ {0, 1, 2, 4} × seed × channel for which the primary Δ_G is computed), reported
beside the primary in `g_sensitivity_slide_equal.csv` and in `REPORT.md`. Exactly one definition:
- **Per-slide-weighted pooled AUROC.** AUROC_sw(S) = Σ_{i∈S} w_i · [Σ_{j∈OOD} (1[s_i > s_j] + ½·1[s_i = s_j])] /
  (n_OOD · Σ_{i∈S} w_i), where s is the same score and the same orientation and tie convention as the primary, a held-out
  patch i of slide σ(i) has weight w_i = 1/n_{σ(i)} with n_σ = number of held-out ID-eval patches of that slide (so
  every slide in S has total weight 1), and each OOD patch has weight 1 (OOD is not reweighted; it is a single
  hospital with 9 patients). Δ_G^sw(k) = AUROC_sw(seen_k) − AUROC_sw(U); the k = 0 placebo uses E₁∪E₂ as in the primary.
  (Per-slide AUROCs averaged over slides are **not** computed; this is the single sensitivity definition.)
- **CI.** Bootstrap over eval slides and OOD images (= OOD patches, as in the R3 helper), B = 2,000, `default_rng(2)`,
  using **the same replicate draws as the primary** (slides resampled with replacement within the seen set and within U;
  OOD patches resampled with replacement; weights recomputed on each resampled slide multiset), so primary and
  sensitivity are paired and the sensitivity consumes no extra random numbers. **The same binding scope statement applies
  ("Scope of the Δ_G confidence intervals" above):** patch-level resampling of the 85,054 OOD patches from 9 patients
  does not model within-patient dependence, so this CI is not uncertainty about new OOD patients or hospitals; the
  bootstrap is not changed.
- **Disagreement flags**, computed for every cell: **sign flag** = sign(Δ_G^sw) ≠ sign(Δ_G); **magnitude flag** =
  |Δ_G^sw − Δ_G| > 0.5·|Δ_G|. **sign() convention (binding; documented in the code spec and implemented by an explicit
  function `sign0` in `scripts/t1/common.py`, not by the default behaviour of a library):** sign0(x) = +1 if x > 0, −1 if
  x < 0, 0 if x == 0 exactly (float64 comparison, no tolerance and no rounding). The sign flag is raised if and only if
  sign0(Δ_G^sw) ≠ sign0(Δ_G), **including the case in which exactly one of the two is exactly 0**; it is not raised when
  both are exactly 0. (With the strict inequality of the magnitude flag, Δ_G = 0 exactly raises the magnitude flag iff
  Δ_G^sw ≠ 0.) The convention adds no threshold. No floor and no other threshold is added; k = 0 cells (Δ ≈ 0 by construction) are flagged
  mechanically and labelled "placebo, expected noise-dominated" in the report. **Any sign or magnitude disagreement
  is reported as a finding** (counts and a table of the flagged cells).
- **Status: it never enters G-0, G-i or G-ii, never changes a verdict, never changes a threshold, and must not be
  used to rescue the core hypothesis if the primary criteria fail.** If G-i or G-ii fails on the primary, a pass of
  the sensitivity analysis is not cited as support. It is not a row of any pass/fail table; its numbers are
  reported, not evaluated.

**Doses are reported in both units (fixed now).** For every dose k and every cell, `g_dose_units.csv` and the REPORT
give, side by side:
- **slide units:** k/4 (the pre-registered dose), G_fit (12 / 15 / 18 / 24), number of seen-eval slides;
- **patch units:** N_fit(k) = patches of T plus fit-eligible patches of E₁…E_k, and N_fit(k)/N_fit(4); the number of
  seen-eval (held-out) patches; and the **patch-weighted ω(k) = held-out patches of E₁…E_k ÷ held-out patches of
  E₁…E₄** (the patch-weighted share of the seen eval patches), next to the slide-count ω = k/4.
Both are deterministic functions of `slide_design.csv`, written at G0 before any fit. **Pre-registered rule:** Δ_G(k)
and the predicted/measured scorer-channel Δ are plotted and tabulated against both axes (`dose.png` has both
panels), together with a report-only linear-fit R² of Δ against k/4 and against ω_patch(k); if the conclusions
drawn on the slide-dose axis and on the patch-dose axis differ (for example a shape, ordering, spacing or
agreement that holds on one axis and not on the other), **that difference is reported as a finding**. The allocation
and the doses are never changed after seeing results, and no criterion is evaluated on the patch axis.

**Interpretation of G-ii (binding; this sentence is copied verbatim into `REPORT.md` and into the verdict table).**
*Across doses k ∈ {1, 2, 4} both the fit set and the seen-eval slide set change (seen = E₁; E₁+E₂; E₁…E₄; unseen =
U, fixed); only k = 0 vs k = 2 shares the same eval slides (E₁+E₂), differing in fit membership; therefore G-ii is
a test of predictive accuracy under joint change of fit set and evaluation set, not a pure measurement of the fit
effect.* A stronger claim about the fit effect alone would require a fixed evaluation set across doses; that is a
design change **not made in this order** and must not be added after registration; it is recorded only as a
possible follow-up order.

### G0. Manifest verification (CPU, ≤ 0.25 h; runs first; precondition of G1 and G2)
1. Read the **real** manifest on HPC: `load_camelyon_metadata` (the R3 helper used in `scripts/r3/item6_dose.py`)
   for slide → hospital and per-slide train / id_val patch counts, and `groups_train` of each of the five FM
   caches (`~/r3work/item1/inputs/camelyon_<fm>_s42.npz`); write `results/t1/G/manifest_real.csv` (`slide,
   hospital, n_train_patches, n_idval_patches`) and require `results/t1/I/MANIFEST_OK` = true.
2. Build the assignment with the algorithm above; write `results/t1/G/slide_design.csv` (`slide, hospital, set
   ∈ {U, T, E1..E4}, n_train_patches, n_heldout, n_fit_eligible`) and its sha256.
3. **Abort (report and stop) if any of:** the hospitals of the 30 training slides ≠ {0, 3, 4}; slides per hospital
   ≠ (10, 10, 10); total ≠ 30; the realised counts per hospital × set differ from the table above in any cell;
   a slide belongs to more than one set or to none; any of the five FM caches' slide set differs from the
   metadata's; hospital 1 or 2 occurs among the 30 slides or among the patches of any fit / fine-tune /
   selection set.
4. On abort: write `results/t1/G/MANIFEST_MISMATCH.md` (real vs expected counts, per hospital × set), set the G
   verdict to `not run: manifest mismatch (S11)`, stop G1 and G2. Do **not** redesign the split, re-count, or pick
   a "nearest" design on the HPC (rule 2); the redesign, if wanted, is a new order.
5. Recorded, non-aborting (WARN): total train / id_val / OOD patch counts vs the documented 302,436 / 33,560 /
   85,054; the per-slide minimum and maximum; per-block patch totals and the realised patch-weighted ω; the
   per-slide share of held-out patches within each seen set and within U (the weights the primary gives each
   slide); `g_dose_units.csv` (N_fit, G_fit, k/4, patch-weighted ω per dose). All report-only; none aborts.

### G1. Frozen backbones (CPU / GPU-light, ≤ 5 GPU-h; cached features)
Backbones: the 5 FMs (uni, virchow2, dinov2_vitb14, dinov2_vitl14, conch_v1_5), features from the R3
caches. Heads: (a) logistic linear probe (L2, C = 1, standardised, trained on the fit set's labels),
(b) a one-hidden-layer adapter (512 units, ReLU, AdamW lr 1e-3, 20 epochs, seed 42); heads change only
logit-based scorers (ViM with head). Scorers: `mahalanobis_l2`, `knn_mean_cosine`, ViM-with-head — **all three are run
and reported, but G-i uses only `mahalanobis_l2` and `knn_mean_cosine`; ViM-with-head is not part of G-i** (see "G-i scope
and cell set" below).
Report Δ_G(k) for every scorer, and — **for `mahalanobis_l2` only** — the prediction P_Δ (A1 pipeline, item A1 step 6, on
the fit set at dose k, "seen" := held-out patches of E₁…E_k, "unseen" := U) with its Tier-L descriptors. For
`knn_mean_cosine` and ViM-with-head no P_Δ is computed (`NA` in `g_predictions.csv`): items A and B register no predictor
of this form for them (trace below), and none is invented here. **Implementation checks that must hold exactly:**
Maha and kNN Δ identical across the two heads; K1 identity. These frozen cells are also a **negative
control for the backbone channel** (it is zero by construction).

**G-i scope and cell set (binding; traced to the predictors registered in items A and B).**
- **Δ_G(0) sanity check:** 5 FMs × {`mahalanobis_l2`, `knn_mean_cosine`} = **10 combinations**; the CI of Δ_G(0) must
  contain 0 in all 10. Both scorers are head-independent (identical across the two heads by the exact check above), so each
  FM × scorer is evaluated once (on head (a)); heads (a) and (b) are not counted as separate combinations. ViM-with-head
  (head-dependent) is run and its Δ_G(k) reported, but its Δ_G(0) is not one of the 10 and never enters G-i; S5 still
  applies to it as a stop for that model × scorer.
- **Trace of the registered predictors.** Item A1 registers one primary P_Δ (step 6): the additive map applied to
  ΔQ_pred, which is the closed form for the **Mahalanobis** squared distance (steps 3–5, ρ̂_w, s_logo, `mahalanobis_l2`;
  its Δ_meas is the `mahalanobis_l2` Δ, step 7); its zero-parameter version is a = 0, b = 1. Item A registers no
  predictor for kNN or ViM. Item B registers, for kNN, x₁ = ρ̂_cos√PR (isotonic, LOBO-calibrated) and the matched toy
  simulator MTS, defined per `paper_2fold` cell × fold; G1 specifies "the A1 pipeline" and G-i names neither, so neither is
  applied to G cells. No registered predictor exists for ViM-with-head in A or B. **No predictor is extended to another
  scorer.**
- **Exact cell set of the median relative error of P_Δ in G-i.** A cell enters iff **all** hold: scorer =
  `mahalanobis_l2`; backbone ∈ the 5 FMs (frozen features, R3 caches); dose k ∈ {1, 2, 4} (k = 0 is the placebo and is
  excluded); the measured primary patch-level Δ_G(FM, `mahalanobis_l2`, k) ≥ 0.02. Head: none (Mahalanobis does not use a
  head; identical across heads), so each (FM, k) counts **once** (counting both heads would duplicate every value and could
  not change the median). Predictor: the item-A1 primary additive-map P_Δ, **zero-parameter** (a = 0, b = 1), not any
  LOBO-calibrated version, not the multiplicative or Gaussian-CLT map, not any variant. **Relative error of an entering cell, defined exactly:**
  e_rel(FM, k) = | P_Δ(FM, k) / Δ_G(FM, k) − 1 |, where P_Δ(FM, k) is the zero-parameter item-A1 primary additive-map prediction and
  Δ_G(FM, k) is the measured primary patch-level Δ_G of `mahalanobis_l2` at dose k. Because only cells with Δ_G(FM, k) ≥ 0.02
  enter, the denominator is always > 0: **no epsilon, no clipping and no other form** (it is the form A1 uses at the ΔQ level,
  applied at the Δ level). The G-i statistic is the **median over the entering cells of e_rel**; check c3 passes when it is ≤ 0.5.
  At most 5 × 3 = 15 cells; the cells that enter are marked `in_gi_median = true` in `g_predictions.csv` and listed in `REPORT.md`.
- **Empty or incomplete cell set (binding).** If no cell enters (no `mahalanobis_l2` cell with k ∈ {1, 2, 4} has measured
  Δ_G ≥ 0.02), the median is undefined and c3 is `not evaluable under the pre-registered criterion` (`pass = na`): never a pass,
  never replaced (no other scorer, k, threshold, predictor or cell is substituted). If any of the 15 values Δ_G(FM, `mahalanobis_l2`, k)
  was not computed (`not run`), it is unknown whether that cell enters; the median is **not** computed over the remaining cells and
  c3 is `not evaluable` (unless the cell set is complete).
- **G-i status (binding).** G-i has three mandatory checks: **c1** the implementation checks (Maha and kNN Δ identical across the
  two heads; K1 identity); **c2** the Δ_G(0) CI contains 0 in all 10 combinations; **c3** median e_rel ≤ 0.5. G-i is `fail` if any
  of c1, c2, c3 was evaluated and failed (c3 evaluated and failed = the median is defined and > 0.5); otherwise `not evaluable`
  (`pass = na`) if any of c1, c2, c3 could not be evaluated; otherwise `pass`. `fail` takes precedence over `not evaluable`.
  A `not evaluable` G-i is never a pass.
- **Excluded from the median (listed, not replaced):** every `knn_mean_cosine` cell (no registered zero-parameter P_Δ in A;
  B's MTS / x₁ not applied by G1/G-i); every ViM-with-head cell (no registered predictor in A or B); every k = 0 cell;
  every cell with Δ_G(k) < 0.02. kNN enters G-i only through the Δ_G(0) check and the implementation checks.

### G2. Full fine-tuning (GPU; stage 4)
Backbones: resnet50, convnext_tiny. Fine-tuning set = T ∪ fit-eligible E₁…E_k (ω_b). Recipe, epochs,
optimiser, augmentations, ID-accuracy gate: **copied verbatim from the isbi_patch2 retrain arm**
(`scripts/r3/item5_b_isbi_patch2.py`, `decisions/precommit_isbi_patch2_2026-10-06.md`); record ID accuracy.
The recipe's "same fold's slides" for checkpoint selection and the ID-accuracy gate maps to **the `id_val`
patches of the slides in the fit set (T ∪ E₁…E_k) only**; never `id_val` or held-out patches of U or of the E
blocks not in the fit set, never the held-out ID-eval patches, never hospital 1 or 2.
Seeds 42, 43, 44. Runs: 2 backbones × 4 doses × 3 seeds = **24 fine-tunes**. After each run extract features
for every fit-eligible and held-out patch and OOD (GPU). Then, per model, the factorial of scorer doses
ω_s ∈ {0, .25, .5, 1} (CPU/GPU-light): Maha and kNN fit on T ∪ E₁…E_{k_s}. Channels:
- **scorer channel:** Δ_G(ω_b = 0, ω_s) − Δ_G(0,0) (backbone never saw E, scorer did);
- **backbone channel:** Δ_G(ω_b, ω_s = 0) − Δ_G(0,0);
- **both:** Δ_G(ω_b, ω_s = ω_b).
Predictions: P_Δ on each model's own features per (ω_b, ω_s) (ρ̂_w, s_logo recomputed for that model and
that fit set) for the scorer channel; **no prediction is made for the backbone channel** (exploratory,
reported as such). Cost: record the measured GPU-h per fine-tune from the first run; planning guess 1–3
GPU-h each (a guess, replace it). If the first-run measurement implies > 80 GPU-h for the 24 runs, apply
the pre-declared reduction in this order: drop seed 44, then seed 43, then convnext_tiny; never reduce
doses. **Consequence for G-ii (binding, see "G-ii under the reduction ladder" below):** the full set is 2 backbones × 3
seeds = 6 model instances; every ladder step leaves fewer than 5 (4, 2, 1), and then G-ii is `not evaluable under the
pre-registered criterion`; the threshold is not changed.

**Run conditions of G (binding wording; the experimental flow is unchanged).**
1. **G0 runs first and must pass** (no S11). If G0 aborts: G is `not run: manifest mismatch (S11)`, G1 and G2 are `not run`, and G-i
   and G-ii are not evaluated. If S2 or S3 stops G, G is `not run: S2` / `not run: S3`.
2. **G1 runs whenever G0 passed, whatever the verdict of A, and G-i is evaluated from it.**
3. **G2 runs only if A = GO.** If A ≠ GO (A is PARTIAL, NO-GO or INCONCLUSIVE; A `not run` means S2 / S3 and then rule 1 applies), G2
   does not run and **G-ii = `not run`** (no G2 instance exists; `pass = na`; `not run` is never a pass).
4. **Verdict map (the only mapping).** G-i ∈ {`pass`, `fail`, `not evaluable`}; G-ii ∈ {`pass`, `fail`, `not evaluable`, `not run`}:

| G-i | G-ii | G verdict |
|---|---|---|
| `fail` | any | **NO-GO** |
| `not evaluable` (no mandatory G-i check failed) | any | **INCONCLUSIVE** (not PARTIAL, not GO) |
| `pass` | `pass` | **GO** |
| `pass` | `fail`, `not evaluable` or `not run` | **PARTIAL** |

G can never be GO unless G-ii was evaluated and passed. With A ≠ GO the possible verdicts are therefore PARTIAL (G-i passes),
NO-GO (G-i fails) or INCONCLUSIVE (G-i `not evaluable`), never GO. INCONCLUSIVE is never counted as a pass.

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| G-0 (precondition) | G0 passed (no S11). Not a performance criterion: if G0 aborts, G is `not run: manifest mismatch` and G-i, G-ii are not evaluated |
| G-i (G1, frozen) | **c1** implementation checks exact; **c2** Δ_G(0) CI contains 0 in all 10 combinations 5 FMs × {`mahalanobis_l2`, `knn_mean_cosine`} (ViM-with-head is not part of G-i); **c3** the zero-parameter P_Δ (item A1 step 6, `mahalanobis_l2` only) has **median e_rel ≤ 0.5, e_rel = \|P_Δ/Δ_G − 1\|**, over exactly the cells {5 FMs} × {k ∈ {1, 2, 4}} with measured Δ_G(`mahalanobis_l2`, k) ≥ 0.02 (once per FM × k; no kNN cell, no ViM cell, no k = 0 cell; see "G-i scope and cell set"). **Status:** `fail` if any of c1–c3 was evaluated and failed; else `not evaluable` (`pass = na`; for example c3 with an empty cell set, i.e. no `mahalanobis_l2` cell with Δ_G(k) ≥ 0.02) if any check could not be evaluated; else `pass`. A `not evaluable` G-i is never a pass |
| G-ii (G2, scorer channel) | **Evaluated only if at least 5 of the 6 model instances (2 backbones × 3 seeds) are completed** (see "G-ii under the reduction ladder"); fewer than 5 completed → `not evaluable under the pre-registered criterion` (never a pass; threshold not changed); A ≠ GO → `not run`. Otherwise: Δ_G(0) sanity holds for every completed model instance (with 6 completed: for all 6); predicted vs measured scorer-channel Δ over doses k ∈ {1, 2, 4}: rank agreement (predicted order of the 3 doses = observed order) in ≥ 5 of 6 model instances (an instance that is not completed counts as not agreeing; the denominator stays 6) **and** median zero-parameter relative error ≤ 0.5 over the cells listed in "G-ii cell set". Computed on the **primary patch-level Δ_G** only. **Interpretation (binding):** the sentence "Interpretation of G-ii" above applies verbatim; it is not re-worded here |
| Report-only (not a criterion) | slide-equal-weight Δ_G^sw and its disagreement flags; both dose units and the slide-axis vs patch-axis comparison. Never enters G-0, G-i, G-ii or the verdict; not used to rescue the core hypothesis; disagreements are reported as findings |
| Verdict | Binding map in "Run conditions of G": **GO**: G-i passes **and G-ii is evaluated and passes**; **PARTIAL**: G-i passes and G-ii fails, is `not evaluable under the pre-registered criterion` or is `not run` (A ≠ GO); **NO-GO**: G-i fails (whatever G-ii is); **INCONCLUSIVE**: G-i is `not evaluable` and no mandatory G-i check failed (whatever G-ii is; never PARTIAL, never GO); **not run**: G0 aborts (S11), or S2 / S3 stops G. A `not evaluable`, `not run` or INCONCLUSIVE component is never counted as a pass, and G is never GO then. With 6 model instances G can veto or support; it cannot establish anything on its own, and the report says so. The G verdict row in `REPORT.md` and in the FINAL verdict table carries the binding interpretation sentence above and states the G-i and G-ii status words |

**G-ii under the reduction ladder (binding; no threshold changes).** The criterion stays "rank agreement in ≥ 5 of 6 model
instances and median zero-parameter relative error ≤ 0.5". It is not rescaled (not 4/4, 2/2 or 1/1, not a proportion), no
subset of instances is chosen after seeing results, and the reduction order is the pre-registered one. A model instance is
**completed** iff its fine-tune finished (ID accuracy recorded, gate as in the recipe) and the features of its fit-eligible, held-out
and OOD patches were extracted; an instance lost for any reason (the ladder, S6 / cap, a failed or unfinished run, a failed gate,
missing features) is not completed. An S5 stop (Δ_G(0) CI of `mahalanobis_l2` or `knn_mean_cosine` excludes 0) in a completed
instance is an evaluated failure of the G-ii sanity condition (G-ii `fail`, provided at least 5 instances are completed).

| Completed G2 instances | G-ii | G verdict if G-i passes | if G-i fails | if G-i `not evaluable` |
|---|---|---|---|---|
| 6 | evaluated verbatim: `pass` or `fail` | GO if G-ii passes; PARTIAL if it fails | NO-GO | INCONCLUSIVE |
| 5 (one instance lost for a reason other than the ladder) | evaluated: sanity for every completed instance; rank agreement needs all 5 completed instances to agree (the lost one counts as not agreeing); median over the cells of the 5 | GO if G-ii passes; PARTIAL if it fails | NO-GO | INCONCLUSIVE |
| fewer than 5 (ladder: 4, 2 or 1; or losses) | `not evaluable under the pre-registered criterion` (no pass / fail flag; `pass = na`) | PARTIAL; G-ii never claimed as passed | NO-GO | INCONCLUSIVE |
| G2 not run (A ≠ GO) | `not run` | PARTIAL | NO-GO | INCONCLUSIVE |

The measurements that were made are reported as measured, with the instance count and the ladder step, but no G-ii pass / fail is
computed from them when G-ii is `not evaluable`. The ladder itself always leaves fewer than 5 instances; an evaluated G-ii needs
the full 6 (or, after a loss that is not a ladder step, at least 5) completed instances within the cap.

**G-ii cell set (binding; same trace as the G-i cell set).** P_Δ in G2 is the item-A1 primary additive-map prediction from ρ̂_w and
s_logo, i.e. Mahalanobis only; items A and B register no predictor for kNN (see the trace in "G-i scope and cell set"). Rank
agreement and the median zero-parameter relative error are therefore computed on the **`mahalanobis_l2` scorer-channel cells only**:
for every completed model instance and each k ∈ {1, 2, 4} (scorer dose ω_s ↔ k, ω_b = 0), the measured primary scorer-channel
Δ_G(ω_b = 0, ω_s) − Δ_G(0, 0) against its predicted counterpart; relative error |P_Δ/Δ_meas − 1| exactly as in G-i. kNN scorer-channel
cells are run and reported but excluded from rank agreement and the median (listed, not replaced). **No floor is applied** (no
Δ ≥ 0.02 filter is added here); if a measured scorer-channel Δ is exactly 0 the relative error is undefined and G-ii is `not
evaluable` (no cell is dropped).

**Permitted post-hoc:** the backbone-channel size relative to the scorer channel; ρ̂_w of the fine-tuned
features vs the frozen ones for the same backbone. Seeds 701; GPU-h per run recorded.

Outputs `results/t1/G/`: `manifest_real.csv`, `slide_design.csv`, `MANIFEST_OK` or `MANIFEST_MISMATCH.md`,
`g1_frozen.csv`, `g2_runs.csv` (ID accuracy, GPU-h, checkpoint hash),
`g2_factorial.csv`, `g_predictions.csv` (columns `backbone, scorer, head, k, delta_G_meas, P_delta_zero_param, e_rel, in_gi_median`; P_Δ for `mahalanobis_l2` only, `NA` for kNN and ViM-with-head; `head` is `NA` for `mahalanobis_l2`; `in_gi_median = true` marks the G-i cell set; `e_rel` = |P_Δ/Δ_G − 1| is written only for cells with `in_gi_median = true` and is `NA` otherwise; the median and its status `pass` / `fail` / `not evaluable` are written to `tests.csv`, with `pass = na` when the cell set is empty or incomplete), `g_sensitivity_slide_equal.csv` (report-only), `g_dose_units.csv`,
`tests.csv`, `dose.png` (slide-dose and patch-dose panels), `DEVIATIONS.md`, `REPORT.md` (must contain the binding
interpretation sentence of G-ii, the G-i status word (`pass` / `fail` / `not evaluable` / `not run`) with the outcomes of c1–c3, the G-ii status word (`pass` / `fail` / `not evaluable under the pre-registered
criterion` / `not run`) with the number of instances completed, the G verdict word (GO / PARTIAL / NO-GO / INCONCLUSIVE / `not run`; INCONCLUSIVE is never a pass), the list of cells that entered the G-i median relative
error with each cell's e_rel = |P_Δ/Δ_G − 1| and the median (or "empty") and the statement that kNN and ViM-with-head cells are excluded from it, the primary estimand statement, the two report-only blocks, and the binding
"Scope of the Δ_G confidence intervals" paragraph next to every CI of Δ_G / Δ_G^sw: the bootstrap resamples OOD
patches (and eval slides) as in the R3 helper; the 85,054 OOD patches come from 9 patients and patch-level resampling
does not model within-patient dependence; the CI must **not** be read as uncertainty about generalising to new OOD
patients or hospitals; no claim stronger than the design allows; patient-level uncertainty is a separate inference
design not added here).

---

## H. Negative controls, placebos, estimator coverage, falsification table (stage 3 cheap; stage 4)

**Compute:** CPU + light GPU. Estimate 3–6 GPU-h, cap 10.

### H1. Null controls (expected Δ = 0) (named H-N1 … H-N3 so that they are not confused with E's methods N1–N3; wording only)
- **H-N1 group-free Gaussian twin:** for every A1 cell × fold, simulate Gaussian data with the **real LW
  covariance** of the fit set, the **real group-size vector**, the real N, zero group effect (ρ = 0), a real
  OOD set standardised to the fit moments; run the full A1 pipeline (ρ̂_w, s_logo, ΔQ_pred, Δ_meas). Expected:
  Δ_meas ≈ 0, ρ̂_w ≈ 0 and ΔQ_pred ≈ 0. This gives the **null distribution of the estimators** (their
  noise floor) — the A1 report quotes it.
- **H-N2 one image per group:** every dataset in the registry with ≥ 500 groups (expected: some medbench
  sets): keep one image per group in fit and eval; a random split is then a group split. Δ_meas (random vs
  group split) expected ≈ 0 within CI. This is a pipeline check, not a test of the formula (ρ̂ is
  not estimable at n = 1).
- **H-N3 identical-split control:** compare a fit/eval split with itself (ω = 0 vs ω = 0 resampled): Δ ≈ 0.
- S9 applies to H-N1–H-N3 (H's null controls): |Δ| > 0.02 with CI excluding 0 in more than 5 % of cells → stop H and report.

### H2. Placebo descriptors for item A
**Requires `results/t1/I/MANIFEST_OK` = true** (the Camelyon permutation is within hospitals {0, 3, 4}). If it is false,
H2 is `not run: manifest mismatch (S11)`: no other grouping, no pooled permutation and no substitute structure is used. H2 is not
split by dataset: when it is not run, the **medbench placebo is `not run` as well** (no dataset-wise substitute). H2 also **requires
the A1 outputs**: if A1 did not run (S4, or A `not run`), H2 is `not run: A1 not available`. In both cases H-ii is `not evaluable`
(rule 18), D4's permuted-label check is `not run`, and nothing is substituted.
Recompute the A1 predictor with **permuted fit-group labels** (within hospital — Camelyon hospitals {0, 3, 4},
`default_rng(801)`, 20 permutations): ρ̂_w and ΔQ_pred should collapse toward the H-N1 null. Report LOBO R²_oos and D_b of the
placebo predictor vs the real one. Expected: placebo skill ≤ B0 and real − placebo has lower 95 % CI bound
> 0. Also permute the pairing of predictions and measurements across backbones (Spearman null) — that permutation is item A's own test (`default_rng(102)`), not part of H2, and is unaffected when H2 is not run.

### H3. Coverage of the estimators' jackknife CIs
Toy simulation with real-matched designs (G, n_g, d from the registry: Camelyon-like G = 15 with n_g from the
real slides, d ∈ {768, 1024, 2560}; every medbench design), ρ ∈ {0.02, 0.05, 0.1, 0.2}, 1,000 datasets per
cell, group-weighted delete-a-group jackknife (Busing, as in crossfit-ood) for ρ̂_w and ΔQ_pred; truth = the
simulated ρ and ΔQ_true (50,000 fresh draws per side for the realised fit set). Rule fixed in R3 item 2:
**coverage ≥ 0.93** for nominal 0.95 in every design cell; cells below 0.93 are listed and the CI is
**not** used in the report (point estimates only); no method change.

### H4. Falsification table
`results/t1/H/falsification_table.csv`, one row per claim (F-01 … F-21), filled mechanically from the items' `tests.csv`.
**Fill-in rules:** every claim row gets exactly one status from the stated set; the number of claims and the status
counts (supported-in-tested-region / not-supported / not-evaluable / not-run) are computed over the claim rows only and
sum to the number of claim rows. **Report-only items (the block after the table) are not claims: they are never rows of
`falsification_table.csv`, have no status, and are excluded from the claim total and from every status count.** A claim whose deciding sub-criterion is `not evaluable`
(for example S11 cases, or G-ii after the reduction ladder) gets status `not-evaluable`; one whose items were not run gets
`not-run`; neither is ever `supported-in-tested-region`, and a status other than `supported-in-tested-region` is never
counted as support (an item verdict INCONCLUSIVE is never support either).


| id | claim | item | threshold (pre-registered) | observed | status {supported-in-tested-region / not-supported / not-evaluable / not-run} |
|---|---|---|---|---|---|
| F-01 | K2 reproduces in the toy | A0a | within 3 SE; theory fn 1 % | | |
| F-02 | Track A Δ reproduced | A0b | 1e-6 | | |
| F-03 | P beats ICC-only across backbones | A-i | LB(mean D_b) > 0 vs B1 and B1w | | |
| F-04 | P beats d/N, ρ̂², (bound if discriminable) | A-ii | LB > 0 | | |
| F-05 | P predicts dependence on G, n, N, λ | A-iii | LB > 0 vs B6-scaling and B5 (A-iii `not evaluable` under S11 → `not-evaluable`; never `not-supported`, never supported) | | |
| F-06 | ΔQ accurate | A-iv | median rel. error ≤ 0.35; R²_oos > 0 | | |
| F-07 | kNN leak collapses on ρ√PR | B-i | R²_oos > 0; LB > 0 vs ICC | | |
| F-08 | matched toy simulator predicts kNN | B-ii | LB > 0 vs ICC | | |
| F-09 | closed form valid in R₀ | C-i | median ≤ 0.10, p90 ≤ 0.25 | | |
| F-10 | cap holds | C-ii | no S8 | | |
| F-11 | kNN collapse in the toy | C-iii | score ≥ 0.90 | | |
| F-12 | T separates overlap | D-i/D-iii | sens, spec ≥ 0.90; AUC ≥ 0.90 on ≥ 11/13 | | |
| F-13 | no regime with material leak and blind T | D-ii | none found | | |
| F-14 | group-level CS valid for θ_group (regimes (a) and (b), E1 and E4) | E-i | ≤ α + 3 SE everywhere | | |
| F-15 | label-free recovery above threshold | E-iii | no failure at ρ_f√(d/2) ≥ 3 | | |
| F-16 | first-order sign law on real cells | F-ii | sens ≥ .80 (LB > .65), spec ≥ .90 | | |
| F-17 | Edgeworth sign on balanced shifts | F-iii | acc ≥ .70 (LB > .5), beats skew baseline | | |
| F-18 | blind-but-detectable on real features | F-iv | ≥ 80 % of backbones | | |
| F-19 | scorer-channel prediction under fine-tuning (primary patch-level Δ_G; predictive accuracy under joint change of fit set and eval set across k ∈ {1, 2, 4}, not a pure fit-effect measurement) | G | rank ≥ 5/6; rel. error ≤ 0.5 (threshold unchanged). **Status is set by G-ii alone:** `pass` → `supported-in-tested-region`; `fail` → `not-supported`; `not evaluable` (fewer than 5 completed instances) → `not-evaluable`; `not run` (A ≠ GO, G0 abort S11, S2 / S3) → `not-run`. The G-i status and the G verdict (PARTIAL / NO-GO / INCONCLUSIVE) do not change it, and a G verdict INCONCLUSIVE is never shown as support | | |
| F-20 | null controls and placebo | H1/H2 | S9 not triggered; real − placebo LB > 0 (S9 triggered → `not-supported`; otherwise H2 `not run` for any reason (S11, or A1 not available) → `not-evaluable`; H `not run` (S2 / S3) → `not-run`; H verdict INCONCLUSIVE → `not-evaluable`; never supported) | | |
| F-21 | estimator CIs cover | H3 | ≥ 0.93 in every cell | | |

**Report-only items (not claims).** Listed here, not in the table above; each appears in the `REPORT.md` of its item
and in the FINAL report; not counted in the number of claims, in any status count or in any total; they are not H4 statuses and
cannot change any claim row or any verdict. F-19r has no status and no threshold. F-02c, F-16i and F-19i (v2.4) only make visible
in the consolidation three sub-criteria that have no claim row above (A0c, F-i, G-i); each shows the item's own pre-registered
threshold and the item's own result word (`pass` / `fail` / `not evaluable` / `not run`) copied from that item's `tests.csv`; they are
not new claims and add no threshold.

| id | item | content | threshold | observed |
|---|---|---|---|---|
| F-02c | A0c | unit conversion of the scorer precision | 1e-9 relative (as A0c) | item's result word, no H4 status |
| F-16i | F-i | F1 reproduction of the Gaussian reference | within tolerance (as F-i) | item's result word, no H4 status |
| F-19i | G-i | frozen-feature checks c1–c3 (Δ_G(0) CI in 10 combinations; median e_rel over the G-i cell set, or "empty") | as G-i | item's result word (`pass` / `fail` / `not evaluable` / `not run`), no H4 status |
| F-19r | G | slide-equal-weight Δ_G^sw vs patch-level Δ_G (sign flags / > 50 % magnitude disagreements) and slide-dose vs patch-dose conclusions | none (never changes F-19) | reported, no status |

### Pre-registered decision

| Sub-criterion | Pass when |
|---|---|
| H-i | S9 not triggered in H-N1–H-N3 |
| H-ii | placebo skill ≤ B0 and real − placebo LB > 0 (needs the A1 outputs). If H2 is `not run` for any reason — `MANIFEST_OK` false (`not run: manifest mismatch (S11)`), or A1 not available (S4 / A `not run`: `not run: A1 not available`) — H-ii is `not evaluable` (counts as not passed, never a pass) |
| H-iii | H3 coverage ≥ 0.93 in every design cell |
| Verdict | **GO** all three evaluated and passed; **NO-GO** H-i fails (an evaluated failure) → every A, B, G verdict is marked "pipeline-suspect" in the consolidation; **PARTIAL** H-i passes and at least one of H-ii, H-iii fails or is `not evaluable` (the affected quantity is reported without CIs / without a skill claim); **INCONCLUSIVE** H-i is `not evaluable` (rule 18); **not run** when S2 or S3 stops H. **If `MANIFEST_OK` is false (S11): H2 is `not run`, H-ii is `not evaluable`, H cannot be GO; if H-i and H-iii are evaluable and pass, H = PARTIAL with H-ii stated `not evaluable`** |

Seeds 801. Outputs `results/t1/H/`: `h1_nulls.csv`, `h2_placebo.csv`, `h3_coverage.csv`,
`falsification_table.csv`, `tests.csv`, `DEVIATIONS.md`, `REPORT.md`.

---

## FINAL. Consolidation (CPU, stage 5)

Write `results/t1/REPORT.md` containing, in this order and nothing else but what the rules require:

1. Header: commit, precommit hash, bundle hashes, environment, dates (Asia/Saigon time).
2. **Verdict table:** one row per item: verdict (GO / PARTIAL / NO-GO / INCONCLUSIVE / not run; items 0 and I: PASS / FAIL; INCONCLUSIVE and `not run` are never shown or counted as a pass or as support), the
   pre-registered sub-criteria with their **computed** values and pass flags, estimated vs actual GPU-h,
   number of deviations (link to each `DEVIATIONS.md`), number of tests run (rows in `tests.csv`) and how
   many are post-hoc. **The G row must (a) state that Δ_G is the patch-level pooled AUROC difference, (b) carry
   the binding sentence of Item G verbatim: "Across doses k ∈ {1, 2, 4} both the fit set and the seen-eval slide set
   change (seen = E₁; E₁+E₂; E₁…E₄; unseen = U, fixed); only k = 0 vs k = 2 shares the same eval slides (E₁+E₂),
   differing in fit membership; therefore G-ii is a test of predictive accuracy under joint change of fit set and
   evaluation set, not a pure measurement of the fit effect", (c) list the report-only slide-equal-weight
   disagreement counts and the slide-dose vs patch-dose findings beside, not inside, the verdict, (d) list the G-i cells
   that entered the median e_rel = |P_Δ/Δ_G − 1| of P_Δ (`mahalanobis_l2` only; kNN and ViM-with-head stated as excluded from
   it, ViM-with-head not part of G-i), with the median value or "empty", (e) state the G-i status word (`pass` / `fail` / `not
   evaluable` / `not run`) and the G-ii status word (`pass` / `fail` / `not evaluable under the pre-registered criterion` /
   `not run`) with the number of model instances completed, and (f) carry the G verdict word (GO / PARTIAL / NO-GO / INCONCLUSIVE /
   `not run`) and this CI-scope statement: "The Δ_G confidence intervals resample eval slides and OOD patches only (85,054 OOD
   patches from 9 patients; within-patient dependence is not modelled); they are not uncertainty about new OOD patients or
   hospitals." A `not evaluable`, `not run` or INCONCLUSIVE component is never shown or counted as a pass: G is PARTIAL only if G-i
   passes and G-ii is not a pass, NO-GO if G-i fails, INCONCLUSIVE if G-i is `not evaluable`, and never GO unless G-ii was
   evaluated and passed.** In every row, a sub-criterion that is `not
   evaluable` or `not run` is shown under that word and never with a pass flag (the H row states H2 `not run` and H-ii `not
   evaluable` when `MANIFEST_OK` is false).
3. **Which idea survives (rules fixed now):**
   - *Idea 2, real-feature theory + protocol:* A = GO ∧ C = GO ∧ H ∈ {GO, PARTIAL} ∧ H-iii pass ∧ D ∈ {GO, PARTIAL}.
   - *Idea 2, toy theory + diagnostic protocol note (no real-feature mechanism claim):* C = GO and A ∈
     {PARTIAL, NO-GO}.
   - *Idea 2 dropped:* C ∈ {PARTIAL, NO-GO}.
   - *Idea 1 stand-alone label-free certificate:* E = GO; *design-effect corollary only:* E = PARTIAL; *dropped:*
     E = NO-GO. State whether it folds into Idea 2 (if Idea 2 survives) or has no home.
   - *Idea 3 stand-alone / lemmas / dropped:* as the F verdict.
   - *Not determined:* any case not matched by the lines above because a listed item verdict is INCONCLUSIVE or `not run` (for
     example C INCONCLUSIVE; A INCONCLUSIVE (including A-iii `not evaluable` under S11 without an evaluated A-i and A-iv failure) or `not run` with C = GO; E or F INCONCLUSIVE): the idea is stated as **not
     determined** — it neither survives nor is dropped. INCONCLUSIVE and `not run` never satisfy a survival condition. The verdict
     of G is not an input of these rules.
   - If more than one survives, give the order of evidence strength by the number of pre-registered
     sub-criteria passed on **real** features, not by the toy.
   - An A-iii that is `not evaluable` (S11) is never cited as evidence against the hypothesis and never counted as support; it contributes to
     A = NO-GO only through the A verdict table's S11 clause (A-i and A-iv both failed as evaluated).
   - **No secondary test may be used to rescue the core hypothesis after A is NO-GO.** If A = NO-GO, no result
     of B, C5, D, F, G, H, no post-hoc analysis and no re-weighting of sub-criteria may be cited as support
     for the closed form (K2) on real features; the only permitted consequence is the "toy theory + diagnostic
     protocol note" line above (which makes no real-feature mechanism claim), and G2 stays `not run`.
4. The falsification table (H4), completed (claims F-01 … F-21 only), followed by the report-only block of H4 (F-02c, F-16i, F-19i, F-19r), which is shown but not counted in claim totals or status counts.
5. **Not claimed:** a bullet list of every statement in this order that was not tested (e.g. non-Gaussian
   proofs, the cap's proof, regimes outside R₀, non-Camelyon datasets with n = 4), the fit effect alone in G (it would need a fixed evaluation set across doses;
   not designed here; a possible follow-up order), and every
   `not run` / `not evaluable` cell.
6. Post-hoc list (every `_posthoc` file and what it showed, labelled).
7. Reproduction block: the exact commands, SLURM job ids and seeds used per item.

No sentence of the consolidation may use "proved", "confirmed", "explains" or "mechanism established".
Also write `results/t1/README.md` (inventory of files, actual vs estimated GPU-h, final deviations,
branch and commit) and push **only** branch `t1`.

---

## Run checklist (the agent follows this; no waiting for approval between stages)

1. Item 0 → commit precommit → item I (including the Camelyon manifest check; `MANIFEST_OK`).
2. A0 → if pass: launch A1/A2/A3 and, in parallel, C0 → C1–C4, B, D1–D2 (≤ 2 GPUs unless idle).
3. Write `results/t1/A/REPORT.md` the moment A finishes; apply the early-stop table (if A-iii is `not evaluable` under S11, the A verdict comes from the S11 clause of the A verdict table; A-iii is never counted as not passed toward NO-GO).
4. Stage 3: C5, D3–D5, E, F1–F3, H1–H3 (H2 only if `MANIFEST_OK` is true, S11).
5. Stage 4: F4; G0 (manifest verification; abort per S11) → G1; if A = GO then G2; H4.
6. FINAL. Then print the one-line verdicts for A–H and the surviving-idea line, nothing else.
