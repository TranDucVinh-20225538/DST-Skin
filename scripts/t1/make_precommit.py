"""Write results/t1/PRECOMMIT_T1.json and decisions/precommit_t1_<date>.md (WORK ORDER T1 v2.5, rule 1 and rule 16).

Reads only the order, the bundle and the zip (hashes); no data, no computation.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.expanduser("~/DST-Skin")
ORDER = os.path.expanduser("~/1110/WORK_ORDER_T1_v2_5.md")
CHANGELOG = os.path.expanduser("~/1110/CHANGELOG_v2_5.md")
BUNDLE = os.path.expanduser("~/handoff/t1/t1_theory_bundle.tgz")
BUNDLE_DIR = os.path.expanduser("~/handoff/t1/bundle")
ZIP = os.path.expanduser("~/handoff/crossfit-ood_v3.zip")
HASHES_TXT = os.path.expanduser("~/1110/PRECOMMIT_T1_hashes.txt")

EXPECTED = {
    "order": "4dec41ec184ce07c811c9ff9c6434bbcd382228b49a192bde7bb00c5cd5c3096",
    "bundle": "035454a4c13a16e17179aa9a656e89252b2f2de866f5a97a7ab0e2425b5d1f06",
    "zip": "c925933300e29dc76ff599cfc16696901aaa483eabc20b8fce3ad14ae2f9dce1",
}
MASTER_SEED = 20261010
ITEM_CODE = {"0": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def work_list(item, j, n_lists, ids):
    sorted_ids = sorted(ids)
    child = np.random.SeedSequence([MASTER_SEED, ITEM_CODE[item], 16, 0]).spawn(n_lists)[j]
    perm = np.random.default_rng(child).permutation(len(sorted_ids))
    joined = "\n".join(sorted_ids)
    return {
        "item": item,
        "list_index_j": j,
        "n_lists": n_lists,
        "seed": f"SeedSequence([{MASTER_SEED}, {ITEM_CODE[item]}, 16, 0]).spawn({n_lists})[{j}]",
        "n_units": len(sorted_ids),
        "sha256_sorted_ids_newline_joined": hashlib.sha256(joined.encode()).hexdigest(),
        "sorted_ids": sorted_ids,
        "execution_order": [sorted_ids[p] for p in perm],
    }


def main():
    hashes = {"order": sha256_file(ORDER), "bundle": sha256_file(BUNDLE), "zip": sha256_file(ZIP)}
    for k, v in hashes.items():
        if v != EXPECTED[k]:
            sys.exit(f"S1: sha256 mismatch for {k}: {v} != {EXPECTED[k]}")
    bundle_files = {}
    for line in open(HASHES_TXT):
        parts = line.strip().split("  ")
        if len(parts) == 2 and parts[1].startswith("t1_bundle/"):
            got = sha256_file(os.path.join(BUNDLE_DIR, parts[1]))
            if got != parts[0]:
                sys.exit(f"S1: bundle member {parts[1]} sha256 {got} != {parts[0]}")
            bundle_files[parts[1]] = got

    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).date().isoformat()

    g1_ids = [f"g1|{fm}|{head}|k{k}"
              for fm in ["uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5"]
              for head in ["linear_probe", "adapter"] for k in [0, 1, 2, 4]]
    g2_ids = [f"g2|{bb}|k{k}|s{s}" for bb in ["resnet50", "convnext_tiny"]
              for k in [0, 1, 2, 4] for s in [42, 43, 44]]
    work_lists = [work_list("G", 0, 2, g1_ids), work_list("G", 1, 2, g2_ids)]
    deferred = {
        "0": "item 0 draws no random numbers and has no capped work list",
        "I": "item I draws no random numbers (inventory)",
        "A": "units are registry cells (dataset x backbone x seed x fold, A2 designs); enumerated from results/t1/I/registry.csv in a precommit addendum before A1 starts",
        "B": "units are registry cells; enumerated in a precommit addendum before B starts",
        "C": "C1/C5 configs are drawn from the C1 design (seed 301); the config list is generated and recorded in a precommit addendum before C launches (after item 0 BUDGET.md)",
        "D": "D1/D2 configs drawn with seed 401 and D3 registry cells; recorded in a precommit addendum before D starts",
        "E": "list j=0 (regime a) and j=1 (regime b) include E3/E4 registry-dependent units; recorded in a precommit addendum before E starts",
        "F": "units are registry / paper-3 cells; recorded in a precommit addendum before F starts",
        "H": "units are registry cells and H3 designs from the registry; recorded in a precommit addendum before H starts",
    }

    pre = {
        "title": "PRECOMMIT T1 (WORK ORDER T1 v2.5) - validate or kill the theory program",
        "date_asia_saigon": date,
        "status": "Pre-registration (rule 1). Committed and pushed on branch t1 before any computation other than item 0 and the inventory. Nothing has been run.",
        "binding_text": {
            "order_file": "WORK_ORDER_T1_v2_5.md",
            "order_sha256": hashes["order"],
            "copy_in_repo": f"decisions/precommit_t1_{date}.md (verbatim order text appended)",
            "rule": "Every criterion, threshold, seed, grid, stop rule and verdict rule is the text of WORK_ORDER_T1_v2_5.md (sha256 above). The structured fields below summarise it; where they differ from the order, the order wins.",
            "changelog_file": "CHANGELOG_v2_5.md",
            "changelog_sha256": sha256_file(CHANGELOG),
        },
        "inputs": {
            "crossfit_ood_v3_zip": {"path": ZIP, "sha256": hashes["zip"], "expected": EXPECTED["zip"]},
            "t1_theory_bundle_tgz": {"path": BUNDLE, "sha256": hashes["bundle"], "expected": EXPECTED["bundle"],
                                      "members_sha256": bundle_files,
                                      "note": "reference toy outputs for A0/C0/F1 reproduction only; never quoted as measurements"},
        },
        "repo": {"repo": "DST-Skin", "branch": "t1", "branched_from": "rigor-pack",
                 "base_commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
                 "push_policy": "push only branch t1"},
        "seeds": {
            "master_seed": MASTER_SEED,
            "random_draws": "numpy.random.SeedSequence([20261010, item_code, stable_hash(config), replicate])",
            "item_code": ITEM_CODE,
            "fixed_resampling_default_rng": {"A_bootstrap": 101, "A_permutation": 102, "B_bootstrap": 201, "B_permutation": 202,
                                              "C": 301, "D": 401, "E": 501, "F": 601, "G": 701, "H": 801},
            "bootstrap_B": {"cross_backbone": 10000, "auroc_level": 2000, "auroc_level_rng": "default_rng(2)"},
            "work_list_ordering": "child_j = SeedSequence([20261010, item_code, 16, 0]).spawn(n_lists)[j]; perm = default_rng(child_j).permutation(len(list)) over IDs sorted by Python string order",
            "G_slide_assignment": "default_rng(SeedSequence([20261010, 7, 0]))",
            "G_heldout_draw": "SeedSequence([20261010, 7, 1, stable_hash(slide)])",
            "A_subsampling": "SeedSequence([20261010, 1, stable_hash(config), rep])",
            "B_MTS": "SeedSequence([20261010, 2, stable_hash(config), rep])",
            "F4_patch_sample": "SeedSequence([20261010, 6, stable_hash(config), rep])",
            "G1_adapter_seed": 42,
            "G2_seeds": [42, 43, 44],
            "stable_hash_definition": "stable_hash(x) = int.from_bytes(sha256(s.encode('utf-8')).digest()[:4], 'big') where s = x if x is a str else json.dumps(x, sort_keys=True, separators=(',', ':')). DEFAULT fixed at precommit: the order uses stable_hash without defining it.",
        },
        "standing_rules": {
            "float64": "all covariance, precision, LOGO, ICC and statistics; torch tf32 off",
            "gpu_knn": "R3 GPU kNN only after equality check vs float64 CPU on 2 cells (k=1, k=50) in item 0",
            "chance_floors": {"AUROC": 0.5, "Delta": 0, "Spearman": "0 by pairing permutation", "R2": "R2_oos vs constant from training backbones",
                              "sign_accuracy": "majority-class fraction", "ARI": 0, "coverage": "nominal", "inversion": "never predict inversion"},
            "holm": "within each item's family",
            "scorers": {"mahalanobis_l2": "Ledoit-Wolf on L2-normalised features", "knn_mean_cosine": "k = 50",
                        "vim": "residual-only unless the cell has a head file"},
            "caps_gpu_h": {"0": 1, "I": 0, "A": 12, "B": 18, "C": 160, "D": 24, "E": 6, "F": 20, "G": 80, "H": 10, "campaign": 330, "max_overrun": 0.20},
            "planning_estimates_gpu_h": {"0": 0.5, "I": 0, "A": "4-8", "B": "6-12", "C": "80-130", "D": "8-15", "E": "1-3", "F": "6-12", "G": "45-70", "H": "3-6"},
            "verdict_rules_not_evaluable": "rule 18 (a)-(f) of the order",
        },
        "stops": {
            "S1": "bundle file missing or wrong sha256 -> campaign stops",
            "S2": "crossfit-ood v3 tests not 44/44 or A0b Track A Delta not reproduced to 1e-6 -> A, B, D3-D5, E3-E4, F2-F4, G, H stop",
            "S3": "A0a or A0c fails -> everything except E and F1 stops",
            "S4": "< 8 Camelyon backbones with cached features and R3 item-1 Delta for mahalanobis_l2 -> A INCONCLUSIVE",
            "S5": "sanity Delta(0) CI excludes 0 -> stop that model x scorer (consequences per order)",
            "S6": "item cap exceeded by 20 % or campaign > 330 GPU-h",
            "S7": "any write under results/r3/, outputs/, or overwrite of a non-T1 file",
            "S8": "C4: non-vacuous config with 99.9 % upper bound of E[Delta] > cap K3",
            "S9": "H1: null control |Delta| > 0.02 with CI excluding 0 in > 5 % of cells",
            "S10": "decision criterion changed after output exists",
            "S11": "Camelyon manifest mismatch (30 train slides != 10 per hospital over {0,3,4}, hospital 1/2 in any fit set, or G design not reproducible)",
        },
        "camelyon_manifest_expected": {
            "train_hospitals": [0, 3, 4], "slides_per_hospital": 10, "train_slides": 30,
            "train_patches": 302436, "id_val_patches": 33560, "ood_hospital": 2, "ood_patches": 85054, "ood_patients": 9,
            "hospital_1": "unused", "paper_2fold_slides_per_hospital_per_fold": 5,
            "patch_total_differences": "WARN, not a stop",
        },
        "items": {
            "0": {"pytest_expected": "44/44 (else S2)", "gpu_knn_check": "2 cells, k = 1 and k = 50, record max |dDelta|",
                  "verdict": "PASS / FAIL"},
            "I": {"MANIFEST_OK": "true iff 30 train slides exactly {0,3,4} x 10; OOD all hospital 2 and disjoint from train; every Camelyon cache slide set == metadata; every fold of every Camelyon cell 5 slides per hospital",
                  "verdict": "PASS if S4 not triggered and MANIFEST_OK true; else FAIL (S4) / FAIL (S11)"},
            "A": {
                "A0a": {"cells": [{"d": 128, "G": 40, "n": 10, "rho": 0.6, "lam": 0, "sim": -170.27, "se": 0.66, "theory": -168.93},
                                  {"d": 128, "G": 100, "n": 5, "rho": 0.3, "lam": 0.1, "sim": -12.73, "se": 0.23, "theory": -12.99},
                                  {"d": 32, "G": 40, "n": 10, "rho": 0.6, "lam": 0, "sim": -10.89, "se": 0.20, "theory": -10.43}],
                        "replicates": 24, "pass": "new sim within 3 combined SE; theory fn within 1 % relative; else S3"},
                "A0b": {"cells": ["resnet50 s42", "uni", "dinov2_vitb14"], "scorer": "mahalanobis_l2", "tolerance": 1e-6, "fail": "S2"},
                "A0c": {"tolerance_relative": 1e-9, "fail": "S3"},
                "A1": {"baselines": ["B0 constant", "B1 rho_raw", "B1w rho_w", "B2 G", "B3 d/N", "B4 rho_w^2", "B5 rho_w*s_logo"],
                       "calibration": "LOBO y = a + b z", "primary": "P_dQ, P_Delta additive map", "LOGO_folds": "G refits, K = 10 group folds if G > 40",
                       "permutation": "pairing permutation default_rng(102), 10,000"},
                "A2": {"G_prime": [3, 6, 9, 12, 15], "n_prime": [10, 30, 100, 300, 1000, "all"], "repeats": 10,
                       "stratification": "G'/3 slides per hospital {0,3,4}", "baseline_B6": "OLS on (rho_w, log G', log n', d/N')",
                       "sanity_S5": "(G'=15, n'=all) reproduces A1 fold Delta to 1e-6", "requires": "MANIFEST_OK"},
                "criteria": {"A-i": "LB95 mean D_b > 0 vs each of B1 and B1w (Camelyon LOBO n = 13)",
                             "A-ii": "same vs B3, B4, and B5 only if r_sat < 0.9 in >= 20 % of cells",
                             "A-iii": "LB95 mean D_b > 0 vs B6-scaling and B5 pooled over designs, and mean within-backbone Spearman > 0 with LB > 0; MANIFEST_OK false -> not evaluable",
                             "A-iv": "median |dQ_pred/dQ_meas - 1| <= 0.35 and pooled R2_oos of LOBO-calibrated dQ_pred vs B0 > 0"},
                "bootstrap": "percentile cluster bootstrap over backbones, B = 10,000, default_rng(101)",
                "verdict": "order section A verdict table incl. v2.5 S11 clause",
            },
            "B": {"k": [1, 5, 10, 20, 50, 100, 200], "primary_k": 50, "MTS_replicates": 16, "MTS_d_cap": 1024,
                  "TwoNN_subsample": 20000,
                  "criteria": {"B-i": "k = 50 Camelyon LOBO: R2_oos of isotonic map from x1 > 0 and LB95 paired LOBO error difference vs B1 > 0",
                               "B-ii": "MTS pooled MAE below B1 and LB95 mean D_b > 0"},
                  "verdict": "GO both; PARTIAL exactly one; NO-GO neither"},
            "C": {"replicates": {"min": 8, "max": 64, "se_dQ_rel": 0.02, "se_Delta": 0.004}, "m_per_side": 4000,
                  "omega": [0, 0.25, 0.5, 1], "K1_tolerance": 1e-12,
                  "OOD": {"O1_c": [1, 2, 4], "O2_c_prime": [0.5, 1, 2], "O3": "null"}, "lambda": [0, 0.01, 0.1, 1],
                  "knn_k": [1, 5, 20],
                  "C0_replicates_min": 64,
                  "C1": {"n_configs": 6000, "seed": 301, "d": [16, 32, 64, 128, 256, 512, 1024, 2048],
                         "rho": [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7], "G": [6, 12, 24, 48, 100, 200, 400],
                         "n": [5, 10, 25, 50, 100, 500], "N_max": 200000,
                         "TRAIN_d": [16, 32, 128, 512], "TEST_interp_d": [64, 256, 1024], "TEST_extrap": "d = 2048 or G = 6 or rho = 0.7",
                         "correction_family": "log(dQ_sim/dQ_th) = b0 + b1/d + b2/G + b3/N + b4 rho + b5 d/N + b6 lam, ridge, 5-fold CV within TRAIN grouped by d",
                         "R0": "d >= 64, G >= 12, N >= 2d"},
                  "C3": {"knn": [1, 5], "G": [50, 100, 200], "n": [5, 10, 25], "d": [16, 32, 64, 128, 256, 512, 1024],
                         "rho": [0.03, 0.05, 0.1, 0.2, 0.3, 0.5], "interior": "0.02 < Delta < 0.45"},
                  "C4": {"exact_TV": {"d": [1, 2, 3], "rho": [0.3, 0.5, 0.7, 0.9, 0.99], "G": [3, 10, 30, 100, 300], "draws_min": 40, "d3_grid": "256^3"},
                         "scorer_grid": {"d": [2, 4, 8, 16], "replicates_min": 64, "upper_bound": 0.999, "non_vacuous": "cap < 0.45"}},
                  "C5": {"configs_per_variant": 1500, "t_nu": [3, 5, 10, 30], "spikes_r": [1, 5, 20], "spike_kappa": [5, 20, 100],
                         "power_alpha": [0.5, 1, 1.5], "classes_K": [2, 5, 10], "class_sep": [0, 1, 3, 6], "small_G": [6, 12, 24]},
                  "criteria": {"C-i": "uncorrected K2 median |dQ_sim/dQ_th - 1| <= 0.10 and p90 <= 0.25 on TEST-interp and TEST-extrap within R0",
                               "C-ii": "no S8", "C-iii": "collapse score >= 0.90 (reported)",
                               "C-iv": "(b)-proportional whitened ridge >= 80 % within 0.25 (reported)"},
                  "verdict": "GO C-i and C-ii; PARTIAL exactly one; NO-GO neither"},
            "D": {"m_eval_max": 5000, "fit_subsample": 20000,
                  "D1": {"n_configs": 1000, "seed": 401, "d": [32, 128, 512, 1024, 2048], "G": [12, 24, 100, 400], "n": [5, 25, 100],
                         "omega": [0, 0.25, 0.5, 1], "material": "Delta_full >= 0.02", "tau_TRAIN_d": [32, 512, 2048], "TEST_d": [128, 1024]},
                  "D2": {"random_configs": 3000, "cmaes_population": 32, "cmaes_generations": 40, "replicates": 16, "seed": 401},
                  "D3": {"omega": [0, 0.25, 0.5, 0.75, 1], "subsamples": 50, "subsample_size": 5000},
                  "criteria": {"D-i": "sens >= 0.90 and spec >= 0.90 on material TEST configs",
                               "D-ii": "no config with Delta_full >= 0.02 and AUC(T) < 0.8",
                               "D-iii": "AUC(T_seen vs T_unseen) >= 0.90 in >= 11 of 13 Camelyon backbones"}},
            "E": {"unit_test": "Bernoulli(0.3), 10,000 reps, horizon 2,000, ever-miscover <= alpha + 3 SE",
                  "E1": {"replicates": 10000, "alpha": 0.05, "horizon_groups": 2000, "eps": [0.05, 0.10],
                         "regime_a": {"n": [5, 20, 50, 200], "rho": [0, 0.02, 0.05, 0.1, 0.2, 0.4], "outcomes": ["flag theta 0.3", "flag theta 0.05", "continuous Beta"],
                                      "sizes": ["constant", "1+Poisson(n-1)", "lognormal CV=1"]},
                         "regime_b": {"n": [5, 20, 50], "rho": [0.05, 0.2], "mechanisms": ["copula kappa +0.6", "copula kappa -0.6", "size proportional to m_g"],
                                      "cells": 54, "theta_size_MC_draws": 10000000, "theta_size_MC_se_max": 1e-4},
                         "betting_grid": 1001},
                  "E2": {"G": 60, "n": [15, 5, 50], "rho_f": [0.05, 0.1, 0.2, 0.3, 0.6], "d": [4, 16, 64, 256, 1024], "score_icc": 0.2, "replicates": 500},
                  "E3": {"patches_per_fold": 4000},
                  "E4": {"sizes_a": [20, 100], "size_cap": 1000, "flag_quantile": 0.05, "spearman_bootstrap": 2000},
                  "criteria": {"E-i": "G1 ever-miscover of theta_group <= 0.05 + 3 SE in every E1 cell (a) and (b) and every E4 backbone x regime",
                               "E-ii": "N1 ever-miscover > 0.05 + 3 SE in >= 90 % of regime-(a) E1 cells with DEFF >= 2",
                               "E-iii": "no E2 config with rho_f sqrt(d/2) >= 3, G >= 60 and M1 coverage < 0.90; E3 match >= 80 %"}},
            "F": {"F1": {"shifts": {"d8": 240, "d32": 240, "d128": 96}, "tol_sign_acc": 0.02, "tol_balanced_mean": 0.005,
                         "balanced_ref": {"d8": 0.489, "d32": 0.491, "d128": 0.495}},
                  "F3": {"a": [1.2, 1.5, 2.0], "blind": "Maha AUROC in [0.45, 0.55]", "detectable": "contrast AUROC >= 0.90"},
                  "F4": {"backbones": ["resnet50 s42", "uni", "dinov2_vitb14"], "patches": 20000, "reduction": [20000, 10000, 5000],
                         "corruptions": ["gaussian_noise", "gaussian_blur", "jpeg", "brightness"], "severities": 3, "cap_gpu_h": 15},
                  "criteria": {"F-i": "F1 within tolerance", "F-ii": ">= 30 inversion units; sens >= 0.80 (Wilson LB > 0.65), spec >= 0.90",
                               "F-iii": "Edgeworth sign acc >= 0.70 (Wilson LB > 0.5) on >= 30 non-degenerate cells (|AUROC-0.5| >= 0.002) and paired sign test p < 0.05 vs skew baseline",
                               "F-iv": ">= 80 % of backbones with a blind-but-detectable cell"}},
            "G": {"slide_design": {"per_hospital": {"U": 2, "T": 4, "E1": 1, "E2": 1, "E3": 1, "E4": 1},
                                   "algorithm": "for h in (0,3,4): sorted slide IDs as strings, p = rng.permutation; U=p[0:2], T=p[2:6], E_j=p[5+j]"},
                  "heldout_fraction": 0.3, "doses_k": [0, 1, 2, 4], "omega": [0, 0.25, 0.5, 1], "G_fit": [12, 15, 18, 24],
                  "bootstrap": "B = 2,000 over eval slides and OOD patches, default_rng(2)",
                  "G1": {"backbones": ["uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5"],
                         "heads": {"linear_probe": "logistic L2 C=1 standardised", "adapter": "512 ReLU AdamW lr 1e-3 20 epochs seed 42"},
                         "scorers": ["mahalanobis_l2", "knn_mean_cosine", "vim_with_head"]},
                  "G2": {"backbones": ["resnet50", "convnext_tiny"], "seeds": [42, 43, 44], "runs": 24, "budget_gpu_h": 80,
                         "reduction_ladder": ["drop seed 44", "drop seed 43", "drop convnext_tiny"], "recipe": "verbatim isbi_patch2 retrain arm"},
                  "criteria": {"G-0": "G0 passed (no S11)",
                               "G-i": "c1 implementation checks exact; c2 Delta_G(0) CI contains 0 in 10 combinations; c3 median e_rel = |P_Delta/Delta_G - 1| <= 0.5 over mahalanobis_l2 FM x k in {1,2,4} with Delta_G >= 0.02",
                               "G-ii": ">= 5 of 6 instances completed; Delta_G(0) sanity; rank agreement in >= 5 of 6 instances and median zero-parameter rel. error <= 0.5 (mahalanobis_l2 scorer channel)"},
                  "verdict": "order section G verdict map"},
            "H": {"H2_permutations": 20, "H2_rng": "default_rng(801)", "H3": {"rho": [0.02, 0.05, 0.1, 0.2], "d": [768, 1024, 2560], "datasets_per_cell": 1000,
                                                                                    "truth_draws": 50000, "coverage_min": 0.93},
                  "criteria": {"H-i": "S9 not triggered", "H-ii": "placebo skill <= B0 and real - placebo LB > 0 (S11 -> not evaluable)", "H-iii": "coverage >= 0.93 in every design cell"}},
        },
        "work_lists": work_lists,
        "work_lists_deferred": deferred,
        "defaults_fixed_at_precommit": [
            "stable_hash definition (see seeds.stable_hash_definition)",
            "work-list ID formats of G: 'g1|<fm>|<head>|k<k>' (40 units; all scorers of a unit run together) and 'g2|<backbone>|k<k>|s<seed>' (24 fine-tunes)",
            "work lists that depend on the item I registry or on sampled designs are recorded in precommit addenda before their item launches (decision of the user, 2026-10-11)",
        ],
    }
    out_json = os.path.join(REPO, "results/t1/PRECOMMIT_T1.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pre, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, out_json)

    md = os.path.join(REPO, f"decisions/precommit_t1_{date}.md")
    if os.path.exists(md):
        sys.exit(f"refusing to overwrite {md}")
    header = f"""# Precommit T1 ({date}, Asia/Saigon) - WORK ORDER T1 v2.5

Pre-registration under rule 1 of the order. Committed on branch `t1` before any computation other than item 0 and the inventory.
Nothing has been run.

- Order: `WORK_ORDER_T1_v2_5.md`, sha256 `{hashes['order']}` (full text appended below, verbatim; it is the binding text).
- Bundle: `t1_theory_bundle.tgz`, sha256 `{hashes['bundle']}`; all {len(bundle_files)} members verified against `PRECOMMIT_T1_hashes.txt`.
- Input zip: `crossfit-ood_v3.zip`, sha256 `{hashes['zip']}`.
- Structured summary: `results/t1/PRECOMMIT_T1.json` (the order wins where they differ).
- Work lists recorded now (rule 16): G list 0 (G1, {len(g1_ids)} units), G list 1 (G2, {len(g2_ids)} units).
  Lists that depend on the item I registry or on sampled designs are recorded in precommit addenda before their item launches.
- Defaults fixed now (the order does not define them): `stable_hash(x) = int.from_bytes(sha256(s).digest()[:4], 'big')`,
  s = x for strings, else `json.dumps(x, sort_keys=True, separators=(',', ':'))`; G work-list ID formats as in the JSON.

---

"""
    with open(md + ".tmp", "w") as f:
        f.write(header)
        f.write(open(ORDER).read())
    os.replace(md + ".tmp", md)
    print(out_json)
    print(md)


if __name__ == "__main__":
    main()
