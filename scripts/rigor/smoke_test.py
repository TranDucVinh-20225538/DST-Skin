#!/usr/bin/env python3
"""End-to-end smoke test of the rigor pack on tiny SYNTHETIC data (+ the real tracked CSVs).

Creates a fake repo layout in a temp dir (WILDS-like metadata.csv, feature caches for
8 archs x 5 seeds, two indexed caches), runs every CPU stage on it, runs w_from_csv.py on the
REAL tracked CSVs (output to the temp dir, not into outputs/reports), and runs unit checks:
weighted AUROC == sklearn, weighted coverage == coverage_at_risk, DeLong AUROC == sklearn,
vectorised W == repo kendall_w, Holm/BH sanity. Numbers produced here are meaningless.

    PYTHONPATH=. python scripts/rigor/smoke_test.py [--keep DIR]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

H2 = [("040", [20], 38, 0.0071), ("041", [21], 37, 0.0027), ("042", [22], 72, 0.5865), ("044", [23], 53, 0.5221),
      ("045", [24], 77, 0.0238), ("046", [25, 26], 81, 0.187), ("048", [27], 46, 0.0241), ("051", [28], 319, 0.8469),
      ("052", [29], 127, 0.5246)]


def unit_tests():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(0, 1, 500)
    a[:20] = np.round(a[:20], 1)
    b[:40] = np.round(b[:40], 1)
    y = np.r_[np.ones(300), np.zeros(500)]
    ref = roc_auc_score(y, np.r_[a, b].astype(np.float32))
    s = C.AurocSorter(a, b)
    assert abs(s.auroc() - ref) < 1e-12, (s.auroc(), ref)
    wi = rng.integers(0, 3, 300).astype(float)
    wo = rng.integers(0, 3, 500).astype(float)
    ref_w = roc_auc_score(y, np.r_[a, b].astype(np.float32), sample_weight=np.r_[wi, wo])
    assert abs(s.auroc(wi, wo) - ref_w) < 1e-10, (s.auroc(wi, wo), ref_w)
    corr = (rng.random(500) < 0.8).astype(float)
    cs = C.CoverageSorter(b, corr)
    for t in C.RISKS:
        assert abs(cs.coverage(None, t) - C.coverage_at_risk(b, corr, t)) < 1e-12
        m = rng.random(500) < 0.5
        assert abs(cs.coverage(m.astype(float), t) - C.coverage_at_risk(b[m], corr[m], t)) < 1e-12
    try:
        sys.path.insert(0, str(C.REPO / "scripts"))
        import camelyon_coverage_val_split as cvs  # imports torch + sklearn

        assert abs(cvs.coverage_at_risk(b, corr) - C.coverage_at_risk(b, corr)) < 1e-12
        print("unit: coverage_at_risk identical to scripts/camelyon_coverage_val_split.py")
    except ImportError as exc:
        print("unit: (skip repo coverage import: %s)" % exc)
    au, cov = C.delong([a, a + rng.normal(0, 0.5, 300)], [b, b + rng.normal(0, 0.5, 500)])
    assert abs(au[0] - ref) < 1e-9 and cov.shape == (2, 2)
    mats = rng.random((6, 7))
    mats[0, :2] = 0.5
    r = C.ranks_rows(mats)
    assert abs(C.kendall_w(r) - C._kendall_w_copy(r)) < 1e-12
    assert abs(float(C.kendall_w_batch(r, C.tie_terms(r))) - C.kendall_w(r)) < 1e-12
    assert np.allclose(C.ranks_rows(mats), np.stack([C._ranks_high_is_1_copy(x) for x in mats]))
    assert np.allclose(C.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(C.bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
    print("unit tests: OK (W source: %s)" % C.W_SOURCE)


def make_synth(root: Path):
    import torch

    rng = np.random.default_rng(1)
    rows = []
    pid = 0
    slide = 0
    for center in (0, 3, 4):
        for _ in range(4):
            p = "%03d" % (center * 20 + pid % 20)
            pid += 1
            for i in range(75):
                rows.append({"patient": p, "node": 0, "x_coord": i, "y_coord": 0, "tumor": int(rng.random() < 0.5),
                             "slide": slide, "center": center, "split": 0 if i < 60 else 1})
            slide += 1
    for j in range(2):
        for i in range(30):
            rows.append({"patient": "02%d" % j, "node": 0, "x_coord": i, "y_coord": 0, "tumor": int(rng.random() < 0.5),
                         "slide": 10 + j, "center": 1, "split": 0})
    for p, slides, n, tf in H2:
        for i in range(n):
            rows.append({"patient": p, "node": 0, "x_coord": i, "y_coord": 0, "tumor": int(rng.random() < tf),
                         "slide": slides[i % len(slides)], "center": 2, "split": 0})
    meta = pd.DataFrame(rows)
    mp = C.metadata_path(root)
    mp.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(mp)
    m2 = C.load_camelyon_metadata(root)
    tr = np.where(m2.wilds_split == 0)[0]
    iv = np.where(m2.wilds_split == 1)[0]
    oo = np.where(m2.wilds_split == 2)[0]
    d = 12
    mu = rng.normal(0, 1, (2, d))
    slide_off = rng.normal(0, 0.4, (40, d))
    shift = rng.normal(0, 1.0, d)
    pat_codes = {p: i for i, p in enumerate(sorted(m2.patient.unique()))}
    pat_off = rng.normal(0, 0.6, (len(pat_codes), d))

    def feats(idx, arch_rng, ood):
        y = m2.tumor.to_numpy()[idx]
        f = mu[y] + slide_off[m2.slide.to_numpy()[idx]] + arch_rng.normal(0, 1.0, (len(idx), d))
        if ood:
            f = f + shift * arch_rng.uniform(0.3, 1.2) + pat_off[[pat_codes[p] for p in m2.patient.to_numpy()[idx]]]
        return np.abs(f).astype(np.float32), y

    for ai, a in enumerate(C.ARCHS):
        for s in C.SEEDS:
            ar = np.random.default_rng(100 * ai + s)
            proj = np.eye(d) + ar.normal(0, 0.3, (d, d))
            W = (mu @ proj).astype(np.float32)
            bvec = np.zeros(2, np.float32)
            out = {}
            perm = ar.permutation(len(tr))
            for key, idx, ood in (("train", tr[perm], False), ("val", iv, False), ("ood", oo, True)):
                f, y = feats(idx, ar, ood)
                f = (f @ proj).astype(np.float32)
                out[key + "_feats"], out[key + "_labels"] = f, y.astype(np.int64)
                out[key + "_logits"] = (f @ W.T + bvec).astype(np.float32)
            out["fc_weight"], out["fc_bias"] = torch.tensor(W), torch.tensor(bvec)
            p = C.feature_path(root, "camelyon17", a, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, p)
            if s == 42 and a in ("resnet18", "densenet121"):
                o2 = dict(out)
                f, y = feats(tr, ar, False)
                o2["train_feats"] = (f @ proj).astype(np.float32)
                o2["train_labels"] = y
                o2["train_logits"] = (o2["train_feats"] @ W.T).astype(np.float32)
                o2.update({"train_idx": tr, "val_idx": iv, "ood_idx": oo,
                           "train_slide": m2.slide.to_numpy()[tr], "val_slide": m2.slide.to_numpy()[iv],
                           "ood_slide": m2.slide.to_numpy()[oo]})
                pi = C.feature_path(root, "camelyon17", a, s, indexed=True)
                pi.parent.mkdir(parents=True, exist_ok=True)
                torch.save(o2, pi)
    rp = root / "outputs/reports"
    rp.mkdir(parents=True, exist_ok=True)
    for f in ("camelyon_coverage_val_split_patients.csv", "camelyon_coverage_val_split.csv"):
        shutil.copy(C.REPO / "outputs/reports" / f, rp / f)


def write_synth_score_csvs(root: Path):
    """Score CSVs in the repo's format, from the synthetic score caches (so w_from_csv runs on them)."""
    from src.utils.benchmark_metrics import calc_auroc, calc_fpr95

    for a in C.ARCHS:
        for s in C.SEEDS:
            z = np.load(C.score_cache_path(C.default_out(root), "camelyon17", a, s))
            rows = []
            for m in C.METHOD_KEYS:
                fpr, thr = calc_fpr95(z["id_" + m], z["ood_" + m])
                rows.append({"Method": m, "AUROC": calc_auroc(z["id_" + m], z["ood_" + m]), "FPR95": fpr, "Threshold": thr})
            p = C.score_csv_path(root, "camelyon17", a, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(p, index=False)


def run(cmd, ok_codes=(0,)):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(C.REPO), env=dict(__import__("os").environ, PYTHONPATH=str(C.REPO)))
    if r.returncode not in ok_codes:
        raise SystemExit("FAILED (%d): %s" % (r.returncode, " ".join(str(c) for c in cmd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="keep synthetic root here")
    args = ap.parse_args()
    unit_tests()
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="rigor_smoke_"))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    make_synth(root)
    py = sys.executable
    R = HERE
    rep = root / "outputs/reports/rigor_pack"
    common = ["--root", root, "--feat-root", root]
    run([py, R / "check_env.py"])
    # REAL tracked CSVs: written OUTSIDE the synthetic report tree so multiple_testing/make_tables
    # never aggregate or print real results during a smoke test (precommit hygiene).
    run([py, R / "w_from_csv.py", "--reports", root / "_real_csv_smoke" / "w_csv", "--n-perm", "200"])
    run([py, R / "power_w.py", "--reports", rep / "power", "--n-sim", "10", "--n-perm", "50"])
    run([py, R / "hospital2_patients.py"] + common + ["--reports", rep / "hospital2"])
    run([py, R / "check_inputs.py"] + common + ["--reports", rep])
    run([py, R / "build_score_cache.py"] + common + ["--list"])
    run([py, R / "build_score_cache.py"] + common + ["--task-id", "0"])
    run([py, R / "build_score_cache.py"] + common)
    write_synth_score_csvs(root)
    run([py, R / "w_from_csv.py", "--root", root, "--reports", rep / "w_csv", "--n-perm", "200", "--domains", "camelyon17"],
        ok_codes=(0, 2))  # 2 = REPRO mismatch, expected on synthetic data
    run([py, R / "persample.py"] + common + ["--B", "30"])
    run([py, R / "coverage_splits.py"] + common)
    run([py, R / "leakfree_knn.py"] + common + ["--archs", "resnet18", "densenet121", "--k", "5"])
    run([py, R / "extract_features_indexed.py", "--feat-root", root, "--dry-run", "--seeds", "42"])
    run([py, R / "multiple_testing.py", "--reports", rep])
    run([py, R / "vit_seeds.py", "--root", root, "--reports", rep / "vit_seeds"])
    run([py, R / "transfer_regret.py", "--root", root, "--reports", rep / "transfer_regret", "--domains", "camelyon17"],
        ok_codes=(0, 2))  # 2 = REPRO mismatch, expected on synthetic data
    run([py, R / "w_ties.py", "--root", root, "--reports", rep / "w_ties", "--domains", "camelyon17"], ok_codes=(0, 2))
    run([py, R / "recipe_seeds.py", "--root", root, "--reports", rep / "recipe_seeds"], ok_codes=(0, 2))
    run([py, R / "make_tables.py", "--root", root, "--reports", rep])
    if not (root / "_real_csv_smoke/w_csv/repro_checks.csv").exists():
        raise SystemExit("real-CSV stage produced no repro_checks.csv")
    need = ["hospital2/hospital2_patients.csv", "hospital2/leakage_checks.csv",
            "persample_camelyon17/auroc_cells.csv", "persample_camelyon17/w_bootstrap.csv",
            "persample_camelyon17/coverage_cells.csv", "coverage_splits/splits_summary.csv",
            "leakage/leakfree_knn.csv", "multiple_testing/pvalues_adjusted.csv", "tables/T5_coverage_splits.tex",
            "fig/fig1b_valsplit.pdf"]
    miss = [n for n in need if not (rep / n).exists()]
    if miss:
        raise SystemExit("missing outputs: %s" % miss)
    print("\nSMOKE TEST PASSED. synthetic root: %s" % root)


if __name__ == "__main__":
    main()
