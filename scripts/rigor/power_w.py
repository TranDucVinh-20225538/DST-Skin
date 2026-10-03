#!/usr/bin/env python3
"""A priori power / sensitivity simulation for the arch-vs-seed tests (no data used).

Generative model for one training run (arch a, seed s) over M=7 detectors:
    AUROC-like profile y[m] = mu[m] + alpha[a, m] + eps[a, s, m]
    mu      : evenly spaced, gap 1 (only ranks matter, so units are arbitrary)
    alpha   ~ N(0, sd_arch^2)  architecture-specific method preference (the effect of interest)
    eps     ~ N(0, sd_seed^2)  seed noise
rho = sd_arch^2 / (sd_arch^2 + sd_seed^2) = share of ranking-relevant variance due to architecture.
sd_seed is calibrated so that E[cross-seed W] at rho=0 is 0.60 / 0.78 / 0.93 (the range
already published for the 4 anchors; only used to set a realistic noise scale).

Reports, per (A in {4, 8}, S=5): expected cross-seed W, expected cross-arch W (k=A at one
seed), and power at alpha=0.05 of
  (i)  the arch-label permutation test on mean within-arch cross-seed W (w_from_csv.py),
  (ii) the nested F test method:arch vs method:seed(arch) (parametric p).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

M = 7


def ranks_high1(y: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-y, axis=-1), axis=-1).astype(np.float64) + 1.0


def sim_runs(rng, A, S, sd_a, sd_s):
    mu = np.arange(M, dtype=np.float64)[::-1]
    alpha = rng.normal(0, sd_a, (A, 1, M)) if sd_a > 0 else np.zeros((A, 1, M))
    eps = rng.normal(0, sd_s, (A, S, M))
    return mu + alpha + eps


def expected_w_seed(rng, sd_s, S=5, n=2000):
    y = sim_runs(rng, n, S, 0.0, sd_s)
    return float(np.mean(C.kendall_w_batch(ranks_high1(y))))


def calibrate(rng, target, S=5):
    lo, hi = 0.01, 20.0
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if expected_w_seed(rng, mid, S) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def nested_F(y):
    A, S, Mm = y.shape
    g = y.mean()
    ym, ya, yas, yma = y.mean(axis=(0, 1)), y.mean(axis=(1, 2)), y.mean(axis=2), y.mean(axis=1)
    ss_ma = S * np.sum((yma - ym[None, :] - ya[:, None] + g) ** 2)
    ss_ms = np.sum((y - yma[:, None, :] - yas[:, :, None] + ya[:, None, None]) ** 2)
    return (ss_ma / ((Mm - 1) * (A - 1))) / (ss_ms / ((Mm - 1) * A * (S - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=None)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    out = Path(args.reports) if args.reports else C.default_reports(C.REPO) / "power"
    C.ensure_dir(out)
    rng = np.random.default_rng(args.seed)
    try:
        from scipy.stats import f as fdist
    except Exception:
        fdist = None
    S = 5
    rows = []
    for target in (0.60, 0.78, 0.93):
        sd_s = calibrate(rng, target, S)
        for rho in (0.0, 0.1, 0.2, 0.3, 0.5):
            sd_a = np.sqrt(rho / (1 - rho)) * sd_s if rho > 0 else 0.0
            for A in (4, 8):
                hits_perm = hits_f = 0
                ws, wa = [], []
                for _ in range(args.n_sim):
                    y = sim_runs(rng, A, S, sd_a, sd_s)
                    rk = ranks_high1(y)
                    obs = float(np.mean(C.kendall_w_batch(rk)))
                    ws.append(obs)
                    wa.append(float(C.kendall_w_batch(rk[:, 0, :])))
                    units = rk.reshape(A * S, M)
                    perms = np.argsort(rng.random((args.n_perm, A * S)), axis=1)
                    null = C.kendall_w_batch(units[perms].reshape(args.n_perm, A, S, M)).mean(axis=1)
                    if (np.sum(null >= obs - 1e-12) + 1) / (args.n_perm + 1) < 0.05:
                        hits_perm += 1
                    if fdist is not None:
                        fv = nested_F(rk)
                        if fdist.sf(fv, (M - 1) * (A - 1), (M - 1) * A * (S - 1)) < 0.05:
                            hits_f += 1
                rows.append({"target_Wseed_rho0": target, "sd_seed": sd_s, "rho_arch_share": rho, "A": A, "S": S,
                             "E_cross_seed_W": float(np.mean(ws)), "E_cross_arch_W_kA": float(np.mean(wa)),
                             "power_arch_label_perm": hits_perm / args.n_sim,
                             "power_nested_F": (hits_f / args.n_sim) if fdist is not None else np.nan,
                             "n_sim": args.n_sim})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "power_w_simulation.csv", index=False)
    lines = ["A priori sensitivity of the arch-vs-seed tests (simulation, no study data).",
             "rho = share of ranking-relevant variance due to architecture; rho=0 row = type-I error.",
             df.round(3).to_string(index=False)]
    C.write_text(out / "power_w_simulation.txt", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
