#!/usr/bin/env python3
"""recompute_paper_ci.py with the crossfit_ood "knn" scorer computed on GPU (R3 item 1, Camelyon).

Same definition as crossfit_ood.scorers.KNNScorer: L2-normalise (x / max(||x||, 1e-12)), exact
Euclidean distance to the k-th nearest fit sample (k capped at n_fit), score = -distance, all in
float64. Exact brute force (no approximate search). Everything else is the unchanged script.

    python recompute_gpu_knn.py <recompute_paper_ci.py args>
    python recompute_gpu_knn.py --check FEATURES.npz [--n-fit N] [--n-query N]   # score-level check
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from crossfit_ood import scorers as S

CHUNK = int(os.environ.get("KNN_CHUNK", "2048"))


class GPUKNNScorer:
    def __init__(self, k: int = 50, normalize: bool = True) -> None:
        self.k = k
        self.normalize = normalize

    def _prep(self, x):
        x = torch.from_numpy(np.ascontiguousarray(S._as_f64(x))).to("cuda")
        if self.normalize:
            x = x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=1e-12)
        return x

    def fit(self, features, labels=None):
        self.t_ = self._prep(features)
        self.k_ = int(min(self.k, self.t_.shape[0]))
        self.tn_ = (self.t_ * self.t_).sum(1)
        return self

    def score(self, features):
        q = self._prep(features)
        out = torch.empty(q.shape[0], dtype=torch.float64, device="cuda")
        for i in range(0, q.shape[0], CHUNK):
            qc = q[i:i + CHUNK]
            d2 = (qc * qc).sum(1, keepdim=True) - 2.0 * qc @ self.t_.T + self.tn_[None, :]
            out[i:i + CHUNK] = torch.topk(d2, self.k_, dim=1, largest=False).values[:, -1]
        return -torch.sqrt(torch.clamp(out, min=0.0)).cpu().numpy()

    def __deepcopy__(self, memo):
        return GPUKNNScorer(self.k, self.normalize)


class GPUKNNMeanCosineScorer:
    """crossfit_ood knn_mean_cosine (Track A) on GPU: x / (||x|| + 1e-8), then cosine distance
    1 - cos(q, t) (sklearn's metric="cosine" renormalises, so exact unit vectors are used here),
    score = -mean distance to the k nearest fit samples; exact brute force, float64."""

    def __init__(self, k: int = 50) -> None:
        self.k = k

    @staticmethod
    def _unit(x):
        x = torch.from_numpy(np.ascontiguousarray(S._l2_tracka(x))).to("cuda")
        return x / torch.linalg.norm(x, dim=1, keepdim=True)

    def fit(self, features, labels=None):
        self.t_ = self._unit(features)
        self.k_ = int(min(self.k, self.t_.shape[0]))
        return self

    def score(self, features):
        q = self._unit(features)
        out = torch.empty(q.shape[0], dtype=torch.float64, device="cuda")
        for i in range(0, q.shape[0], CHUNK):
            sim = q[i:i + CHUNK] @ self.t_.T
            out[i:i + CHUNK] = (1.0 - torch.topk(sim, self.k_, dim=1, largest=True).values).mean(1)
        return -out.cpu().numpy()

    def __deepcopy__(self, memo):
        return GPUKNNMeanCosineScorer(self.k)


def check(path, n_fit, n_query):
    z = np.load(path, allow_pickle=True)
    rng = np.random.default_rng(0)
    xtr, xq = z["features_train"], np.concatenate([z["features_id_eval"], z["features_ood"]])
    xtr = xtr[np.sort(rng.choice(len(xtr), min(n_fit, len(xtr)), replace=False))]
    xq = xq[np.sort(rng.choice(len(xq), min(n_query, len(xq)), replace=False))]
    for name, cpu, gpu in (("knn", S.KNNScorer, GPUKNNScorer), ("knn_mean_cosine", S.KNNMeanCosineScorer,
                                                                GPUKNNMeanCosineScorer)):
        a = cpu(k=50).fit(xtr).score(xq)
        b = gpu(k=50).fit(xtr).score(xq)
        print("check %s %s n_fit=%d n_query=%d d=%d max_abs_diff=%.3e max_rel_diff=%.3e"
              % (name, os.path.basename(path), len(xtr), len(xq), xtr.shape[1], np.max(np.abs(a - b)),
                 np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--check", required=True)
        ap.add_argument("--n-fit", type=int, default=10**9)
        ap.add_argument("--n-query", type=int, default=10**9)
        a = ap.parse_args()
        check(a.check, a.n_fit, a.n_query)
        raise SystemExit(0)
    S._REGISTRY["knn"] = GPUKNNScorer
    S._REGISTRY["knn_mean_cosine"] = GPUKNNMeanCosineScorer
    sys.path.insert(0, os.path.join(os.environ["CROSSFIT_DIR"], "scripts"))
    import recompute_paper_ci as R
    R.get_scorer = S.get_scorer
    R.main(sys.argv[1:])
