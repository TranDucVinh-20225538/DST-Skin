import torch
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


from .ood_vim_react import (
    fit_vim,
    vim_score,
    fit_react_on_fc,
    react_energy_score,
)

class OODScorer:
    def __init__(
        self,
        k_nearest: int = 50,
        use_react: bool = False,
        react_percentile: float = 90.0,
        use_vim: bool = False,
        vim_dim: int = None,
    ):
        self.k = k_nearest
        self.mu = None
        self.precision = None
        self.knn_index = None
        self.train_features_norm = None

        self.use_react = use_react
        self.react_percentile = react_percentile
        self.react_params = None

        self.use_vim = use_vim
        self.vim_dim = vim_dim
        self.vim_params = None

    # ====================== 1. FIT TRÊN TRAIN FEATURES ======================

    def fit(
        self,
        train_features: np.ndarray,
        train_labels=None,
        train_logits: np.ndarray = None,
        fc_weight: np.ndarray = None,
        fc_bias: np.ndarray = None,
    ):
        train_features_norm = self.l2_normalize(train_features)

        lw = LedoitWolf().fit(train_features_norm)
        self.mu = lw.location_
        self.precision = lw.precision_

        self.knn_index = NearestNeighbors(
            n_neighbors=self.k,
            metric="cosine",
            algorithm="auto"
        )
        self.knn_index.fit(train_features_norm)
        self.train_features_norm = train_features_norm

        if self.use_react:
            if fc_weight is None or fc_bias is None:
                raise ValueError("ReAct needs fc_weight and fc_bias.")
            self.react_params = fit_react_on_fc(
                train_feats=train_features,
                fc_weight=fc_weight,
                fc_bias=fc_bias,
                percentile=self.react_percentile,
            )

        if self.use_vim:
            if train_logits is None or fc_weight is None or fc_bias is None:
                raise ValueError("ViM needs train_logits, fc_weight, fc_bias.")
            self.vim_params = fit_vim(
                train_feats=train_features,
                train_logits=train_logits,
                fc_weight=fc_weight,
                fc_bias=fc_bias,
                d=self.vim_dim,
            )

        print(f"✅ OODScorer fitted với {len(train_features_norm)} mẫu (k={self.k}).")

    def get_all_scores(self, logits: np.ndarray, features: np.ndarray, T: float = 1.0):
        feats_n = self.l2_normalize(features)

        msp = self.score_msp(logits)
        energy = self.score_energy(logits, T)
        logit_norm = np.linalg.norm(logits, axis=1)
        cvid_logit_norm = np.std(logit_norm) / (np.abs(np.mean(logit_norm)) + 1e-8)

        maha = self.score_mahalanobis(feats_n)
        knn = self.score_knn(feats_n)

        scores = {
            "msp": msp,
            "energy": energy,
            "logit_norm": logit_norm,
            "mahalanobis": maha,
            "knn": knn,
            "cvid_logit_norm": cvid_logit_norm,
        }

        if self.use_react and self.react_params is not None:
            scores["react_energy"] = react_energy_score(features, self.react_params, T=T)

        if self.use_vim and self.vim_params is not None:
            scores["vim"] = -vim_score(features, self.vim_params)

        return scores

    # ====================== 3. HÀM CON CHI TIẾT ======================

    @staticmethod
    def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + eps)

    # ---- Logit-based ----
    @staticmethod
    def score_msp(logits: np.ndarray) -> np.ndarray:
        """
        Maximum Softmax Probability (higher = more ID/confident).
        """
        probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        return probs.max(axis=1)

    @staticmethod
    def score_energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
        """
        Energy-based score (Liu et al. 2020).
        Trả về: higher = more ID/confident.
        """
        z = torch.from_numpy(logits) / T
        energy = -T * torch.logsumexp(z, dim=1)
        return (-energy).numpy()

    # ---- Mahalanobis ----
    def score_mahalanobis(self, feats_norm: np.ndarray) -> np.ndarray:
        if self.mu is None or self.precision is None:
            raise RuntimeError("Mahalanobis hasn't been fit. Call fit() first.")

        diff = feats_norm - self.mu
        left = diff @ self.precision
        d2 = np.sum(left * diff, axis=1)
        dist = np.sqrt(d2)
        return -dist

    # ---- k-NN ----
    def score_knn(self, feats_norm: np.ndarray) -> np.ndarray:
        
        if self.knn_index is None:
            raise RuntimeError("k-NN hasn't been fit. Call fit() first.")

        dists, _ = self.knn_index.kneighbors(feats_norm)
        return -dists.mean(axis=1)

    # ====================== 4. ECE (CALIBRATION METRIC) ======================

    @staticmethod
    def compute_ece(probs: np.ndarray,
                    labels: np.ndarray,
                    n_bins: int = 15) -> float:
        
        if probs.ndim == 2:
            
            probs = probs[:, 1]

        preds = (probs > 0.5).astype(int)
        confidences = np.maximum(probs, 1 - probs)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop = np.mean(in_bin)
            if prop > 0:
                acc = np.mean(labels[in_bin] == preds[in_bin])
                conf = np.mean(confidences[in_bin])
                ece += np.abs(conf - acc) * prop

        return ece