import numpy as np
from numpy.linalg import pinv, norm
from scipy.special import logsumexp
from typing import Dict, Any
from sklearn.covariance import EmpiricalCovariance


def compute_react_tau(train_feats: np.ndarray, percentile: float = 90.0) -> float:
    return float(np.percentile(train_feats, percentile))


def apply_react(feats: np.ndarray, tau: float) -> np.ndarray:
    return np.clip(feats, a_min=None, a_max=tau)


def fit_react_on_fc(
    train_feats: np.ndarray,
    fc_weight: np.ndarray,
    fc_bias: np.ndarray,
    percentile: float = 90.0,
) -> Dict[str, Any]:
    tau = compute_react_tau(train_feats, percentile)
    return {
        "tau": tau,
        "w": fc_weight.astype(np.float32),
        "b": fc_bias.astype(np.float32),
    }


def react_energy_score(
    feats: np.ndarray,
    params: Dict[str, Any],
    T: float = 1.0,
) -> np.ndarray:
    tau = params["tau"]
    w = params["w"]
    b = params["b"]

    feats_clip = apply_react(feats, tau)
    logits = feats_clip @ w.T + b
    energy = T * logsumexp(logits / T, axis=1)
    return energy.astype(np.float32)


def fit_vim(
    train_feats: np.ndarray,
    train_logits: np.ndarray,
    fc_weight: np.ndarray,
    fc_bias: np.ndarray,
    d: int = None,
) -> Dict[str, Any]:
    w = np.asarray(fc_weight, dtype=np.float64)
    b = np.asarray(fc_bias, dtype=np.float64)
    train_feats = np.asarray(train_feats, dtype=np.float64)
    train_logits = np.asarray(train_logits, dtype=np.float64)

    if d is None:
        d = train_feats.shape[1] - train_logits.shape[1]
    d = max(1, min(d, train_feats.shape[1] - 1))

    # new origin
    u = -np.matmul(pinv(w), b)

    # covariance around new origin
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feats - u)

    eig_vals, eigen_vectors = np.linalg.eigh(ec.covariance_)

    # official logic: keep residual subspace after top-d principal dimensions
    largest_eigvals_idx = np.argsort(eig_vals * -1)[d:]
    principal_subspace = np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)

    x_p_t = np.matmul(train_feats - u, principal_subspace)
    vlogits = norm(x_p_t, axis=-1)

    alpha = train_logits.max(axis=-1).mean() / (vlogits.mean() + 1e-12)

    return {
        "u": u.astype(np.float32),
        "principal_subspace": principal_subspace.astype(np.float32),
        "alpha": float(alpha),
        "w": w.astype(np.float32),
        "b": b.astype(np.float32),
        "d": int(d),
    }


def vim_score(feats: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float64)

    u = params["u"].astype(np.float64)
    P = params["principal_subspace"].astype(np.float64)
    alpha = float(params["alpha"])
    w = params["w"].astype(np.float64)
    b = params["b"].astype(np.float64)

    logits = np.matmul(feats, w.T) + b
    x_p_t = norm(np.matmul(feats - u, P), axis=-1)
    vlogit = x_p_t * alpha
    energy = logsumexp(np.clip(logits, -100, 100), axis=-1)

    score = -vlogit + energy
    return (-score).astype(np.float32)