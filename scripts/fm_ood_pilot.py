#!/usr/bin/env python3
"""Frozen pathology FM OOD probe — Camelyon17 + MIDOG only.

Does not touch Camelyon CNN zoo (59493) or CIFAR-10-C.
Encoder frozen. Linear head on ID-train features for MSP/Energy/ViM/ReAct.
Maha/kNN on frozen features.
"""

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.models.pathology_fm import (
    SPECS,
    eval_transform,
    load_frozen_encoder,
    mismatch_record,
)
from src.utils.benchmark_metrics import build_score_comparison_df
from src.utils.scoring import OODScorer

CNN_BASELINE = {
    "camelyon17": Path(
        "outputs/reports/camelyon17/frac1/seed42/resnet50_score_comparison.csv"
    ),
    "midog": Path("outputs/reports/midog/seed42/resnet50_score_comparison.csv"),
}
METHODS = (
    "msp",
    "energy",
    "logit_norm",
    "react_energy",
    "mahalanobis",
    "knn",
    "vim",
)
VIM_TRAIN_CAP = 30_000
SEED = 42


def out_dir(domain: str, fm: str) -> Path:
    return Path(f"outputs/features/fm/{domain}/{fm}")


def report_dir(domain: str, fm: str) -> Path:
    return Path(f"outputs/reports/fm/{domain}/{fm}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def check_access(names: list[str]) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        from huggingface_hub.utils import GatedRepoError
    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = Exception  # type: ignore[misc,assignment]

    status = {}
    for name in names:
        spec = SPECS[name]
        try:
            hf_hub_download(spec.repo, filename="config.json")
            status[name] = "ok"
            print(f"ACCESS {name}: ok ({spec.repo})", flush=True)
        except GatedRepoError as exc:
            status[name] = f"gated:{exc}"
            print(
                f"ACCESS {name}: GATED — accept terms at {spec.access_url}",
                flush=True,
            )
        except HfHubHTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            status[name] = f"http:{code}"
            print(f"ACCESS {name}: HTTP {code} {spec.repo}", flush=True)
        except Exception as exc:
            # some repos use pytorch_model.bin without config.json at root
            try:
                hf_hub_download(spec.repo, filename="pytorch_model.bin")
                status[name] = "ok"
                print(f"ACCESS {name}: ok via pytorch_model.bin", flush=True)
            except GatedRepoError:
                status[name] = "gated"
                print(f"ACCESS {name}: GATED — {spec.access_url}", flush=True)
            except Exception as exc2:
                status[name] = f"error:{type(exc2).__name__}"
                print(f"ACCESS {name}: FAIL {type(exc).__name__}/{type(exc2).__name__}", flush=True)
    Path("outputs/reports/fm").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/fm/access_status.json").write_text(json.dumps(status, indent=2))
    return status


def loaders_for(domain: str, spec, batch_size: int, num_workers: int):
    tf = eval_transform(spec)
    if domain == "camelyon17":
        from src.datasets.camelyon_ood import get_dataloaders

        return get_dataloaders(
            backbone="resnet18",
            batch_size=batch_size,
            num_workers=num_workers,
            include_ood=True,
            train_frac=1.0,
            seed=SEED,
            transform=tf,
            shuffle_train=False,
        )
    if domain == "midog":
        from src.datasets.midog_ood import get_dataloaders

        return get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            include_ood=True,
            transform=tf,
        )
    raise ValueError(domain)


def id_eval_key(domain: str) -> str:
    return "id_val" if domain == "camelyon17" else "id_test"


def extract_split(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", leave=False):
            imgs, y, *_ = batch
            out = model(imgs.to(device, non_blocking=True))
            feats.append(out.detach().float().cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def fit_linear_head(train_feats: np.ndarray, train_labels: np.ndarray):
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
    )
    clf.fit(train_feats, train_labels)
    W = np.asarray(clf.coef_, dtype=np.float32)
    b = np.asarray(clf.intercept_, dtype=np.float32)
    n_cls = int(W.shape[0] if W.ndim == 2 else 1)

    def logits(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        z = X @ W.T + b
        if z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1):
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            return np.stack([-z, z], axis=1)
        return np.asarray(z, dtype=np.float32)

    return logits, W, b, n_cls, float(clf.score(train_feats, train_labels))


def csv_auroc(path: Path, method: str) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hit = df[df["Method"].astype(str) == method]
    if hit.empty:
        return None
    return float(hit.iloc[0]["AUROC"])


def verdict_line(
    domain: str,
    fm: str,
    maha: float,
    msp: float,
    knn: float,
    cnn_maha: float | None,
    cnn_msp: float | None,
) -> str:
    fm_sign = maha > msp
    if cnn_maha is None or cnn_msp is None:
        return (
            f"{domain}/{fm}: Maha {maha:.3f} vs MSP {msp:.3f} "
            f"(kNN {knn:.3f}); no CNN baseline CSV"
        )
    cnn_sign = cnn_maha > cnn_msp
    if fm_sign != cnn_sign:
        kind = "REVERSE"
    elif maha < 0.70 and cnn_maha >= 0.85:
        kind = "NEW_REGIME (FM Maha collapse)"
    elif maha >= 0.85 and cnn_maha < 0.70:
        kind = "NEW_REGIME (FM Maha rescue)"
    else:
        kind = "REPLICATE"
    return (
        f"{domain}/{fm}: {kind} — FM Maha {maha:.3f} MSP {msp:.3f} kNN {knn:.3f} "
        f"| R50 CNN Maha {cnn_maha:.3f} MSP {cnn_msp:.3f}"
    )


def extract_one(domain: str, fm: str, batch_size: int, num_workers: int) -> Path:
    device = get_device()
    model, spec = load_frozen_encoder(fm)
    model.to(device)
    rec = mismatch_record(domain, spec)
    print(json.dumps(rec, indent=2), flush=True)
    loaders = loaders_for(domain, spec, batch_size, num_workers)
    splits = {}
    keys = ["train", id_eval_key(domain), "ood"]
    for key in keys:
        print(f"=== extract {domain}/{fm}/{key} n={len(loaders[key].dataset)} ===", flush=True)
        feats, labels = extract_split(model, loaders[key], device)
        splits[key] = {"feats": feats, "labels": labels}
        print(f"  feats {feats.shape}", flush=True)
    dest = out_dir(domain, fm)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_feats": splits["train"]["feats"],
        "train_labels": splits["train"]["labels"],
        "val_feats": splits[id_eval_key(domain)]["feats"],
        "val_labels": splits[id_eval_key(domain)]["labels"],
        "ood_feats": splits["ood"]["feats"],
        "ood_labels": splits["ood"]["labels"],
        "mismatch": rec,
        "fm": fm,
        "repo": spec.repo,
        "domain": domain,
        "id_eval_split": id_eval_key(domain),
    }
    path = dest / "features.pt"
    torch.save(payload, path)
    (dest / "mismatch.json").write_text(json.dumps(rec, indent=2))
    print(f"Saved {path}", flush=True)
    del model, loaders, splits, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def analyze_one(domain: str, fm: str) -> dict:
    feat_path = out_dir(domain, fm) / "features.pt"
    try:
        data = torch.load(feat_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(feat_path, map_location="cpu")
    train_feats = np.asarray(data["train_feats"], dtype=np.float32)
    train_labels = np.asarray(data["train_labels"])
    val_feats = np.asarray(data["val_feats"], dtype=np.float32)
    ood_feats = np.asarray(data["ood_feats"], dtype=np.float32)

    logits_fn, W, b, n_cls, train_acc = fit_linear_head(train_feats, train_labels)
    print(
        f"linear probe train_acc={train_acc:.4f} n_cls={n_cls} d={train_feats.shape[1]}",
        flush=True,
    )
    train_logits = logits_fn(train_feats)
    val_logits = logits_fn(val_feats)
    ood_logits = logits_fn(ood_feats)

    scorer = OODScorer(
        k_nearest=50,
        use_react=True,
        react_percentile=90.0,
        use_vim=False,
        vim_dim=None,
    )
    scorer.fit(
        train_feats,
        train_labels=None,
        train_logits=train_logits,
        fc_weight=W,
        fc_bias=b,
    )
    from src.utils.ood_vim_react import fit_vim

    vim_feats, vim_logits = train_feats, train_logits
    if len(train_feats) > VIM_TRAIN_CAP:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train_feats), VIM_TRAIN_CAP, replace=False)
        vim_feats = train_feats[idx]
        vim_logits = train_logits[idx]
        print(f"ViM subsample n={VIM_TRAIN_CAP}", flush=True)
    scorer.use_vim = True
    scorer.vim_params = fit_vim(vim_feats, vim_logits, W, b, d=None)

    scores_id = scorer.get_all_scores(val_logits, val_feats)
    scores_ood = scorer.get_all_scores(ood_logits, ood_feats)
    methods = [m for m in METHODS if m in scores_id]
    df = build_score_comparison_df(scores_id, scores_ood, methods, seed=SEED)
    rdir = report_dir(domain, fm)
    rdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(rdir / "score_comparison.csv", index=False)

    maha = float(df.loc[df["Method"] == "mahalanobis", "AUROC"].iloc[0])
    msp = float(df.loc[df["Method"] == "msp", "AUROC"].iloc[0])
    knn = float(df.loc[df["Method"] == "knn", "AUROC"].iloc[0])
    cnn_path = CNN_BASELINE[domain]
    cnn_maha = csv_auroc(cnn_path, "mahalanobis")
    cnn_msp = csv_auroc(cnn_path, "msp")
    line = verdict_line(domain, fm, maha, msp, knn, cnn_maha, cnn_msp)
    print(line, flush=True)
    rec = {
        "domain": domain,
        "fm": fm,
        "repo": str(data.get("repo", "")),
        "n_train": int(len(train_feats)),
        "feat_dim": int(train_feats.shape[1]),
        "linear_train_acc": train_acc,
        "mahalanobis_auroc": maha,
        "msp_auroc": msp,
        "knn_auroc": knn,
        "cnn_r50_mahalanobis_auroc": cnn_maha,
        "cnn_r50_msp_auroc": cnn_msp,
        "verdict": line,
        "mismatch": data.get("mismatch", {}),
        "cnn_baseline_csv": str(cnn_path),
        "cnn_r50_note": (
            "midog uses in-house R50 score CSV (not OpenMIBOOD 0.59 freeze)"
            if domain == "midog"
            else "camelyon full-train frac1 seed42 R50"
        ),
    }
    (rdir / "verdict.json").write_text(json.dumps(rec, indent=2, default=str))
    (rdir / "verdict.txt").write_text(line + "\n")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("access", "extract", "analyze", "all"),
        default="all",
    )
    parser.add_argument(
        "--fm",
        nargs="+",
        default=["uni", "gigapath", "phikon"],
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        default=["midog", "camelyon17"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    names = args.fm
    domains = args.domain
    Path("outputs/reports/fm").mkdir(parents=True, exist_ok=True)

    if args.stage in ("access", "all"):
        status = check_access(names)
        blocked = [k for k, v in status.items() if v != "ok"]
        if args.stage == "access":
            return
        if blocked:
            print(f"Skip extract for gated/failed: {blocked}", flush=True)
            names = [n for n in names if status.get(n) == "ok"]
            if not names:
                raise SystemExit("No FM accessible yet.")

    rows = []
    for fm in names:
        for domain in domains:
            feat = out_dir(domain, fm) / "features.pt"
            if args.stage in ("extract", "all") and not feat.exists():
                extract_one(domain, fm, args.batch_size, args.num_workers)
            if args.stage in ("analyze", "all"):
                if not feat.exists():
                    print(f"missing {feat}, skip analyze", flush=True)
                    continue
                rows.append(analyze_one(domain, fm))

    if rows:
        df = pd.DataFrame(rows)
        out = Path("outputs/reports/fm/summary.csv")
        df.to_csv(out, index=False)
        print(f"\nWrote {out}")
        print(df[["domain", "fm", "mahalanobis_auroc", "msp_auroc", "knn_auroc"]].to_string(index=False))
        print("\n".join(df["verdict"].tolist()))


if __name__ == "__main__":
    main()
