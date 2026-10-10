#!/usr/bin/env python3
"""T1 item 0: environment record, GPU microbenchmarks, GPU-kNN equality check (rule 4).

    item0.py env                 -> results/t1/0/env.json
    item0.py bench               -> results/t1/0/throughput_<gpu>.json   (60 s budget per GPU type)
    item0.py knncheck CELL SEED  -> results/t1/0/knn_equality_<cell>.json (k = 1 and k = 50)
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "0"


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e}"


def env():
    import scipy
    import sklearn
    rec = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
           "scikit_learn": sklearn.__version__, "hostname": platform.node()}
    try:
        import torch
        rec.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        rec["torch"] = f"ERROR {e}"
    rec["nvidia_smi_L"] = _sh("nvidia-smi -L")
    rec["sinfo"] = _sh('sinfo -o "%P %a %l %D %t %G %N"')
    rec["crossfit_ood_version"] = _sh(f"{sys.executable} -m pip show crossfit-ood | head -2")
    CM.atomic_write_text(OUT / "env.json", json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def bench():
    torch = CM.set_float64_torch()
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    res = {"gpu": name, "dtype": "float64"}
    t_total = time.time()
    a = torch.randn(4096, 4096, dtype=torch.float64, device=dev)
    b = torch.randn(4096, 4096, dtype=torch.float64, device=dev)
    torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while time.time() - t0 < 10:
        a @ b
        n += 1
        torch.cuda.synchronize()
    res["matmul4096_tflops"] = 2 * 4096 ** 3 * n / (time.time() - t0) / 1e12
    x = torch.randn(5000, 2560, dtype=torch.float64, device=dev)
    s = x.T @ x / 5000 + 0.1 * torch.eye(2560, dtype=torch.float64, device=dev)
    for fn, key in ((torch.linalg.inv, "inv2560_s"), (torch.linalg.cholesky, "cholesky2560_s")):
        fn(s)
        torch.cuda.synchronize()
        t0, n = time.time(), 0
        while time.time() - t0 < 5:
            fn(s)
            n += 1
            torch.cuda.synchronize()
        res[key] = (time.time() - t0) / n
    q = torch.randn(50000, 2560, dtype=torch.float64, device=dev)
    t = torch.randn(100000, 2560, dtype=torch.float64, device=dev)
    tn = (t * t).sum(1)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, 50000, 2048):
        qc = q[i:i + 2048]
        d2 = (qc * qc).sum(1, keepdim=True) - 2.0 * qc @ t.T + tn[None, :]
        torch.topk(d2, 200, dim=1, largest=False)
    torch.cuda.synchronize()
    res["knn_50k_x_100k_x_2560_topk200_s"] = time.time() - t0
    res["wall_s"] = time.time() - t_total
    tag = name.replace(" ", "_")
    CM.atomic_write_text(OUT / f"throughput_{tag}.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def knncheck(cell, seed):
    sys.path.insert(0, str(CM.REPO / "scripts" / "r3"))
    CM.set_float64_torch()
    import recompute_gpu_knn as G
    from crossfit_ood import scorers as S
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, seed)
    rows = []
    for k in (1, 50):
        for name, cpu, gpu in (("knn_mean_cosine", S.KNNMeanCosineScorer, G.GPUKNNMeanCosineScorer),
                               ("knn", S.KNNScorer, G.GPUKNNScorer)):
            t0 = time.time()
            c = CM.paper2fold_delta(cpu(k=k), xtr, gtr, xid, gid, xood, fm, seed=seed)
            tc = time.time() - t0
            t0 = time.time()
            g = CM.paper2fold_delta(gpu(k=k), xtr, gtr, xid, gid, xood, fm, seed=seed)
            tg = time.time() - t0
            # score-level difference on fold 0's fit set
            tf = __import__("crossfit_ood.core", fromlist=["x"])._apply_map(fm[0], gtr)
            fit = xtr[tf == 0]
            q = np.concatenate([xid[:5000], xood[:5000]])
            sc, sg = cpu(k=k).fit(fit).score(q), gpu(k=k).fit(fit).score(q)
            rows.append(dict(cell=cell, scorer=name, k=k, cpu_auroc_seen=c[0], cpu_auroc_unseen=c[1], cpu_delta=c[2],
                             gpu_auroc_seen=g[0], gpu_auroc_unseen=g[1], gpu_delta=g[2],
                             abs_diff_delta=abs(c[2] - g[2]),
                             max_abs_diff_auroc=max(abs(c[0] - g[0]), abs(c[1] - g[1])),
                             max_abs_diff_score_fold0_10k=float(np.max(np.abs(sc - sg))),
                             cpu_s=tc, gpu_s=tg))
            print(rows[-1], flush=True)
    CM.atomic_write_text(OUT / f"knn_equality_{cell}.json", json.dumps(rows, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "env":
        env()
    elif mode == "bench":
        bench()
    elif mode == "knncheck":
        knncheck(sys.argv[2], int(sys.argv[3]))
