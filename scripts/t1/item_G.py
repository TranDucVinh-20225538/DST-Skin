#!/usr/bin/env python3
"""T1 item G: controlled group overlap on Camelyon17 (G0 manifest + slide design, G1 frozen backbones, analysis).

  python item_G.py g0            CPU: manifest_real.csv, slide_design.csv (+ sha256), abort checks, WARN records,
                                 g_dose_units.csv, MANIFEST_OK or MANIFEST_MISMATCH.md
  python item_G.py g1 <fm>       GPU: per FM, doses k in {0,1,2,4} x heads {logreg, adapter} x scorers
                                 {mahalanobis_l2, knn_mean_cosine, vim_head}; Delta_G, Delta_G^sw, paired bootstrap,
                                 K1 identity, head-identity check, P_Delta (A1 pipeline, mahalanobis_l2 only)
                                 -> results/t1/G/raw/g1_<fm>.json
  python item_G.py analyze       g1_frozen.csv, g_predictions.csv, g_sensitivity_slide_equal.csv, tests.csv,
                                 dose.png, verdict.json
G2 is not run (A != GO), so G-ii = not run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "G"
RAW = OUT / "raw"
CACHE = Path.home() / "r3work" / "item1" / "inputs"
FMS = ["uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5"]
HOSP = (0, 3, 4)
SETS = ["U", "T", "E1", "E2", "E3", "E4"]
EXPECT = {"U": 2, "T": 4, "E1": 1, "E2": 1, "E3": 1, "E4": 1}
DOSES = (0, 1, 2, 4)
GFIT = {0: 12, 1: 15, 2: 18, 4: 24}
HEADS = ("logreg", "adapter")
SCORERS = ("mahalanobis_l2", "knn_mean_cosine", "vim_head")
NB = 2000
KNN_K = 50
DOC = dict(train=302436, id_val=33560, ood=85054)


def seen_blocks(k):
    return ["E1", "E2"] if k == 0 else ["E%d" % j for j in range(1, k + 1)]


def fit_blocks(k):
    return ["T"] + ["E%d" % j for j in range(1, k + 1)]


def meta():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rigor_common", CM.REPO / "scripts" / "rigor" / "common.py")
    RC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RC)
    return RC.load_camelyon_metadata(CM.REPO)


def write_csv(path, rows):
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def assign(slides_by_h):
    rng = np.random.default_rng(np.random.SeedSequence([CM.MASTER_SEED, 7, 0]))
    out = {}
    for h in HOSP:
        p = rng.permutation(sorted(slides_by_h[h], key=str))
        sets = {"U": p[0:2], "T": p[2:6]}
        for j in range(1, 5):
            sets["E%d" % j] = [p[5 + j]]
        for s, members in sets.items():
            for x in members:
                out.setdefault(str(x), []).append(s)
    return out


def heldout_positions(slide, n):
    rng = np.random.default_rng(np.random.SeedSequence([CM.MASTER_SEED, 7, 1, CM.stable_hash(slide)]))
    return np.sort(rng.choice(n, round(0.3 * n), replace=False))


def load_design():
    rows = list(csv.DictReader(open(OUT / "slide_design.csv")))
    return {r["slide"]: r for r in rows}


def g0():
    OUT.mkdir(parents=True, exist_ok=True)
    m = meta()
    sp = m["wilds_split"].to_numpy()
    slide = m["slide"].astype(str).to_numpy()
    center = m["center"].to_numpy()
    tr, iv, te = sp == 0, sp == 1, sp == 2
    train_slides = sorted(set(slide[tr].tolist()), key=str)
    hosp = {s: sorted(set(center[tr & (slide == s)].tolist())) for s in train_slides}
    ntr = {s: int((tr & (slide == s)).sum()) for s in train_slides}
    niv = {s: int((iv & (slide == s)).sum()) for s in train_slides}
    write_csv(OUT / "manifest_real.csv", [dict(slide=s, hospital=hosp[s][0] if len(hosp[s]) == 1 else "|".join(map(str, hosp[s])),
                                               n_train_patches=ntr[s], n_idval_patches=niv[s]) for s in train_slides])
    aborts, warns = [], []
    i_ok = (CM.RES / "I" / "MANIFEST_OK").exists() and (CM.RES / "I" / "MANIFEST_OK").read_text().strip() == "true"
    if not i_ok:
        aborts.append("results/t1/I/MANIFEST_OK is not 'true'")
    if any(len(v) != 1 for v in hosp.values()):
        aborts.append("a training slide spans more than one hospital")
    hs = {s: hosp[s][0] for s in train_slides}
    if sorted(set(hs.values())) != list(HOSP):
        aborts.append(f"hospitals of training slides = {sorted(set(hs.values()))} != [0, 3, 4]")
    per_h = {h: [s for s in train_slides if hs[s] == h] for h in HOSP}
    if [len(per_h[h]) for h in HOSP] != [10, 10, 10]:
        aborts.append(f"slides per hospital = {[len(per_h[h]) for h in HOSP]} != (10, 10, 10)")
    if len(train_slides) != 30:
        aborts.append(f"total training slides = {len(train_slides)} != 30")
    if any(hs[s] in (1, 2) for s in train_slides):
        aborts.append("hospital 1 or 2 among the 30 slides")
    a = assign(per_h)
    multi = [s for s, v in a.items() if len(v) != 1]
    none = [s for s in train_slides if s not in a]
    if multi or none:
        aborts.append(f"slides in more than one set {multi} or in none {none}")
    count = {(h, st): sum(1 for s in per_h[h] if a.get(s, [None])[0] == st) for h in HOSP for st in SETS}
    bad = [(h, st, count[(h, st)], EXPECT[st]) for h in HOSP for st in SETS if count[(h, st)] != EXPECT[st]]
    if bad:
        aborts.append(f"per hospital x set counts differ from the table: {bad}")
    cache_sets = {}
    for fm in FMS:
        z = np.load(CACHE / f"camelyon_{fm}_s42.npz", allow_pickle=True)
        g = z["groups_train"].astype(str)
        cache_sets[fm] = sorted(set(g.tolist()), key=str)
        if cache_sets[fm] != train_slides:
            aborts.append(f"{fm} cache slide set differs from the metadata")
        if not np.array_equal(g, slide[tr]):
            aborts.append(f"{fm} groups_train order differs from the metadata train order (labels/held-out alignment)")
        if len(z["features_ood"]) != int(te.sum()):
            warns.append(f"{fm}: features_ood rows {len(z['features_ood'])} != metadata test rows {int(te.sum())}")
    design = []
    for s in train_slides:
        st = a[s][0]
        n = ntr[s]
        nh = round(0.3 * n) if st != "T" else 0
        design.append(dict(slide=s, hospital=hs[s], set=st, n_train_patches=n, n_heldout=nh, n_fit_eligible=n - nh))
    write_csv(OUT / "slide_design.csv", design)
    sha = hashlib.sha256((OUT / "slide_design.csv").read_bytes()).hexdigest()
    (OUT / "slide_design.sha256").write_text(sha + "  slide_design.csv\n")
    # fit/fine-tune/selection sets contain only slides of T and E (hospitals 0, 3, 4)
    fit_h = sorted({int(r["hospital"]) for r in design if r["set"] != "U"})
    if any(h in (1, 2) for h in fit_h):
        aborts.append("hospital 1 or 2 among fit-set patches")
    # WARN (report-only)
    tot = dict(train=int(tr.sum()), id_val=int(iv.sum()), ood=int(te.sum()))
    for k_, v in DOC.items():
        if tot[k_] != v:
            warns.append(f"total {k_} patches {tot[k_]} != documented {v}")
    by = {r["slide"]: r for r in design}
    blk = {st: dict(slides=[r["slide"] for r in design if r["set"] == st],
                    n_train=sum(r["n_train_patches"] for r in design if r["set"] == st),
                    n_heldout=sum(r["n_heldout"] for r in design if r["set"] == st),
                    n_fit_eligible=sum(r["n_fit_eligible"] for r in design if r["set"] == st)) for st in SETS}
    held_E = sum(blk["E%d" % j]["n_heldout"] for j in range(1, 5))
    shares = []
    for k in DOSES:
        for role, bl in (("seen", seen_blocks(k)), ("unseen", ["U"])):
            ss = [s for b in bl for s in blk[b]["slides"]]
            den = sum(by[s]["n_heldout"] for s in ss)
            for s in ss:
                shares.append(dict(k=k, role=role, slide=s, set=by[s]["set"], n_heldout=by[s]["n_heldout"],
                                   share=by[s]["n_heldout"] / den))
    write_csv(OUT / "g_slide_shares.csv", shares)
    units = []
    nfit4 = blk["T"]["n_train"] + sum(blk["E%d" % j]["n_fit_eligible"] for j in range(1, 5))
    for k in DOSES:
        nfit = blk["T"]["n_train"] + sum(blk["E%d" % j]["n_fit_eligible"] for j in range(1, k + 1))
        nseen = sum(blk[b]["n_heldout"] for b in seen_blocks(k))
        units.append(dict(k=k, slide_dose=k / 4, G_fit=GFIT[k], n_seen_slides=3 * len(seen_blocks(k)),
                          N_fit=nfit, N_fit_over_N_fit4=nfit / nfit4, n_seen_heldout=nseen,
                          n_unseen_heldout=blk["U"]["n_heldout"],
                          omega_patch=(sum(blk["E%d" % j]["n_heldout"] for j in range(1, k + 1)) / held_E),
                          note="placebo: seen = E1+E2 (not in fit)" if k == 0 else ""))
    write_csv(OUT / "g_dose_units.csv", units)
    rec = dict(aborts=aborts, warns=warns, totals=tot, documented=DOC, sha256=sha,
               per_slide_train_min=min(ntr.values()), per_slide_train_max=max(ntr.values()),
               blocks=blk, counts={f"{h}|{st}": count[(h, st)] for h in HOSP for st in SETS}, manifest_I_ok=i_ok)
    json.dump(rec, open(OUT / "g0.json", "w"), indent=1)
    if aborts:
        L = ["# G0 MANIFEST MISMATCH (S11)", "", "G verdict: `not run: manifest mismatch (S11)`; G1, G2 not run.", ""]
        L += [f"- {x}" for x in aborts]
        L += ["", "| hospital | set | real | expected |", "|---|---|---|---|"]
        L += [f"| {h} | {st} | {count[(h, st)]} | {EXPECT[st]} |" for h in HOSP for st in SETS]
        (OUT / "MANIFEST_MISMATCH.md").write_text("\n".join(L) + "\n")
        print("G0 ABORT", aborts)
        return 1
    (OUT / "MANIFEST_OK").write_text("true\n")
    print("G0 ok", sha, "warns:", warns)
    return 0


# ---------------------------------------------------------------- G1
def boot_draws(design):
    """Fixed replicate draws (default_rng(2)): per replicate, slide multiplicities resampled with replacement within
    U, within E1, within E1+E2 (the k=0 placebo and k=2 seen set share it), within E1..E4, then OOD patch counts."""
    pools = {"U": ["U"], "S1": ["E1"], "S2": ["E1", "E2"], "S4": ["E1", "E2", "E3", "E4"]}
    sl = {p: [s for s, r in design.items() if r["set"] in b] for p, b in pools.items()}
    for p in sl:
        sl[p] = sorted(sl[p], key=str)
    rng = np.random.default_rng(2)
    nood = DOC["ood"]
    mult = {p: np.zeros((NB, len(sl[p]))) for p in sl}
    wo = np.zeros((NB, nood), dtype=np.uint8)
    for b in range(NB):
        for p in ("U", "S1", "S2", "S4"):
            mult[p][b] = np.bincount(rng.integers(0, len(sl[p]), len(sl[p])), minlength=len(sl[p]))
        wo[b] = np.bincount(rng.integers(0, nood, nood), minlength=nood)
    return sl, mult, wo


POOL_OF = {0: "S2", 1: "S1", 2: "S2", 4: "S4"}


class Pair:
    """Pooled (patch) and slide-equal AUROCs of an eval set vs OOD, with the paired bootstrap."""

    def __init__(self, s, slide_idx, n_slide, s_ood):
        order = np.argsort(s_ood, kind="mergesort")
        so = s_ood[order]
        self.order = order
        self.lo = np.searchsorted(so, s, side="left")
        self.hi = np.searchsorted(so, s, side="right")
        self.si = slide_idx
        self.inv_n = 1.0 / n_slide[slide_idx]
        self.nood = len(s_ood)

    def point(self):
        v = (self.lo + 0.5 * (self.hi - self.lo)) / self.nood
        per = np.bincount(self.si, weights=v * self.inv_n)
        return float(v.mean()), float(per.sum() / len(per))

    def boot(self, mult, cum):
        tot = cum[-1]
        below = np.where(self.lo > 0, cum[np.maximum(self.lo - 1, 0)], 0.0)
        tie = np.where(self.hi > 0, cum[np.maximum(self.hi - 1, 0)], 0.0) - below
        v = (below + 0.5 * tie) / tot
        w = mult[self.si]
        ws = w * self.inv_n
        return float((w * v).sum() / w.sum()), float((ws * v).sum() / ws.sum())


def _medbench():
    """Import scripts/rigor/medbench_scores (it imports the rigor `common`, which shadows the T1 `common`)."""
    if "medbench_scores" in sys.modules:
        return sys.modules["medbench_scores"]
    saved = sys.modules.pop("common")
    sys.path.insert(0, str(CM.REPO / "scripts" / "rigor"))
    try:
        import medbench_scores
    finally:
        sys.path.pop(0)
        sys.modules["common"] = saved
    return medbench_scores


def k1_dev(s_seen, s_unseen, s_ood):
    """K1: the pooled AUROC of an omega-mixture (weights omega/n_s, (1-omega)/n_u) equals the mixture of AUROCs."""
    wauroc = _medbench().wauroc
    order = np.argsort(s_ood, kind="mergesort")
    so = s_ood[order]
    cum = np.arange(1, len(so) + 1, dtype=np.float64)
    a_s = wauroc(s_seen, np.ones(len(s_seen)), so, cum, cum[-1])
    a_u = wauroc(s_unseen, np.ones(len(s_unseen)), so, cum, cum[-1])
    dev = 0.0
    for om in (0.25, 0.5, 0.75):
        si = np.concatenate([s_seen, s_unseen])
        wi = np.concatenate([np.full(len(s_seen), om / len(s_seen)), np.full(len(s_unseen), (1 - om) / len(s_unseen))])
        dev = max(dev, abs(wauroc(si, wi, so, cum, cum[-1]) - (om * a_s + (1 - om) * a_u)))
    return dev


def knn_scores(Xfit_unit, Q_unit):
    import torch
    out = []
    for i in range(0, Q_unit.shape[0], 2048):
        s = Q_unit[i:i + 2048] @ Xfit_unit.T
        v, _ = torch.topk(s, KNN_K, dim=1, largest=True)
        out.append(-(1.0 - v).mean(1))
    return torch.cat(out).cpu().numpy()


def unit_t(x):
    import scorer64 as S64
    import torch
    t = S64.to_t(S64.l2n(x))
    return t / torch.linalg.norm(t, dim=1, keepdim=True)


def maha_block(Xf, gf, Xs, Xu, Xo, seen_groups, cell_tag, with_pred):
    """mahalanobis_l2 scores (Track A score -sqrt(Q)) and, if with_pred, the A1 step-6 primary additive-map P_Delta."""
    import scorer64 as S64
    from item_A1_cell import logo_s
    F = S64.to_t(S64.l2n(Xf))
    lw = S64.LW(F)
    qs, qu, qo = (lw.q(S64.to_t(S64.l2n(x))).cpu().numpy() for x in (Xs, Xu, Xo))
    sc = dict(seen=-np.sqrt(qs), unseen=-np.sqrt(qu), ood=-np.sqrt(qo))
    pred = None
    if with_pred:
        N = F.shape[0]
        delta = lw.shrinkage
        st, u, cnt = S64.group_stats(F, gf, lw.P)
        s_logo, scheme = logo_s(F, gf, lw, cell_tag, 0)
        n_of = dict(zip(u, cnt))
        ug, cg = np.unique(seen_groups, return_counts=True)
        n_g = np.array([n_of[x] for x in ug], dtype=float)
        pi_g = cg / cg.sum()
        rho = st["rho_w"]
        dq = S64.dq_pred_groups(n_g, pi_g, rho, s_logo, N, delta)
        a_unseen = S64.auroc_id_pos(-qu, -qo)
        p_delta = S64.auroc_id_pos(-(qu - abs(dq)), -qo) - a_unseen
        pred = dict(p_delta=p_delta, dq_pred=dq, dq_meas=float(qs.mean() - qu.mean()), rho_w=rho,
                    rho_raw=st["rho_raw"], rho_anova=st["rho_anova"], s_logo=s_logo, s_logo_scheme=scheme,
                    s_tr=lw.s_tr(), delta_lw=delta, N=N, G=st["G"], n_g_seen_mean=float(n_g.mean()),
                    pi_g=pi_g.tolist(), n_g=n_g.tolist(), a_unseen_q=a_unseen)
    del F, lw
    return sc, pred


def train_heads(Xf, yf, seed=42):
    """(a) logistic probe (L2, C=1, standardised) folded into raw-feature (W, b) with 2-class logits (0, z);
    (b) adapter D->512 ReLU->2, AdamW lr 1e-3, 20 epochs, batch 512, seed 42, on standardised inputs."""
    import torch
    from sklearn.linear_model import LogisticRegression
    mu = Xf.mean(0).astype(np.float64)
    sd = Xf.std(0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((Xf - mu) / sd).astype(np.float64)
    lr = LogisticRegression(C=1.0, penalty="l2", max_iter=1000).fit(Z, yf)
    coef = lr.coef_[0] / sd
    b1 = float(lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum())
    head_a = dict(W=np.vstack([np.zeros_like(coef), coef]), b=np.array([0.0, b1]), n_iter=int(lr.n_iter_[0]))
    del Z
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() and __import__("os").environ.get("T1_DEV", "cuda") == "cuda" else "cpu"
    D = Xf.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(D, 512), torch.nn.ReLU(), torch.nn.Linear(512, 2)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=dev)
    sd_t = torch.as_tensor(sd, dtype=torch.float32, device=dev)
    Xt = torch.as_tensor(Xf, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(yf, dtype=torch.long, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(20):
        perm = torch.randperm(len(yt), generator=gen).to(dev)
        for i in range(0, len(yt), 512):
            j = perm[i:i + 512]
            opt.zero_grad()
            lossf(net((Xt[j] - mu_t) / sd_t), yt[j]).backward()
            opt.step()
    net.eval()

    def hidden(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                x = torch.as_tensor(X[i:i + 65536], dtype=torch.float32, device=dev)
                out.append(net[1](net[0]((x - mu_t) / sd_t)).double().cpu().numpy())
        return np.concatenate(out)

    head_b = dict(W=net[2].weight.detach().double().cpu().numpy(), b=net[2].bias.detach().double().cpu().numpy(),
                  hidden=hidden)
    del Xt
    return head_a, head_b


def vim_scores(head, Xf, Xs, Xu, Xo, hidden=False):
    from crossfit_ood.scorers import ViMScorer
    f = head["hidden"] if hidden else (lambda X: np.asarray(X, dtype=np.float64))
    v = ViMScorer(weight=head["W"], bias=head["b"]).fit(f(Xf))
    return dict(seen=v.score(f(Xs)), unseen=v.score(f(Xu)), ood=v.score(f(Xo)))


def head_acc(head, X, y, hidden=False):
    f = head["hidden"] if hidden else (lambda Z: np.asarray(Z, dtype=np.float64))
    return float(((f(X) @ head["W"].T + head["b"]).argmax(1) == y).mean())


def g1(fm):
    import scorer64 as S64  # noqa: F401
    CM.set_float64_torch()
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    design = load_design()
    m = meta()
    sp = m["wilds_split"].to_numpy()
    y_tr = m["tumor"].to_numpy()[sp == 0].astype(int)
    z = np.load(CACHE / f"camelyon_{fm}_s42.npz", allow_pickle=True)
    X = z["features_train"]
    g = z["groups_train"].astype(str)
    Xo = z["features_ood"]
    raw_mode = __import__("os").environ.get("T1_RAW")
    held = np.zeros(len(g), bool)
    for s, r in design.items():
        if r["set"] == "T":
            continue
        pos = np.flatnonzero(g == s)
        assert len(pos) == int(r["n_train_patches"])
        held[pos[heldout_positions(s, len(pos))]] = True
    set_of = np.array([design[s]["set"] for s in g])
    sl, mult, wo = boot_draws(design)
    if raw_mode:  # smoke test only
        sub = np.random.default_rng(0).random(len(g)) < float(raw_mode)
        oo = np.random.default_rng(1).random(len(Xo)) < float(raw_mode)
        Xo = Xo[oo]
        wo = wo[:, oo]
    stopped = {}
    rows, preds, checks, accs = [], [], [], []
    for k in DOSES:
        fit_m = np.isin(set_of, fit_blocks(k)) & ~held
        seen_m = np.isin(set_of, seen_blocks(k)) & held
        uns_m = (set_of == "U") & held
        if raw_mode:
            fit_m &= sub
            seen_m &= sub
            uns_m &= sub
        Xf, gf, yf = X[fit_m], g[fit_m], y_tr[fit_m]
        Xs, Xu = X[seen_m], X[uns_m]
        assert len(set(gf)) == GFIT[k] and not np.isin(gf, sl["U"]).any()
        scores = {}
        tk = time.time()
        for hi, head in enumerate(HEADS):
            sc, pred = maha_block(Xf, gf, Xs, Xu, Xo, g[seen_m], f"G|{fm}|k{k}", with_pred=(hi == 0 and k > 0))
            scores[(head, "mahalanobis_l2")] = sc
            if pred is not None:
                preds.append(dict(fm=fm, k=k, **pred))
            F = unit_t(Xf)
            scores[(head, "knn_mean_cosine")] = dict(seen=knn_scores(F, unit_t(Xs)), unseen=knn_scores(F, unit_t(Xu)),
                                                     ood=knn_scores(F, unit_t(Xo)))
            del F
        ha, hb = train_heads(Xf, yf)
        scores[("logreg", "vim_head")] = vim_scores(ha, Xf, Xs, Xu, Xo)
        scores[("adapter", "vim_head")] = vim_scores(hb, Xf, Xs, Xu, Xo, hidden=True)
        yu = y_tr[uns_m]
        accs.append(dict(fm=fm, k=k, acc_U_logreg=head_acc(ha, Xu, yu), acc_U_adapter=head_acc(hb, Xu, yu, True),
                         logreg_n_iter=ha["n_iter"]))
        for sname in ("mahalanobis_l2", "knn_mean_cosine"):
            a, b = scores[("logreg", sname)], scores[("adapter", sname)]
            checks.append(dict(fm=fm, k=k, scorer=sname, check="identical_across_heads",
                               value=float(max(np.abs(a[r] - b[r]).max() for r in ("seen", "unseen", "ood"))),
                               ok=all(np.array_equal(a[r], b[r]) for r in ("seen", "unseen", "ood"))))
        pool = POOL_OF[k]
        ix_s = {s: i for i, s in enumerate(sl[pool])}
        ix_u = {s: i for i, s in enumerate(sl["U"])}
        sidx_s = np.array([ix_s[x] for x in g[seen_m]])
        sidx_u = np.array([ix_u[x] for x in g[uns_m]])
        n_s = np.bincount(sidx_s, minlength=len(sl[pool])).astype(float)
        n_u = np.bincount(sidx_u, minlength=len(sl["U"])).astype(float)
        for (head, sname), sc in scores.items():
            key = (head, sname)
            if stopped.get(key):
                rows.append(dict(fm=fm, head=head, scorer=sname, k=k, status="not run: S5 (Delta_G(0) CI excludes 0)"))
                continue
            dk = k1_dev(sc["seen"], sc["unseen"], sc["ood"])
            checks.append(dict(fm=fm, k=k, scorer=sname, head=head, check="K1_identity", value=dk, ok=dk <= 1e-12))
            ps, pu = Pair(sc["seen"], sidx_s, n_s, sc["ood"]), Pair(sc["unseen"], sidx_u, n_u, sc["ood"])
            (as_, as_sw), (au, au_sw) = ps.point(), pu.point()
            bd, bsw = np.zeros(NB), np.zeros(NB)
            order = ps.order
            for bi in range(NB):
                cum = np.cumsum(wo[bi][order].astype(np.float64))
                s1, s1w = ps.boot(mult[pool][bi], cum)
                u1, u1w = pu.boot(mult["U"][bi], cum)
                bd[bi], bsw[bi] = s1 - u1, s1w - u1w
            d, dsw = as_ - au, as_sw - au_sw
            lo, hi = np.percentile(bd, [2.5, 97.5])
            lo_sw, hi_sw = np.percentile(bsw, [2.5, 97.5])
            rows.append(dict(fm=fm, head=head, scorer=sname, k=k, status="ok", auroc_seen=as_, auroc_unseen=au,
                             delta=d, ci_lo=float(lo), ci_hi=float(hi), auroc_seen_sw=as_sw, auroc_unseen_sw=au_sw,
                             delta_sw=dsw, ci_lo_sw=float(lo_sw), ci_hi_sw=float(hi_sw),
                             n_seen=int(seen_m.sum()), n_unseen=int(uns_m.sum()), n_ood=len(sc["ood"]),
                             N_fit=int(fit_m.sum()), G_fit=len(set(gf))))
            if k == 0 and not (lo <= 0 <= hi):
                stopped[key] = True
        print(f"{fm} k={k} done {time.time() - tk:.0f}s", flush=True)
    json.dump(dict(fm=fm, rows=rows, preds=preds, checks=checks, accs=accs, wall_s=time.time() - t0,
                   smoke=bool(raw_mode)), open(RAW / f"g1_{fm}.json", "w"), indent=1, default=float)
    return 0


# ---------------------------------------------------------------- analysis
def lin_r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan")
    b = np.polyfit(x, y, 1)
    r = y - np.polyval(b, x)
    return float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum())


def analyze():
    units = {int(r["k"]): r for r in csv.DictReader(open(OUT / "g_dose_units.csv"))}
    rows, preds, checks, accs, missing = [], [], [], [], []
    for fm in FMS:
        p = RAW / f"g1_{fm}.json"
        if not p.exists():
            missing.append(fm)
            continue
        d = json.load(open(p))
        rows += d["rows"]
        preds += d["preds"]
        checks += d["checks"]
        accs += d["accs"]
    for r in rows:
        u = units[int(r["k"])]
        r["slide_dose"] = float(u["slide_dose"])
        r["omega_patch"] = float(u["omega_patch"])
    write_csv(OUT / "g1_frozen.csv", [{k: v for k, v in r.items() if not k.endswith("_sw")} for r in rows])
    sens = []
    for r in rows:
        if r["status"] != "ok":
            continue
        d, dsw = r["delta"], r["delta_sw"]
        sens.append(dict(fm=r["fm"], head=r["head"], scorer=r["scorer"], k=r["k"], channel="frozen (scorer only)",
                         delta=d, delta_sw=dsw, ci_lo_sw=r["ci_lo_sw"], ci_hi_sw=r["ci_hi_sw"],
                         sign_flag=CM.sign0(dsw) != CM.sign0(d), magnitude_flag=abs(dsw - d) > 0.5 * abs(d),
                         label="placebo, expected noise-dominated" if r["k"] == 0 else ""))
    if sens:
        write_csv(OUT / "g_sensitivity_slide_equal.csv", sens)
    # g_predictions.csv
    meas = {(r["fm"], r["k"]): r for r in rows if r["scorer"] == "mahalanobis_l2" and r["head"] == "logreg"}
    pr = {(p["fm"], p["k"]): p for p in preds}
    gp = []
    for fm in FMS:
        for k in DOSES:
            for sname in SCORERS:
                for head in (HEADS if sname == "vim_head" else ("(head-independent)",)):
                    hh = "logreg" if head == "(head-independent)" else head
                    r = next((x for x in rows if x["fm"] == fm and x["k"] == k and x["scorer"] == sname
                              and x["head"] == hh), None)
                    dm = r.get("delta") if r and r["status"] == "ok" else None
                    row = dict(fm=fm, k=k, scorer=sname, head=head, delta_meas=dm, p_delta="NA", e_rel="NA",
                               in_gi_median=False, status=(r["status"] if r else "not run"))
                    if sname == "mahalanobis_l2" and k > 0:
                        p = pr.get((fm, k))
                        if p:
                            row.update(p_delta=p["p_delta"], dq_pred=p["dq_pred"], dq_meas=p["dq_meas"],
                                       rho_w=p["rho_w"], s_logo=p["s_logo"], delta_lw=p["delta_lw"], N=p["N"],
                                       G=p["G"], n_g_seen_mean=p["n_g_seen_mean"])
                            if dm is not None and dm >= 0.02:
                                row["in_gi_median"] = True
                                row["e_rel"] = abs(p["p_delta"] / dm - 1)
                    gp.append(row)
    write_csv(OUT / "g_predictions.csv", gp)
    # G-i
    c1_cells = [c for c in checks]
    c1_ok = bool(c1_cells) and all(c["ok"] for c in c1_cells) and not missing
    c1 = "pass" if c1_ok else ("fail" if c1_cells and not all(c["ok"] for c in c1_cells) else "not evaluable")
    comb = [(fm, s) for fm in FMS for s in ("mahalanobis_l2", "knn_mean_cosine")]
    c2_vals = []
    for fm, s in comb:
        r = next((x for x in rows if x["fm"] == fm and x["k"] == 0 and x["scorer"] == s and x["head"] == "logreg"), None)
        c2_vals.append(None if r is None or r["status"] != "ok" else (r["ci_lo"] <= 0 <= r["ci_hi"]))
    c2 = "fail" if any(v is False for v in c2_vals) else ("not evaluable" if any(v is None for v in c2_vals) else "pass")
    cand = [(fm, k) for fm in FMS for k in (1, 2, 4)]
    known = [meas.get(c) for c in cand]
    complete = all(r is not None and r["status"] == "ok" for r in known) and all(c in pr for c in cand
                                                                                if meas.get(c) and meas[c]["delta"] >= 0.02)
    ent = [x for x in gp if x["in_gi_median"]]
    med = float(np.median([x["e_rel"] for x in ent])) if ent and complete else None
    if not complete or not ent:
        c3 = "not evaluable"
    else:
        c3 = "pass" if med <= 0.5 else "fail"
    st = [c1, c2, c3]
    gi = "fail" if "fail" in st else ("not evaluable" if "not evaluable" in st else "pass")
    gii = "not run"
    verdict = {"fail": "NO-GO", "not evaluable": "INCONCLUSIVE", "pass": "PARTIAL"}[gi]
    tests = [dict(test="G-0", status="pass", detail="MANIFEST_OK"),
             dict(test="G-i c1 implementation checks", status=c1, detail=f"{sum(c['ok'] for c in checks)}/{len(checks)} ok; missing FMs {missing}"),
             dict(test="G-i c2 Delta_G(0) CI contains 0 (10 combos)", status=c2,
                  detail=f"{sum(v is True for v in c2_vals)}/10 contain 0"),
             dict(test="G-i c3 median e_rel <= 0.5", status=c3,
                  detail=f"median={med}; entering cells={len(ent)}; cell set complete={complete}"),
             dict(test="G-i", status=gi, detail=""),
             dict(test="G-ii", status=gii, detail="G2 not run (A = NO-GO)"),
             dict(test="verdict", status=verdict, detail="")]
    for t in tests:
        t["pass"] = {"pass": True, "fail": False}.get(t["status"], "na")
    write_csv(OUT / "tests.csv", tests)
    write_csv(OUT / "g_checks.csv", checks)
    if accs:
        write_csv(OUT / "g_head_acc.csv", accs)
    # dose.png + linear-fit R2 on both axes
    lin = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for fm in FMS:
            rs = sorted([r for r in rows if r["fm"] == fm and r["scorer"] == "mahalanobis_l2" and r["head"] == "logreg"
                         and r["status"] == "ok"], key=lambda r: r["k"])
            if not rs:
                continue
            for j, axis in enumerate(("slide_dose", "omega_patch")):
                x = [r[axis] for r in rs]
                ax[j].errorbar(x, [r["delta"] for r in rs], yerr=[[r["delta"] - r["ci_lo"] for r in rs],
                               [r["ci_hi"] - r["delta"] for r in rs]], marker="o", capsize=2, label=f"{fm} meas")
                pp = [(r[axis], pr[(fm, r["k"])]["p_delta"]) for r in rs if (fm, r["k"]) in pr]
                if pp:
                    ax[j].plot(*zip(*pp), ls="--", marker="x", label=f"{fm} P_Delta")
        ax[0].set_xlabel("slide dose k/4")
        ax[1].set_xlabel("patch-weighted omega(k)")
        ax[0].set_ylabel("Delta_G (mahalanobis_l2)")
        for a in ax:
            a.axhline(0, c="k", lw=0.5)
        ax[1].legend(fontsize=6)
        fig.suptitle("G1 frozen: Delta_G vs dose (CI: eval slides + OOD patches; not patient-level)")
        fig.tight_layout()
        fig.savefig(OUT / "dose.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot failed", e)
    for fm in FMS:
        for sname in SCORERS:
            for head in HEADS:
                rs = [r for r in rows if r["fm"] == fm and r["scorer"] == sname and r["head"] == head and r["status"] == "ok"]
                if len(rs) >= 3:
                    lin.append(dict(fm=fm, scorer=sname, head=head,
                                    r2_slide=lin_r2([r["slide_dose"] for r in rs], [r["delta"] for r in rs]),
                                    r2_patch=lin_r2([r["omega_patch"] for r in rs], [r["delta"] for r in rs])))
    if lin:
        write_csv(OUT / "g_linfit.csv", lin)
    json.dump(dict(verdict=verdict, G_i=gi, G_ii=gii, c1=c1, c2=c2, c3=c3, median_e_rel=med,
                   entering=[(x["fm"], x["k"]) for x in ent], missing=missing,
                   n_sign_flags=sum(s["sign_flag"] for s in sens), n_mag_flags=sum(s["magnitude_flag"] for s in sens)),
              open(OUT / "verdict.json", "w"), indent=1)
    print(verdict, gi, c1, c2, c3, med)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "g0":
        sys.exit(g0())
    if cmd == "g1":
        sys.exit(g1(sys.argv[2]))
    if cmd == "analyze":
        sys.exit(analyze())
