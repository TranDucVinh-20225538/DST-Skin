#!/usr/bin/env python3
"""DermaMNIST part of decisions/precommit_foundation_leakage_gate_2026-10-07.md (L3, CPU).

From the frozen FM embeddings (outputs/rigor_pack/foundation_gate/feats/dermamnist_{fm}.npz, keyed), per arm
std / b0 / b1: probe = fm_ood_pilot.fit_linear_head on the arm's training embeddings; writes the medbench
run npz outputs/rigor_pack/medbench/dermamnist/fm_{fm}_s42_{arm}.npz (feats / logits / labels / keys per set,
fc_weight / fc_bias = probe), so that medbench_scores.py runs unchanged with --arch fm_{fm}.
Probe accuracies -> outputs/reports/rigor_pack/foundation_gate/dermamnist/probe_{fm}_{arm}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import medbench_common as M  # noqa: E402

REPO = M.REPO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    args = ap.parse_args()
    from fm_ood_pilot import fit_linear_head
    z = np.load(REPO / "outputs/rigor_pack/foundation_gate/feats" / ("dermamnist_%s.npz" % args.fm))
    pos = {k: i for i, k in enumerate(z["all_keys"].tolist())}
    F = z["all_feats"]
    fdir = REPO / "outputs/rigor_pack/medbench/dermamnist"
    rdir = REPO / "outputs/reports/rigor_pack/foundation_gate/dermamnist"
    fdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    for armn in ("std", "b0", "b1"):
        sets = {k: v for k, v in M.arm("dermamnist", armn).items() if k != "val"}
        feats = {k: F[[pos[x] for x in v["keys"]]] for k, v in sets.items()}
        logits, W, b, ncls, tr_acc = fit_linear_head(feats["train"], sets["train"]["labels"])
        out = {"fc_weight": W, "fc_bias": b}
        rec = {"fm": args.fm, "arm": armn, "n_classes": ncls, "probe_train_acc": tr_acc,
               "n": {k: len(v["keys"]) for k, v in sets.items()}}
        for k, v in sets.items():
            lo = logits(feats[k])
            out.update({"%s_feats" % k: feats[k].astype(np.float32), "%s_logits" % k: lo.astype(np.float32),
                        "%s_labels" % k: v["labels"], "%s_keys" % k: np.array(v["keys"])})
            if not k.startswith("ood"):
                rec["acc_%s" % k] = float((lo.argmax(1) == v["labels"]).mean())
        np.savez(fdir / ("fm_%s_s42_%s.npz" % (args.fm, armn)), **out)
        (rdir / ("probe_%s_%s.json" % (args.fm, armn))).write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
