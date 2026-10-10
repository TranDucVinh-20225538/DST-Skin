#!/usr/bin/env python3
"""T1 item H (H1, H2, H3) runner: units of the H work list in execution order, unit i -> task i % NTASKS; cap 10 GPU-h
counted from the per-unit seconds of all H raw files (S9 is checked in the analysis; H4 runs at stage 4).

    item_H_run.py TASK NTASKS   -> results/t1/H/raw/h_<task>.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "H" / "raw")))
CAP_S = 10 * 3600.0


def spent():
    tot = 0.0
    for f in RAW.glob("h_*.jsonl"):
        for line in open(f):
            try:
                tot += float(json.loads(line).get("seconds", 0.0))
            except ValueError:
                pass
    return tot


def main(task, ntasks):
    CM.set_float64_torch()
    import item_H1 as H1
    import item_H2 as H2
    import item_H3 as H3
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_H.json"))
    order = add["work_lists"][0]["execution_order"]
    params = add["unit_params"]
    mine = [u for i, u in enumerate(order) if i % ntasks == task]
    out = RAW / f"h_{task}.jsonl"
    have = set()
    if out.exists():
        with open(out, "rb+") as fh:
            data = fh.read()
            if data and not data.endswith(b"\n"):
                fh.truncate(data.rfind(b"\n") + 1)
        have = {json.loads(line)["unit"] for line in open(out)}
    pre = CM.precommit_hash()
    with open(out, "a") as fh:
        for uid in [u for u in mine if u not in have][:int(os.environ.get("T1_MAXUNITS", "100000000"))]:
            if spent() >= CAP_S:
                print("H cap reached", flush=True)
                return
            t0 = time.time()
            kind = uid.split("|")[0]
            cell = params[uid].get("cell")
            if kind in ("hn1", "hn2", "hn3"):
                folds = {"hn1": H1.hn1, "hn2": H1.hn2, "hn3": H1.hn3}[kind](uid, cell)
                res = dict(control=kind, cell=cell, dataset=H1.REG[cell]["dataset"], backbone=H1.REG[cell]["backbone"],
                           folds=folds, **H1.summarise(folds))
            elif kind == "h2":
                res = H2.run_cell(uid, cell)
                res.pop("unit")
            else:
                res = H3.run_unit(uid, params[uid], deadline=t0 + max(0.0, 1.1 * CAP_S - spent()) / ntasks)
            res["seconds"] = time.time() - t0
            fh.write(json.dumps(dict(unit=uid, params=params[uid], precommit=pre, **res)) + "\n")
            fh.flush()
            print(uid, f"{res['seconds']:.0f}s", res.get("delta", res.get("coverage_dq_pred", res.get("placebo_p_delta"))), flush=True)
    if "T1_MAXUNITS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
