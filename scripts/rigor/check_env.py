#!/usr/bin/env python3
"""Report which Python packages the rigor pack needs are importable (Python 3.8+)."""

from __future__ import annotations

import importlib
import sys

NEEDS = {
    "numpy": "all stages", "pandas": "all stages", "scipy": "p-values (F, chi2, t); optional",
    "sklearn": "score cache, AUPR", "matplotlib": "make_tables figure", "torch": "score cache, extract, leakfree",
    "torchvision": "extract (models)", "wilds": "extract only (GPU stage)", "tqdm": "extract only",
}


def main() -> None:
    print("python", sys.version.split()[0], sys.executable)
    missing = []
    for mod, why in NEEDS.items():
        try:
            m = importlib.import_module(mod)
            print("  ok      %-12s %-12s (%s)" % (mod, getattr(m, "__version__", "?"), why))
        except Exception as exc:  # noqa: BLE001
            print("  MISSING %-12s (%s): %s" % (mod, why, exc.__class__.__name__))
            missing.append(mod)
    try:
        import torch

        print("  cuda available:", torch.cuda.is_available())
    except Exception:
        pass
    if missing:
        print("missing:", " ".join(missing))
        print("hint: use DST_PY=<torch-env python> or '$PY -m pip install --user <pkg>'")
    sys.exit(1 if set(missing) & {"numpy", "pandas"} else 0)


if __name__ == "__main__":
    main()
