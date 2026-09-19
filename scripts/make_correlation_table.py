#!/usr/bin/env python3
"""Covariate correlation table: skin + Camelyon FULL + MIDOG.

Do not source Camelyon from the 5% pilot mixed into
pilot_primary_delta_logitgap.csv — train_frac is a hidden confounder.
CIFAR-10/SVHN stays out (semantic/far-OOD).
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runpy.run_path(str(Path(__file__).with_name("make_fid_vs_gap_table.py")), run_name="__main__")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
