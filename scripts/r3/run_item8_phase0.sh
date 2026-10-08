#!/bin/bash
#SBATCH --job-name=r3-8-p2a-ph0
#SBATCH --exclude=node002
#SBATCH -c 24
#SBATCH --mem=110G
#SBATCH --output=logs/r3_8_phase0_%j.out
# R3 item 8 / P2-a Phase 0 (CPU): staging to 256 px, audit sanity, ChestMNIST audit, leak gate.
set -euo pipefail
R=$HOME/DST-Skin
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
TMPDIR=$(mktemp -d /tmp/r3_8_ph0_XXXXXX)
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
cd $R
$HOME/.venvs/crossfit-r3/bin/python scripts/r3/item8_p2a_phase0.py
