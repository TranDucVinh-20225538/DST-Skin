#!/usr/bin/env bash
# Submit CIFAR-10 vs SVHN pilot (1 seed, 3 backbones) on A100 via srun.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/cifar_svhn_pilot.log"

mkdir -p "${ROOT}/logs"

echo "=== SLURM queue snapshot $(date) ===" | tee "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %.6D %R" | tee -a "${LOG}"
squeue -p defq | head -10 | tee -a "${LOG}"

# SVHN download on CPU node (no GPU) — overlaps with CIFAR train
nohup srun --gres=gpu:0 -c 2 --mem=4G -t 02:00:00 \
  --job-name=dst-svhn-dl \
  bash -lc "
    set -e
    mkdir -p ${ROOT}/data/raw/svhn
    cd ${ROOT}/data/raw/svhn
    rm -f test_32x32.mat
    wget -c -O test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
    ls -lh test_32x32.mat
  " >> "${ROOT}/logs/svhn_download.log" 2>&1 &

nohup srun --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 \
  --job-name=dst-cifar-pilot \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/cifar_svhn_pilot.py --stage all --backbone all --seed 42 --batch-size 128 --num-workers 4
    echo '=== done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted background srun; log: ${LOG}"
sleep 3
squeue -u "$(whoami)" | tee -a "${LOG}"
