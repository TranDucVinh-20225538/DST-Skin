# T1 item 0 — Setup

- Repo: DST-Skin, branch `t1` from `rigor-pack` HEAD `4857c485fef134ba443131375da4b329d98297e7`.
- Precommit commit: `e326d392fd42ced26fbebe6c0a38ed983267d187` (2026-10-10T18:36:46Z); DEPOSIT `decisions/precommit_t1_2026-10-11.DEPOSIT.txt`.
- Precommit file hashes: `precommit_t1_2026-10-11.md` fa7f1076…0e15, `PRECOMMIT_T1.json` d7732251…0791.

## Hashes

| file | sha256 | expected | ok |
|---|---|---|---|
| WORK_ORDER_T1_v2_5.md | 4dec41ec184ce07c811c9ff9c6434bbcd382228b49a192bde7bb00c5cd5c3096 | same | yes |
| t1_theory_bundle.tgz | 035454a4c13a16e17179aa9a656e89252b2f2de866f5a97a7ab0e2425b5d1f06 | same | yes |
| crossfit-ood_v3.zip | c925933300e29dc76ff599cfc16696901aaa483eabc20b8fce3ad14ae2f9dce1 | same | yes |
| bundle members (27, unpacked to `~/handoff/t1/bundle/`) | all match `PRECOMMIT_T1_hashes.txt` | | yes |

S1 not triggered.

## Environment

python 3.11.5, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.9.1, torch 2.1.1, CUDA 12.1; GPU `NVIDIA A100-SXM4-80GB` (node004);
`sinfo`: defq — node004 mix (8 GPUs), node001 drain, node002 down. crossfit-ood 0.0.3.dev0 (`~/.venvs/crossfit-r3`, reused read-only). Details: `env.json`.

## Tests

| package copy | pytest -q |
|---|---|
| v3 zip, pristine | **44 passed / 44** |
| `~/handoff/crossfit-ood_v3` (R3 copy with Track A scorers, used by T1) | 47 passed, 3 skipped, 0 failed |

S2 (test part) not triggered. See DEVIATIONS D0-3.

## Microbenchmarks (float64, A100) — `throughput.json`, `BUDGET.md`

| quantity | value |
|---|---|
| matmul 4096 × 4096 | 17.8 TFLOP/s |
| `torch.linalg.inv` d = 2560 | 0.0152 s |
| `torch.linalg.cholesky` d = 2560 | 0.0019 s |
| kNN 50,000 × 100,000 × 2,560 (top-200) | 1.87 s |

## GPU-kNN equality check (rule 4) — `knn_equality_*.json`

Paper_2fold Δ through the package's own `_fit_scorer`, R3 GPU scorers (`scripts/r3/recompute_gpu_knn.py`) vs sklearn float64 CPU.

| cell | scorer | k | Δ (CPU) | max abs(ΔΔ) | max abs(ΔAUROC) | max abs(Δscore), 10k points |
|---|---|---|---|---|---|---|
| isic2019_resnet50_s42_std | knn_mean_cosine | 1 | 0.121694 | 0 | 0 | 4.6e-15 |
| isic2019_resnet50_s42_std | knn | 1 | 0.121694 | 0 | 0 | 1.5e-08 |
| isic2019_resnet50_s42_std | knn_mean_cosine | 50 | 0.015405 | 0 | 0 | 1.1e-15 |
| isic2019_resnet50_s42_std | knn | 50 | 0.004785 | 0 | 0 | 3.8e-15 |
| camelyon_resnet50_s42 | knn_mean_cosine | 1 | 0.224231 | 0 | 0 | 4.0e-15 |
| camelyon_resnet50_s42 | knn | 1 | 0.224231 | 0 | 0 | 2.7e-14 |
| camelyon_resnet50_s42 | knn_mean_cosine | 50 | 0.203345 | 0 | 0 | 1.2e-15 |
| camelyon_resnet50_s42 | knn | 50 | 0.196319 | 0 | 0 | 1.7e-14 |

**Maximum abs(Δ(AUROC)) = 0** over both cells and both k. The GPU kNN may be used (rule 4).
(Camelyon resnet50 s42 `knn_mean_cosine` k = 50 Δ = 0.2033451879 equals the R3 item-1 value 0.2033451879365477.)

## GPU-hours and wall clock

Estimated 0.5, cap 1. Actual: SLURM job 65255, 1 × A100, ≈ 51 min wall clock ≈ 0.85 GPU-h (`sacct` confirmation pending).
Deviations: 8 (`DEVIATIONS.md`).

**Verdict: PASS**
