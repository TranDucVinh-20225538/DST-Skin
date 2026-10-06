# Medbench inventory (2026-10-06)

All downloads are public direct links (no login, form, licence click or DUA); stored under
`data/raw/medbench/` (not committed). Script: `scripts/rigor/medbench_download.sh`.

| dataset | source | files | size |
|---|---|---|---|
| ISIC 2019 | isic-challenge-data S3 | Training_Input.zip, GroundTruth.csv, Metadata.csv (lesion_id) | 9.8 GB |
| ISIC 2018 Task 3 | S3 | Training_LesionGroupings.csv (repo already had train / val images) | 0.5 MB |
| DermaMNIST | Zenodo 10519652 (MedMNIST+) | dermamnist.npz (28), dermamnist_224.npz | 1.1 GB |
| DermaMNIST-C / -E | Zenodo 11101338 + GitHub kakumarabhishek/Corrected-Skin-Image-Datasets | corrected / extended 224 npz, CSVs, `dermamnist_split_info.csv` (DermaMNIST index → ISIC image id), HAM10000_metadata.csv | 2.5 GB |
| BloodMNIST, OCTMNIST, PneumoniaMNIST | Zenodo 10519652 | bloodmnist_224, octmnist_64, pneumoniamnist_64 | 1.9 GB |
| Kermany v2 | Mendeley rscbjbr9sj v2 | OCT2017.tar.gz (train 83,484 / test 1,000, no val/ folder), ChestXRay2017.zip | 7.0 GB |
| Kermany v3 | Mendeley rscbjbr9sj v3 | ZhangLabData.zip (OCT train 108,309 / test 1,000; chest_xray) | 8.4 GB |
| BreakHis | UFPR direct link | BreaKHis_v1.tar.gz | 4.3 GB |
| CRC-VAL-HE-7K | Zenodo 1214456 | CRC-VAL-HE-7K.zip | 0.8 GB |
| Brain MRI (Cheng) | figshare 1512427 | 4 zips (3,064 .mat), cvind.mat | 0.9 GB |
| PAD-UFES-20 | already in repo | 2,298 images | — |

Not used: Br35H (Kaggle login → skipped; brain OOD = Kermany pediatric CXR only); OASIS-3 (DUA, not on
disk → skipped). Environment: torch-env unchanged except `h5py` (already importable, 3.11.0; a pip
install attempt failed for lack of network on the login node and was not needed). `medmnist` is not
installed and not needed (npz files are read directly).

Staging: `scripts/rigor/medbench_stage.py` → `data/staged/<ds>_256.tar` and `<ds>_343.tar` (short side
resized once, JPEG q95), copied to node-local `/tmp/$SLURM_JOB_ID` by `run_medbench.sh` (EXIT trap
removes it).
