#!/bin/bash
#SBATCH --job-name=med-dl
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/med_dl_%j.out
# Public downloads only (no login / form / licence click). Resumable (curl -C -).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
D=data/raw/medbench; mkdir -p $D/{isic2019,isic2018,medmnist,dermamnist_ce,kermany_v2,kermany_v3,breakhis,crc_val,brain_cheng}
get() { local out=$1 url=$2; [ -f "$out.ok" ] && return 0
  for t in 1 2 3 4 5; do curl -sSL --fail -C - --retry 5 -o "$out" "$url" && { touch "$out.ok"; echo "OK $out $(stat -c %s "$out")"; return 0; }; sleep 30; done
  echo "FAIL $out $url"; return 1; }
S3=https://isic-challenge-data.s3.amazonaws.com
Z=https://zenodo.org/records
M=https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files
(
get $D/isic2019/ISIC_2019_Training_GroundTruth.csv $S3/2019/ISIC_2019_Training_GroundTruth.csv
get $D/isic2019/ISIC_2019_Training_Metadata.csv $S3/2019/ISIC_2019_Training_Metadata.csv
get $D/isic2018/ISIC2018_Task3_Training_LesionGroupings.csv $S3/2018/ISIC2018_Task3_Training_LesionGroupings.csv
for f in dermamnist.npz dermamnist_224.npz bloodmnist_224.npz octmnist_64.npz pneumoniamnist_64.npz; do get $D/medmnist/$f "$Z/10519652/files/$f?download=1"; done
for f in DermaMNIST-C.csv DermaMNIST-E.csv dermamnist_corrected_224.npz dermamnist_extended_224.npz; do get $D/dermamnist_ce/$f "$Z/11101338/files/$f?download=1"; done
get $D/crc_val/CRC-VAL-HE-7K.zip "$Z/1214456/files/CRC-VAL-HE-7K.zip?download=1"
for id in 3381290 3381293 3381296 3381302; do get $D/brain_cheng/figshare_$id.zip https://ndownloader.figshare.com/files/$id; done
get $D/brain_cheng/cvind.mat https://ndownloader.figshare.com/files/7005344
) &
get $D/isic2019/ISIC_2019_Training_Input.zip $S3/2019/ISIC_2019_Training_Input.zip &
get $D/breakhis/BreaKHis_v1.tar.gz http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz &
( get $D/kermany_v2/OCT2017.tar.gz $M/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded
  get $D/kermany_v2/ChestXRay2017.zip $M/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded ) &
get $D/kermany_v3/ZhangLabData.zip $M/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded &
wait
echo DONE; ls -la $D/*/
