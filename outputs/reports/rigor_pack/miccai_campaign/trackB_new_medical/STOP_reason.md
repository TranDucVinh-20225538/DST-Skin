# Track B (new medical datasets with strong group IDs): STOP

Decided in the precommit (L1), before any campaign number.

- Chest X-ray with patient ID (NIH ChestX-ray14 / CheXpert / MIMIC-CXR): not on the HPC. A search of the
  project data tree and the shared storage (directory names containing chest / cxr / nih / chexpert / mimic,
  depth 4) found nothing. NIH ChestX-ray14 is ~45 GB of images and CheXpert / MIMIC-CXR need a registration /
  DUA, which we do not accept for the user. No download in this campaign.
- TCGA with site as group: the only TCGA data on the HPC is an RNA-seq table (BRCA) in another user's home
  directory: not images, not ours. Not used.
- The 90% recoverable-group-ID condition therefore cannot be met; Track B is not run. No Phase-0 audit, no A-gap /
  A-fit, no retrain. 0 GPU-h.

Next step if wanted later (not run): stage NIH ChestX-ray14 (patient ID in `Data_Entry_2017.csv`, ~100% images)
and run the medbench Phase-0 split audit with patient as group.
