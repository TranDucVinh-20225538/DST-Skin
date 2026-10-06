# Medbench Phase 0 split audit

Precommit `decisions/precommit_group_leakage_medbench_2026-10-06.md` (0fa52af). RNG default_rng(0) per dataset.
status: ok = within 25% of the expected value; S3-PAUSE = > 25% off (look for a mapping / parse bug).

## dermamnist

| check | value | expected | status | note |
|---|---|---|---|---|
| train images (npz) | 7007 | 7007.0 | ok |  |
| train rows in mapping CSV | 7007 | 7007.0 | ok |  |
| val images (npz) | 1003 | 1003.0 | ok |  |
| val rows in mapping CSV | 1003 | 1003.0 | ok |  |
| test images (npz) | 2005 | 2005.0 | ok |  |
| test rows in mapping CSV | 2005 | 2005.0 | ok |  |
| images with lesion_id (fraction) | 1.0000 | 1.0 | ok |  |
| mapping label agreement (npz label == HAM dx) | 1.0000 | 1.0 | ok |  |
| pixel check: fraction of 300 sampled images with rho >= 0.98 | 0.9467 | 1.0 | ok | median rho 0.9940, min 0.9689 |
| test images whose lesion is in train | 693 | 886.0 | ok |  |
| test lesions shared with train | 641 | 641.0 | ok |  |
| val images whose lesion is in train | 348 |  |  |  |
| val lesions shared with train | 332 |  |  |  |
| test images whose lesion is in val | 120 |  |  |  |
| test lesions shared with val | 113 |  |  |  |
| DermaMNIST-C lesions shared across splits | 0 | 0.0 |  |  |
| DermaMNIST-E lesions shared across splits (HAM images) | 0 | 0.0 |  |  |
| DermaMNIST-E test images outside HAM (ISIC 2018 test, no lesion id) | 1511 |  |  | disjoint by source |
| repo ISIC 2018 val images with a public lesion id | 0 |  |  | of 193; val lesion ids not public -> train/val lesion overlap not determinable |
| S1: standard test images sharing a lesion with train (fraction) | 0.3456 |  |  | leaky |
| B fold0: train / val / seen / unseen images | 4199 / 466 / 343 / 5007 |  |  |  |
| B fold0: unseen groups | 3751 |  |  |  |
| B fold1: train / val / seen / unseen images | 4207 / 468 / 332 / 5008 |  |  |  |
| B fold1: unseen groups | 3719 |  |  |  |
| M_std train images / epochs | 7007 / 43 |  |  |  |
| A: test_seen / test_unseen images | 693 / 1312 |  |  |  |
| A: test_unseen lesions | 1227 |  |  |  |

## isic2019

| check | value | expected | status | note |
|---|---|---|---|---|
| images | 25331 | 25331.0 | ok |  |
| images with lesion_id | 23247 | 23247.0 | ok |  |
| S2: lesion_id coverage | 0.9177 |  |  | ok |
| distinct lesions | 11847 | 11847.0 | ok |  |
| composition: lesion prefix BCN | 12413 |  |  |  |
| composition: lesion prefix HAM | 10015 |  |  |  |
| composition: lesion prefix no_lesion_id | 2084 |  |  |  |
| composition: lesion prefix MSK | 819 |  |  |  |
| images present in zip | 25331 | 25331.0 | ok |  |
| DF/VASC images whose lesion also has an ID-class image (removed from OOD) | 0 | 0.0 |  |  |
| S1/expected: test images whose lesion is in train (fraction) | 0.6081 | 0.6 | ok | leaky |
| M_std train images (before val) / epochs | 19871 / 16 |  |  |  |
| A: test_seen / test_unseen images | 3021 / 1947 |  |  |  |
| A: test_unseen groups | 1829 |  |  |  |
| OOD images (DF + VASC) | 492 |  |  |  |
| B fold0: train / val / seen / unseen images | 10090 / 1121 / 1216 / 12412 |  |  |  |
| B fold0: unseen groups | 6873 |  |  |  |
| B fold1: train / val / seen / unseen images | 10093 / 1122 / 1197 / 12427 |  |  |  |
| B fold1: unseen groups | 6810 |  |  |  |
| sensitivity (lesion_id only): ID images | 22755 |  |  |  |

## kermany

| check | value | expected | status | note |
|---|---|---|---|---|
| v2 fingerprint: counts by split | {'train': 83484, 'test': 1000} |  |  | sha256(sorted names)[:16]=a6b2b457960fd2ef |
| v2 has val/ folder | False |  |  |  |
| S2: v2 patient-id parse rate | 1.0000 |  |  | ok |
| v3 fingerprint: counts by split | {'train': 108309, 'test': 1000} |  |  | sha256(sorted names)[:16]=c540e4fe493ab6f0 |
| v3 has val/ folder | False |  |  |  |
| S2: v3 patient-id parse rate | 1.0000 |  |  | ok |
| v2 images (train / val / test) | 83484 / 0 / 1000 |  |  | expected 83484 / 32 / 968 |
| v2 test images whose patient is in v2 train (fraction) | 0.9140 | 0.92 | ok |  |
| v3 test images whose patient is in v3 train (fraction) | 0.0000 |  |  | expected 0 |
| v3 test images whose patient is in v2 train (fraction) | 0.0000 |  |  |  |
| v3 filenames also present in v2 (fraction) | 0.7044 |  |  |  |
| v3 test filenames present in v2 (any split) (fraction) | 0.0000 |  |  |  |
| v3 test filenames present in v2 test (fraction) | 0.0000 |  |  |  |
| DRUSEN patients also with ID-class images (count, both versions) | 475 |  |  | descriptive |
| S1: v2 standard test (ID classes) sharing a patient with train | 0.8853 |  |  | leaky |
| M_std(v2) train images (ID, before val) / epochs | 74868 / 10 |  |  |  |
| A: test_seen / test_unseen(v2) / unseen_extra(v3 test) images | 664 / 86 / 750 |  |  |  |
| A: ID_unseen patients (v2 test unseen + v3 extra) | 557 |  |  |  |
| A OOD: v2 DRUSEN images (all splits) | 8866 |  |  |  |
| B fold0: train / val / seen / unseen images | 38207 / 4245 / 7484 / 49757 |  |  |  |
| B fold0: unseen groups | 2220 |  |  |  |
| B fold1: train / val / seen / unseen images | 38069 / 4230 / 7458 / 49936 |  |  |  |
| B fold1: unseen groups | 2239 |  |  |  |
| far OOD: pediatric CXR test images (v3) | 624 | 624.0 | ok |  |

## breakhis

| check | value | expected | status | note |
|---|---|---|---|---|
| images | 7909 | 7909.0 | ok |  |
| S2: filename parse rate | 1.0000 |  |  | ok |
| patients | 81 | 82.0 | ok |  |
| images per patient (min / median / max) | 38 / 90 / 246 |  |  |  |
| patients in > 1 subtype | 1 |  |  | 14-13412 |
| OOD-subtype images of patients who also have ID images (removed from OOD) | 0 |  |  |  |
| ID images / OOD images | 6896 / 1013 |  |  |  |
| S1: random image 80/20 split, test images whose patient is in train | 1.0000 |  |  | leaky |
| repeat 0: P_out patients / images; M_std train (before val) / seen | 16 / 1811; 4068 / 1017 |  |  | per-repeat cell < 20 groups (underpowered); pooled over repeats |
| repeat 1: P_out patients / images; M_std train (before val) / seen | 16 / 1484; 4330 / 1082 |  |  | per-repeat cell < 20 groups (underpowered); pooled over repeats |
| repeat 2: P_out patients / images; M_std train (before val) / seen | 15 / 1460; 4349 / 1087 |  |  | per-repeat cell < 20 groups (underpowered); pooled over repeats |
| repeat 3: P_out patients / images; M_std train (before val) / seen | 14 / 1143; 4602 / 1151 |  |  | per-repeat cell < 20 groups (underpowered); pooled over repeats |
| repeat 4: P_out patients / images; M_std train (before val) / seen | 11 / 998; 4718 / 1180 |  |  | per-repeat cell < 20 groups (underpowered); pooled over repeats |
| M_std(r) train images (mean, before val) / epochs | 4413 / 50 |  |  |  |
| pooled ID_unseen over 5 repeats: patients / images | 72 / 6896 |  |  |  |
| far OOD: CRC-VAL-HE-7K images | 7180 | 7180.0 | ok |  |

## brain_cheng

| check | value | expected | status | note |
|---|---|---|---|---|
| images | 3064 | 3064.0 | ok |  |
| patients (PID) | 233 | 233.0 | ok |  |
| S2: PID present | 1.0000 |  |  |  |
| cvind length / folds | 3064 / [1, 2, 3, 4, 5] |  |  | variable cvind |
| patients spanning > 1 cvind fold | 0 | 0.0 |  |  |
| S1: image 80/20 test images sharing a patient with train | 0.9967 |  |  | leaky |
| M_std train images (before val) / epochs | 2451 / 50 |  |  |  |

