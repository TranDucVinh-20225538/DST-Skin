# Inventory

Paths relative to `outputs/reports/rigor_pack/`.

| path | content |
|---|---|
| `miccai_campaign/README.md` | compute, REPRO, skips, commits |
| `miccai_campaign/summary.md`, `summary.csv`, `tables.json` | all campaign tables |
| `miccai_campaign/decision_miccai_vs_midl.md` | precommit L7 readout (internal) |
| `miccai_campaign/venue_note.md` | chosen venue (MIDL 2027) and reasons |
| `miccai_campaign/access_status.json` | FM load status, STOPs |
| `miccai_campaign/trackA_foundation/summary.csv` | per backbone × seed: standard / leaky / disjoint AUROC, Δ_fit + CI, ranking, FM logit gap |
| `miccai_campaign/trackA_foundation/medical_fm_runs.csv` | medbench rows for FM probes |
| `miccai_campaign/trackA_foundation/ranking_medical_seed42.csv` | winner leaky vs disjoint, CI-significant swap, τ-b |
| `miccai_campaign/trackA_foundation/slide_identity.{csv,md}` | post-hoc mechanism check |
| `miccai_campaign/trackC_clinical_tau/tau95.csv`, `median_tpr_new_by_dataset.csv` | τ@95% TPR, realized TPR / FA / OOD pass |
| `miccai_campaign/trackC_clinical_tau/kermany_confound.{csv,md}` | post-hoc Kermany v2 / v3 split |
| `miccai_campaign/trackC_clinical_tau/risk_coverage_skip.md` | skip reason |
| `miccai_campaign/trackD_backbone_leak/side_by_side.csv` | locked Track D table |
| `miccai_campaign/trackD_backbone_leak/two_channel*.{csv,md}` | post-hoc within-model seen − unseen, 7 scores |
| `miccai_campaign/trackE_f2_medical/f2_vs_f1.csv` | F2 vs F1 per backbone |
| `miccai_campaign/cpu_cells/` | per (dataset, model, arm) Track C / E cells |
| `miccai_campaign/repro_diag/` | float32 / float64 Ledoit-Wolf diagnosis (job 63944) |
| `miccai_campaign/trackB_new_medical/`, `trackF_openmibood/`, `trackI_stain/` | STOP / skip / follow-up notes |
| `foundation_gate/gate_decision.md`, `summary.md`, `summary.csv`, `tables.json`, `access_status.json` | L4 gate (4 FMs + Virchow2 variant) |
| `foundation_gate/cells/camelyon_*.json` | Camelyon cells per model × seed |
| `foundation_gate/{breakhis,isic2019,dermamnist}/` | FM probe and medbench score JSONs |
| `leakage_lw64/` | H11c table with float64 Ledoit-Wolf, REPRO comparison, float32 vs float64 diff |

Not committed (gitignored): per-image score caches `outputs/rigor_pack/miccai_campaign/scores/`, FM feature files
`outputs/rigor_pack/foundation_gate/feats/`, job logs `logs/`.
