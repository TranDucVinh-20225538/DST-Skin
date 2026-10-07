# Track I (stain ablation retrain): not started — follow-up note

Not started in this pass (precommit L4 / L6: Track I is cut 1; at most 3 GPUs are used by the campaign, one GPU is
kept for the user's other jobs). No partial stain job was submitted.

Already available (mechanism pass, commit 7fada85): frozen-backbone re-extraction of Camelyon under Macenko and
grayscale (M1 cells `outputs/reports/rigor_pack/mechanism_fix/cells/m1_camelyon_*`). These ablations reduce Δ_fit
but also destroy ID accuracy (e.g. Camelyon 0.996 → 0.56-0.60), so they cannot separate "stain drives Δ" from "the
model no longer works on the ablated input".

Exact scope for a next prompt (cheapest design that answers "does stain drive Δ?"):
1. Precommit first. Macenko-normalise (mechanism M1 parameters: Io 240, alpha 1, beta 0.15, standard HERef /
   maxCRef) train + id_val + OOD patches once, pack to a tar (node-local staging).
2. Retrain ResNet50 seed 42 on the normalised train split with the published recipe (no hyperparameter change),
   plus the two H11c slide-disjoint fold retrains (isbi_patch2 recipe, fold RNG unchanged): 3 runs, ≈ 3-4 GPU-h.
3. Score with the campaign scorer (`fm_gate_score.py --cnn` on the new indexed features): Δ_fit feature vs ReAct,
   and the within-model MSP / Energy gap; gate on ID accuracy >= 0.9 before reading Δ.
4. Optional: frozen FM embeddings (DINOv2-B / UNI) of the normalised patches, scored the same way (≈ 1 GPU-h).
