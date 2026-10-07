# Deviations: precommit_foundation_leakage_gate_2026-10-07.md

1. (before any FM number) The explicit STOP on multi-seed is lifted by the user's one-pass campaign
   authorisation (`precommit_miccai_full_campaign_2026-10-07.md`, committed before any FM number): Camelyon
   seeds 43 / 44 are queued together with seed 42. The L4 gate is still computed at seed 42 and reported;
   `PROMPT_SEED_SWEEP_FOUNDATION_2026-10-07.md` is not needed (the sweep runs in the campaign).
2. (before any FM number) Virchow2 is added by the campaign precommit (L2 item 4) as an extra FM. The gate
   L4 is computed on the FMs listed here (DINOv2-B, UNI, CONCH v1.5, optional DINOv2-L) and, separately,
   with Virchow2 included; both are reported.
3. (technical, before any number) The smoke subset is an evenly strided subset, not the first n images
   (the first n images come from a single slide and leave folds empty).
4. CONCH v1 (`MahmoodLab/CONCH`): GatedRepoError 403 with the configured token → STOP; the precommitted
   fallback CONCH v1.5 (TITAN) is used.
5. Dependencies installed into torch-env: einops 0.8.1, einops-exts 0.0.4 (`--no-deps`), needed by
   `conch_v1_5.py`.
6. GPU concurrency <= 3 (1 GPU kept for the user's other jobs).
