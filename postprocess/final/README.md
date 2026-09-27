# Final post-processing steps (0.987692 file -> final file, public LB 0.988757)

The last leaderboard-scored file before these steps is `stack_pkf2_a1_fr_fxs_us_sf_swr_R_drB_mhk2_fr2` (public LB 0.987692),
built by the chain in the main README. The final file (`output/matching_results.tsv`) applies eight more steps. Each step
is stored as exact pair lists in `deltas/` (`<step>_add.parquet`, `<step>_remove.parquet`; columns `s1_id`, `m_id`).
The lists are derived from the competition test data, so they ship only in the competition submission package, not here.

    python apply_deltas.py --base <dir of the 0.987692 file> --out <out dir> --check output/matching_results.tsv

rebuilds the final file and checks it pair by pair (verified: identical pair set).

| Step | Adds | Removes | What it is | How the list was made |
|---|---|---|---|---|
| 01 r3_recall_adds | 531 | 0 | India round-3 record-side re-search adds (gate 0.9, receiver holds >= 1 record) + teammate floor-parser and empty-S1 riders | `postprocess/fixes` re-search, gate calibrated on training states |
| 02 us_research_adds | 201 | 0 | US record-side re-search adds (all states, gate 0.9; training precision 0.992) | same method as 01 |
| 03 fr_pocketA_swaps | 0 | 6,003 | French pairs where a common French word is swapped or added (test-only decoy family) | label-free count-shape test vs French controls |
| 04 us_decoy_list1 | 0 | 850 | US decoys: legal form added and house number offset with a digit dropped (13.5x train-truth density) | truth-density test per cell |
| 05 best2_overlays | 3,885 | 4,027 | teammate high-precision adds (US, India, France), 2,233 US decoy-family removals, 586 India legal-type changes, 388 India exact-name shift adds | truth-density test per cell (test density vs train-truth density) |
| 06 best3_legaltype | 0 | 567 | India and France legal-type / legal-family changes (test-only decoys) | truth-density test, France control rule |
| 07 best4g_removals | 0 | 1,982 | French multi-word-drop swaps (944), swaps hidden in domain names (133), and 905 US/India pairs a stage-3 blend drops below 0.75 | `scripts/s3_blend.py` (blend of LightGBM variants on the stage-3 inputs, S1-half cross-fit), `scripts/flips.py` |
| 08 best5t_adds | 2,943 | 0 | stage-3 blend adds (VALID precision 0.765) and adds into S1 holding one record with blend score >= 0.7 (no empty receivers) | `scripts/flips.py`, `scripts/radd.py`, then trimmed by review |

`scripts/` holds the scripts exactly as run. Paths inside them point to the run's working folders. The label-free tests
and every accepted or rejected step are described in `docs/EXPERIMENT_LOG.md`.
