# Exact-match generators (msrit, 27 Sep)

Inputs (in `work`, i.e. ER_WORK_DIR): `norm_v2/{train,test}_source{1,2,3}.parquet` (build_normalized output),
`parquet/train_ground_truth.parquet`, `best_valid_scored.parquet` (stack_s3a2 VALID pairs, used only to report VALID
precision), `e5hyb2_matching_results.tsv.gz` (the uploaded stack_e5hyb2 file: defines which records are unassigned),
`e5hyb2_candidate_pairs.tsv.gz`, `e5hyb2/decoy_vocab_validated.json`. Python 3.12, polars 1.x, no GPU.

| script | output | rule |
|---|---|---|
| france_exact_fix.py | france_fix/fr_exact_adds.parquet, fr_exact_moves.parquet | French record vs unique S1 with same name_core + house-number multiset + street Jaccard >= .5, raw body equal after accent folding and legal-form removal, legal forms compatible; or the unique same-body same-legal-form member of a same-address group. The advisor's same-street filter (on main: work/advisor/fr_exact_adds_samestreet.parquet) is applied after it; moves were not used. |
| usin_exact_adds.py | usin_fix/usin_exact_adds.parquet | same unique-key rule for US/India records unassigned in stack_e5hyb2, plus no added decoy-vocabulary word; prints the VALID analogue precision |
| usin_samestreet_filter.py | usin_fix/usin_exact_adds_samestreet.parquet | keeps adds with IDF-weighted street Jaccard >= .5 (weights from test S1 addresses of the country); prints VALID calibration per bucket |

Run order: `python france_exact_fix.py`, `python usin_exact_adds.py`, `python usin_samestreet_filter.py`
(each reads/writes under ER_WORK_DIR via the W variable at the top; change W if your layout differs).
