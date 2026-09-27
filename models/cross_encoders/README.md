# msrit GPU/CPU lane: every script behind the msrit inputs to the stack (reproducibility for the final package)

All models: MIT/Apache, <= 8B params (intfloat/multilingual-e5-small 118M MIT, intfloat/multilingual-e5-base 278M MIT,
microsoft/mdeberta-v3-base 278M MIT). Training never uses VALID labels (VALID = US NY + India AP/TS); stage-3 style
measurements use VALID only cross-fitted by `s1_id.hash(13) % 2`.

| script | produces (exports/) | notes |
|---|---|---|
| kaggle/bgbm-ce | ce_model (e5-small cross-encoder, 1.09M stage-2-state pairs tx/up/ka, 2 epochs) | Kaggle 2xT4 |
| kaggle/bgbm-ce-team | ce_team_valid / ce_team_test_<C> (e5 v1 on the team's pairs) | |
| kaggle/bgbm-ce2 | ce2_team_* (e5-small, tagged input "num up 9; adds holding \|\| ...") | |
| kaggle/bgbm-ce3 | ce3_team_* (e5-base, tags + "num extra" / "drops") | |
| kaggle/bgbm-ce2hi | ce2hi_* (e5 v2 on high-p pairs; veto closed) | |
| kaggle/bgbm-dr, bgbm-drtest | dense retrieval (e5 retrieval + ce2 judge): dr_test_judged / dr_test_accepted | VALID +0.00036 |
| kaggle/bgbm-tta | cetta_team_* (e5 v1 adapted with test rule-decoys + confident test pairs) | test-side |
| kaggle/bgbm-comp, bgbm-k1x | record competition test; k1 export + pipeline union | |
| kaggle/bgbm-final, bgbm-final2 | k1 pipeline stage 3 + rules (msrit's own files) | |
| lightning/l4_ce4.py | ce4_team_* (mdeberta-v3-base, tags, 1.9M pairs) | L4 |
| lightning/l4_student.py | student_team_* (mdeberta self-trained on test pseudo-labels) | L4, test-side |
| lightning/l4_loc.py, l4_prof.py, l4_combo.py, l4_s3x.py | s3x_valid / s3x_test_<C> (record competition, profile, locality) | CPU |
| research/*.py (feats_test.py, h5_test.py, bundle_valid.py) | research_valid / research_test_<C> (researcher H1/H2/H3/H5) | CPU |
| lightning/l4_dreval*.py, l4_brand.py | measurements only (dense-retrieval gates, brand dictionary: closed) | |

Kaggle inputs are other kernels' outputs (kernel_sources) plus datasets bgbm-code (this repo) and bgbm-exports (the team's
pair lists and stack scores). Paths inside the scripts assume those mounts (/kaggle/input/...) or work on Lightning.
