#!/bin/bash
# Fit stage 3 for test: pkf = a2+ce+l12+e5 + teammate package (ce2, ce4 as d4, cetta as ta, s3x 16 cols, research 34 cols), no context.
W=${ER_WORK_DIR:-work}; X=$W/arch2x; C=$W/ce_out; T=$W/msrit_x; Q=work/queue_pkfit.log; L=logs
XS="rec_ncand p_rank_rec c_rank_rec p_best_other c_best_other p_margin c_margin rec_s3 rec_native rec_noaddr rec_domain rec_nlen loc_exact loc_alias loc_conflict rec_has_loc"
RS="hn_frac_exact hn_frac_near hn_frac_none hn_lmax_exact hn_prim_rel hn_multi_exact hn_n_s1 hn_n_rec ne_raw_eq ne_ci_eq ne_ws_eq ne_an_eq ne_set_eq ne_extra_legal ne_extra_other ne_miss_legal ne_miss_other ne_typo ne_title ne_case ne_leet ne_phone ne_brack ne_dup dw_rare_in dw_rare_idf dw_cov_s1 dw_cov_rec dw_extra_maxidf dw_s1_namecnt dw_rec_namecnt s1_addrcnt rec_addrcnt rec_keytok_cnt"
export ER_WORK_DIR=$W ER_THREADS=4 POLARS_MAX_THREADS=4 PYTHONUNBUFFERED=1
cd src
n=pkf; M=$W/p1/model_v10$n; mkdir -p $M/stage3
ln -sf ../model_v10/valid_scored.parquet $M/valid_scored.parquet
ln -sf ../../model_v10/stage3/cn_valid.parquet $M/stage3/cn_valid.parquet; ln -sf ../../model_v10/stage3/cn_test.parquet $M/stage3/cn_test.parquet
for c in US India France; do ln -sf stage2_scored_model_v10.parquet $W/p1/test/$c/stage2_scored_model_v10$n.parquet; done
EF="a2=$X/valid_peer.parquet@$X/test_{c}.parquet;ce=$C/ce_valid.parquet@$C/ce_test_{c}.parquet;l12=$C/ce_valid_l12.parquet@$C/ce_test_l12_{c}.parquet;e5=$T/ce_team_valid.parquet@$T/ce_team_test_{c}.parquet;ce2=$T/ce2_team_valid.parquet@$T/ce2_team_test_{c}.parquet;d4=$T/ce4_team_valid.parquet@$T/ce4_team_test_{c}.parquet;ta=$T/cetta_team_valid.parquet@$T/cetta_team_test_{c}.parquet"
EO="a2=$X/valid_peer.parquet;ce=$C/ce_valid.parquet;l12=$C/ce_valid_l12.parquet;e5=$T/ce_team_valid.parquet;ce2=$T/ce2_team_valid.parquet;d4=$T/ce4_team_valid.parquet;ta=$T/cetta_team_valid.parquet"
for f in $XS; do EF="$EF;x_$f=$T/x_${f}_valid.parquet@$T/x_${f}_test_{c}.parquet"; EO="$EO;x_$f=$T/x_${f}_valid.parquet"; done
for f in $RS; do EF="$EF;r_$f=$T/r_${f}_valid.parquet@$T/r_${f}_test_{c}.parquet"; EO="$EO;r_$f=$T/r_${f}_valid.parquet"; done
echo "[$(date +%T)] start fit $n" >> $Q
${PYTHON:-python3} models/stage3/stage3.py fit --model $M --extra "$EF" > $L/fit_$n.log 2>&1 || { echo "[$(date +%T)] FAIL fit $n" >> $Q; exit 1; }
${PYTHON:-python3} work/s3_save_oof.py model_v10$n "$EO" > $L/oof_$n.log 2>&1 || { echo "[$(date +%T)] FAIL oof $n" >> $Q; exit 1; }
echo "[$(date +%T)] PK FIT DONE" >> $Q
