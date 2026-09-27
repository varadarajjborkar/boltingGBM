#!/bin/bash
# Stage-3 blend used by final steps 07-08. Run from the repo root; needs the stage-3 inputs in $ER_WORK_DIR.
W=${ER_WORK_DIR:-work}; X=$W/arch2x; C=$W/ce_out; T=$W/msrit_x
eval "$(grep -E '^(XS|RS|EF)=' models/stage3/fit_stage3.sh)"          # stage-3 input columns, as in the package fit
export ER_WORK_DIR=$W ER_THREADS=4 POLARS_MAX_THREADS=4 PYTHONUNBUFFERED=1
for f in $XS; do EF="$EF;x_$f=$T/x_${f}_valid.parquet@$T/x_${f}_test_{c}.parquet"; done
for f in $RS; do EF="$EF;r_$f=$T/r_${f}_valid.parquet@$T/r_${f}_test_{c}.parquet"; done
mkdir -p logs && ${PYTHON:-python3} postprocess/final/scripts/s3_blend.py "$EF" > logs/s3_blend.log 2>&1
