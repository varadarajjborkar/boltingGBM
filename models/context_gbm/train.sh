#!/bin/bash
# arch2 training: VALID feats after VALID p1, train, VALID, test feats.
cd "$(dirname "$0")"
export ER_WORK_DIR=${ER_WORK_DIR:-work} A2_THREADS=4 POLARS_MAX_THREADS=4 PYTHONUNBUFFERED=1 OMP_WAIT_POLICY=PASSIVE
PY=${PYTHON:-python3}
L=logs
while kill -0 3293 2>/dev/null; do sleep 15; done
$PY run.py --stages feats,context --only valid:US,valid:India > $L/arch2_feats_valid.log 2>&1
$PY run.py --stages train > $L/arch2_train.log 2>&1
$PY run.py --stages valid > $L/arch2_valid.log 2>&1
$PY run.py --stages feats,context --only test:US,test:India,test:France > $L/arch2_feats_test.log 2>&1
echo QUEUE2 DONE >> $L/arch2_valid.log
