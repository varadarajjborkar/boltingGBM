#!/bin/bash
# arch2 test scores with the peer variant (run after train.sh).
cd "$(dirname "$0")"
export ER_WORK_DIR=${ER_WORK_DIR:-work} A2_THREADS=4 POLARS_MAX_THREADS=4 PYTHONUNBUFFERED=1 OMP_WAIT_POLICY=PASSIVE
while kill -0 10666 2>/dev/null; do sleep 15; done
${PYTHON:-python3} run.py --stages test --variants peer > logs/arch2_test.log 2>&1
echo QUEUE3 DONE >> logs/arch2_test.log
