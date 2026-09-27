#!/bin/bash
# Baseline pipeline: raw TSVs -> normalisation -> blocking -> 2-stage LightGBM -> output/*.tsv (run from the repo root).
# Data: data/student_resource/dataset/{train,test}/*.tsv, or set ER_DATA_DIR. Intermediate files go to work/ (ER_WORK_DIR).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
(cd src && $PY convert_to_parquet.py && $PY build_normalized.py \
  && $PY p1_block.py --split train && $PY p1_block.py --split test && $PY p1_train.py)
T=$($PY -c "import json; print(json.load(open('${ER_WORK_DIR:-work}/p1/model/valid_report.json'))['best']['threshold'][0])")
(cd src && $PY p1_score.py --version repro --decision threshold --t "$T")
echo "Outputs: output_bucket/repro/matching_results.tsv and candidate_pairs.tsv"
