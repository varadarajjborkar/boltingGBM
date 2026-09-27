"""Filter v10 f_* columns of test feats_all parts to the arch2 stage-2 pair list (runs where feats_all lives).

Usage: python pull_fcols.py --pairs pairs_test_US.parquet --feats $ER_WORK_DIR/p1/test/US/feats_all \
           --cols $ER_WORK_DIR/p1/model_v10/feature_cols.json --out fcols_test_US.parquet
pairs: [s1_id, m_id] (arch2 p1_test_<c> with p1 >= 0.005). Output: [s1_id, m_id, f_*] float32, one row per pair.
Light: one part in memory at a time, polars only.
"""
import argparse
import json
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--cols", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = json.loads(Path(a.cols).expanduser().read_text())
    keys = pl.read_parquet(a.pairs, columns=["s1_id", "m_id"])
    res = []
    for p in sorted(Path(a.feats).expanduser().glob("part_*.parquet")):
        res.append(pl.read_parquet(p, columns=["s1_id", "m_id"] + cols).join(keys, on=["s1_id", "m_id"], how="semi")
                     .with_columns(pl.col(cols).cast(pl.Float32)))
    d = pl.concat(res).unique(["s1_id", "m_id"], keep="first")
    d.write_parquet(a.out)
    print(f"{a.out}: {d.height} of {keys.height} pairs")


if __name__ == "__main__":
    main()
