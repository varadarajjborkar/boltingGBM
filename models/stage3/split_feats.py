import os
"""Split the teammate's wide feature files (s3x_test_<C>, research_test_<C>) into one stage-3 extra per column
(s1_id, m_id, p), named like the VALID ones: x_<col>_test_<C>.parquet, r_<col>_test_<C>.parquet."""
import sys
from pathlib import Path
import polars as pl
T = Path(os.environ.get("ER_WORK_DIR", "work")) / "msrit_x"
for c in sys.argv[1].split(","):
    for wide, pre in (("s3x", "x_"), ("research", "r_")):
        f = T / f"{wide}_test_{c}.parquet"
        if not f.exists():
            print("missing", f); continue
        d = pl.read_parquet(f).unique(["s1_id", "m_id"])
        for col in [x for x in d.columns if x not in ("s1_id", "m_id")]:
            if not (T / f"{pre}{col}_valid.parquet").exists():
                continue
            d.select("s1_id", "m_id", pl.col(col).cast(pl.Float64).alias("p")).write_parquet(T / f"{pre}{col}_test_{c}.parquet")
        print(c, wide, d.height, "rows", flush=True)
