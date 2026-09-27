"""Test pairs that lack the teammate's package features (s3x / research; ~9% of test pairs, none in the decision zone) keep
the current stack's p (model_v10pc_s3) instead of the package model's p, whose inputs for them would be out of the VALID range.
Reads work_v4/p1/test/<C>/stage2_scored_<M>.parquet (raw package p), writes it back blended (raw kept as *_raw.parquet).
Usage: python postprocess/overlay/pkf_fallback.py model_v10pkf_s3"""
import os
import shutil
import sys
from pathlib import Path

import polars as pl

M = sys.argv[1]
E = Path(os.environ.get("ER_EXPORTS", "exports"))
R = {"India": Path("work/errfix/inputs/research_test_India_partial.parquet")} if M == "model_v10pkf_s3" else {}
for c in ("US", "India", "France"):
    d = Path(f"work_v4/p1/test/{c}")
    raw = d / f"stage2_scored_{M}_raw.parquet"
    if not raw.exists():
        shutil.copy(d / f"stage2_scored_{M}.parquet", raw)
    new = pl.read_parquet(raw).select("s1_id", "m_id", "p")
    old = pl.read_parquet(d / "stage2_scored_model_v10pc_s3.parquet").select("s1_id", "m_id", pl.col("p").alias("p_old"))
    cov = pl.read_parquet(E / f"s3x_test_{c}.parquet", columns=["s1_id", "m_id"]).unique().join(
        pl.read_parquet(R.get(c, E / f"research_test_{c}.parquet"), columns=["s1_id", "m_id"]).unique(), on=["s1_id", "m_id"])
    x = new.join(old, on=["s1_id", "m_id"], how="left").join(cov.with_columns(pl.lit(True).alias("cov")), on=["s1_id", "m_id"], how="left")
    x = x.with_columns(pl.when(pl.col("cov")).then(pl.col("p")).otherwise(pl.coalesce("p_old", "p")).alias("p"))
    x.select("s1_id", "m_id", "p").write_parquet(d / f"stage2_scored_{M}.parquet")
    print(f"{c}: {x.height} pairs, covered {x['cov'].fill_null(False).mean():.4f}, fallback to old p {(~x['cov'].fill_null(False)).sum()}", flush=True)
