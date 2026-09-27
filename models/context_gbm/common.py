"""Arch2 shared paths, loaders and the VALID scorer (macro F0.5 per S1, both S1-hash halves, singleton accuracy).

Env: ER_WORK_DIR (default <root>/work_v4), A2_THREADS (default 4).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("ER_WORK_DIR", str(ROOT / "work_v4")))
OUT = WORK / "arch2"
THREADS = int(os.environ.get("A2_THREADS", "4"))
COUNTRIES = ["US", "India"]
TEST_COUNTRIES = ["US", "India", "France"]
VALID_STATES = {"US": ["ny"], "India": ["ap", "ts"]}
S2_STATES = ["tx", "up", "ka"]
P1_MIN = 0.005
B2 = 0.25
TXT_COLS = ["entity_id", "name_core", "name_phon", "addr_text", "addr_nums", "addr_state"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 ** 3


def write_atomic(df: pl.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def split_dir(split, c):
    return WORK / "p1" / split / c


def texts(split, c):
    """[side, id, name_core, name_phon, addr_text, addr_nums, addr_state] for S1 and pool of one split-country."""
    d = split_dir(split, c)
    s1 = pl.read_parquet(d / "s1.parquet", columns=TXT_COLS)
    pool = pl.read_parquet(d / "pool.parquet", columns=TXT_COLS)
    return s1, pool


def labels():
    gt = pl.read_parquet(WORK / "parquet" / "train_ground_truth.parquet").filter(pl.col("matched_entity_ids") != "")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
              .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id"), pl.lit(1, pl.Int8).alias("y")))


def valid_universe():
    """[s1_id, country, ntrue, half] for every VALID S1 (NY; AP + TS)."""
    parts = []
    for c, st in VALID_STATES.items():
        s = pl.read_parquet(split_dir("train", c) / "s1.parquet", columns=["entity_id", "addr_state"])
        parts.append(s.filter(pl.col("addr_state").is_in(st)).select(pl.col("entity_id").alias("s1_id"), pl.lit(c).alias("country")))
    u = pl.concat(parts)
    nt = labels().join(u.select("s1_id"), on="s1_id", how="semi").group_by("s1_id").agg(pl.len().alias("ntrue"))
    return (u.join(nt, on="s1_id", how="left").with_columns(pl.col("ntrue").fill_null(0),
                                                             (pl.col("s1_id").hash(13) % 2).alias("half")))


def one_s1_per_record(sc: pl.DataFrame, col="p") -> pl.DataFrame:
    return sc.sort(col, descending=True).unique(subset=["m_id"], keep="first")


def f05_table(sc: pl.DataFrame, u: pl.DataFrame, t: float, col="p") -> pl.DataFrame:
    """Per-S1 F0.5 for decision = one S1 per record, then p >= t. sc: [s1_id, m_id, y, col]."""
    pred = one_s1_per_record(sc, col).filter(pl.col(col) >= t)
    agg = pred.group_by("s1_id").agg(pl.len().alias("npred"), pl.col("y").sum().alias("tp"))
    d = u.join(agg, on="s1_id", how="left").with_columns(pl.col("npred", "tp").fill_null(0))
    p = pl.col("tp") / pl.col("npred")
    r = pl.col("tp") / pl.col("ntrue")
    f = (pl.when(pl.col("ntrue") == 0).then((pl.col("npred") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise((1 + B2) * p * r / (B2 * p + r)))
    return d.with_columns(f.alias("f"))


def score(sc: pl.DataFrame, u: pl.DataFrame, t: float, col="p") -> dict:
    d = f05_table(sc, u, t, col)
    out = {"t": t, "macro": d["f"].mean()}
    for c in COUNTRIES:
        out[c] = d.filter(pl.col("country") == c)["f"].mean()
    for h in (0, 1):
        out[f"h{h}"] = d.filter(pl.col("half") == h)["f"].mean()
    out["single"] = d.filter(pl.col("ntrue") == 0)["f"].mean()
    return out


def sweep(sc, u, ts=None, col="p"):
    ts = ts if ts is not None else [round(x, 2) for x in np.arange(0.4, 0.91, 0.05)]
    rows = [score(sc, u, t, col) for t in ts]
    best = max(rows, key=lambda r: r["macro"])
    return best, rows


def fmt(r):
    return (f"t={r['t']:.2f} macro {r['macro']:.5f} US {r['US']:.5f} India {r['India']:.5f} "
            f"h0 {r['h0']:.5f} h1 {r['h1']:.5f} single {r['single']:.4f}")
