"""Rebuild the final submission (best5t, public LB 0.988757) from the 0.987692 base file by applying the
post-processing steps in order. Each step is stored as two exact pair lists: deltas/<NN>_<step>_add.parquet and
deltas/<NN>_<step>_remove.parquet (columns s1_id, m_id). What each step is and how its list was made: see
README.md in this folder.

Usage:
    python apply_deltas.py --base <dir with matching_results.tsv of the 0.987692 file> --out <out dir>
             [--check <matching_results.tsv to compare against, e.g. output/matching_results.tsv>]
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent


def read_pairs(tsv):
    """Read a matching_results.tsv into (all S1 ids, one row per matched pair)."""
    m = pl.read_csv(tsv, separator="\t", infer_schema=False, quote_char=None)
    s1 = m.select(pl.col("source1_entity_id").alias("s1_id"))
    pairs = (m.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"})
              .with_columns(pl.col("m_id").str.split(",")).explode("m_id")
              .filter(pl.col("m_id").is_not_null() & (pl.col("m_id") != "")))
    return s1, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    s1, pairs = read_pairs(Path(a.base) / "matching_results.tsv")
    steps = sorted({p.name.rsplit("_", 1)[0] for p in (HERE / "deltas").glob("*.parquet")})
    for st in steps:
        rem = pl.read_parquet(HERE / "deltas" / f"{st}_remove.parquet")
        add = pl.read_parquet(HERE / "deltas" / f"{st}_add.parquet")
        pairs = pl.concat([pairs.join(rem, on=["s1_id", "m_id"], how="anti"), add]).unique(["s1_id", "m_id"])
        assert pairs["m_id"].n_unique() == pairs.height, f"{st}: a record is matched to two S1"
        print(f"{st:28s} -{rem.height:>6,} +{add.height:>6,} -> {pairs.height:,} pairs")
    lists = pairs.group_by("s1_id").agg(pl.col("m_id").sort().str.join(",").alias("matched_entity_ids"))
    out = s1.join(lists, on="s1_id", how="left").rename({"s1_id": "source1_entity_id"})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.write_csv(Path(a.out) / "matching_results.tsv", separator="\t", quote_style="never")
    if a.check:
        _, ref = read_pairs(a.check)
        diff = pairs.join(ref, on=["s1_id", "m_id"], how="anti").height + ref.join(pairs, on=["s1_id", "m_id"], how="anti").height
        print("CHECK:", "identical pair set" if diff == 0 else f"{diff} pairs differ")


if __name__ == "__main__":
    main()
