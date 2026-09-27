"""Moved house-number classes on kept pairs (label-free): class counts for the count-shape test."""
import os
import re
import numpy as np
import polars as pl
from rapidfuzz import fuzz
S = os.environ.get("ER_OUT", "work/checks")  # output folder
BASE = "output_bucket/stack_pkf2_a1_fr_fxs_us_sf_swr_R_drB_mhk2_fr2/matching_results.tsv"
def nums(s): return [int(x) for x in re.findall(r"\d+", s) if len(x) <= 4]
m = pl.read_csv(BASE, separator="\t", infer_schema=False, quote_char=None).fill_null("")
kc = m.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k")).select(pl.col("source1_entity_id").alias("s1_id"), "k")
pairs = (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"}))
res = {}
for C in ["France", "US", "India"]:
    s1 = pl.read_parquet(f"work_v4/p1/test/{C}/s1.parquet", columns=["entity_id", "name_core", "addr_text"]).fill_null("")
    pool = pl.read_parquet(f"work_v4/p1/test/{C}/pool.parquet", columns=["entity_id", "name_core", "addr_text"]).fill_null("")
    d = (pairs.join(s1.rename({"entity_id": "s1_id", "name_core": "sn", "addr_text": "sa"}), on="s1_id")
              .join(pool.rename({"entity_id": "m_id", "name_core": "rn", "addr_text": "ra"}), on="m_id"))
    keep = []
    for sn, rn, sa, ra in d.select("sn", "rn", "sa", "ra").iter_rows():
        a, b = nums(sa), nums(ra)
        if not a or not b or sorted(a) == sorted(b) or set(a) <= set(b) or set(b) <= set(a):
            keep.append(None); continue
        keep.append("exact" if sn == rn else ("near" if fuzz.token_sort_ratio(sn, rn) >= 85 else "other"))
    d = d.with_columns(pl.Series("mv", keep)).filter(pl.col("mv").is_not_null())
    print(f"\n== {C}: kept pairs with moved numbers {d.height:,}")
    for c in ["exact", "near", "other"]:
        x = d.filter(pl.col("mv") == c)
        one = x.group_by("s1_id").len("n").filter(pl.col("n") == 1).join(kc, on="s1_id")
        h = np.bincount(one["k"].clip(0, 8).to_numpy(), minlength=9) / max(one.height, 1) * 100
        print(f"  {c:6s} {x.height:8,} one-pair S1 {one.height:7,} P(k=1) {h[1]:4.1f} P(k=2) {h[2]:4.1f} mean {one['k'].mean():.2f}")
    res[C] = d
res["France"].select("s1_id", "m_id", "mv").write_parquet(f"{S}/fr_moved_all.parquet")
