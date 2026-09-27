"""Candidate adds for final step 08: blend adds and adds into S1 that hold one record (density-checked vs VALID)."""
import polars as pl, glob
P = "output_bucket/stack_pkf2_a1_fr_fxs_us_sf_swr_R_drB_mhk2_fr2_r3_us_fra_us1_best4g/matching_results.tsv"
import os; S = os.environ["TMPDIR"] + "/blend"
kept = pl.read_csv(P, separator="\t", infer_schema=False, quote_char=None).with_columns(pl.col("matched_entity_ids").str.split(",")).explode(
    "matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"}).filter(pl.col("m_id") != "")
k = kept.group_by("s1_id").agg(pl.len().alias("k"))
bl = [f for f in glob.glob("work/errfix/out/anom_*.parquet") + glob.glob("work/errfix/out/fr_*.parquet") + ["work/advisor/fr_swap_remove_realword.parquet", S + "/g_remove.parquet"] if "tie_" not in f]
blk = pl.concat([pl.read_parquet(f).select(pl.col("s1_id").cast(pl.String), pl.col("m_id").cast(pl.String)) for f in bl if "s1_id" in pl.read_parquet_schema(f)]).unique()
print("block pairs", blk.height)
out = []
nS = {"US": 663106, "India": 809986, "France": 259452}
for c, f in (("US", "stage2_scored_model_v10pkf2_s3"), ("India", "stage2_scored_model_v10pkf2_s3"), ("France", "stage2_scored_model_v10e_v10_s3")):
    d = pl.read_parquet(f"work_v4/p1/test/{c}/{f}.parquet")
    d = d.join(kept.select("m_id"), on="m_id", how="anti").with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r")).filter(pl.col("r") == 1)
    d = d.join(k, on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
    a = d.filter((pl.col("k") == 1) & (pl.col("p") >= 0.6) & (pl.col("p") < 0.75)).join(blk, on=["s1_id", "m_id"], how="anti")
    print(c, "k1 adds", a.height, "per1k %.2f (VALID 1.59)" % (a.height / nS[c] * 1000))
    out.append(a.select("s1_id", "m_id").with_columns(pl.lit("k1").alias("src"), pl.lit(c).alias("country")))
b = pl.read_parquet(S + "/flips725.parquet").filter(pl.col("action") == "add").join(blk, on=["s1_id", "m_id"], how="anti")
print("blend adds", b.height, "per1k %.2f (VALID 2.56)" % (b.height / 1473092 * 1000))
out.append(b.select("s1_id", "m_id", pl.lit("blend").alias("src"), "country"))
o = pl.concat(out).unique("m_id", keep="first")
print(o.group_by("src", "country").len().sort("src", "country")); o.write_parquet(S + "/r_adds.parquet")
