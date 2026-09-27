# Flip overlay: where the blended stage 3 disagrees with the package stage 3 (US/India), vs best3. VALID calibration too.
import polars as pl, json, sys
t0, tb = float(sys.argv[1]), float(sys.argv[2]); SC = sys.argv[3]  # new score file suffix dir
B = "output_bucket/stack_pkf2_a1_fr_fxs_us_sf_swr_R_drB_mhk2_fr2_r3_us_fra_us1_best3/matching_results.tsv"
m = pl.read_csv(B, separator="\t", infer_schema=False, quote_char=None)
kept = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename(
    {"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"}).filter(pl.col("m_id") != "")
blk = pl.concat([pl.read_parquet(f"work/errfix/out/{f}.parquet").select("s1_id", "m_id") for f in
      ["anom_us2_stage4_bad_adds", "anom_us2_stage4_marginal_adds", "anom_us2_union", "anom_in2_legal_type_decoy",
       "anom_in3_legal_type2", "anom_us_legal_shorter"]])
def flips(df, pold, pnew):
    df = df.with_columns(pl.col(pnew).rank("ordinal", descending=True).over("m_id").alias("rn"))
    rem = df.filter((pl.col(pold) >= t0) & (pl.col(pnew) < tb))
    add = df.filter((pl.col(pold) < t0) & (pl.col(pnew) >= tb) & (pl.col("rn") == 1))
    return rem, add
out = {}
v = pl.read_parquet(f"{SC}/blend_oof.parquet")
r, a = flips(v, "p0", "pb")
print(f"VALID removals {r.height} false share {1 - r['y'].mean():.3f} | adds {a.height} precision {a['y'].mean() if a.height else 0:.3f}")
nS1 = {"US": 49_000, "India": 64_500}
R, A = [], []
for c in ["US", "India"]:
    o = pl.read_parquet(f"work_v4/p1/test/{c}/stage2_scored_model_v10pkf2_s3.parquet").rename({"p": "po"})
    n = pl.read_parquet(f"{SC}/test_{c}.parquet").rename({"p": "pn"})
    df = o.join(n, on=["s1_id", "m_id"])
    r, a = flips(df, "po", "pn")
    r = r.join(kept, on=["s1_id", "m_id"], how="semi")
    a = a.join(kept.select("m_id"), on="m_id", how="anti").join(blk, on=["s1_id", "m_id"], how="anti")
    print(f"test {c}: removals in best3 {r.height}, adds (record free, not blocked) {a.height}, pairs {df.height}")
    R.append(r.select("s1_id", "m_id").with_columns(pl.lit("remove").alias("action"), pl.lit(c).alias("country")))
    A.append(a.select("s1_id", "m_id").with_columns(pl.lit("add").alias("action"), pl.lit(c).alias("country")))
pl.concat(R + A).write_parquet(f"{SC}/blend_flips.parquet")
