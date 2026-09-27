"""TEST adds from the raw-name tie model: top tied S1 per address-less record with q >= .70, the record NOT assigned in
stack_e5hyb2 and not in any rider / exact / DR list already shipped. Output tie_test_adds.parquet (s1_id, m_id, country, q, k)."""
import gzip, io, os
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
t = pl.read_parquet(f"{W}/tie_test_top.parquet").filter(pl.col("q") >= 0.70)
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
lists = [f"{W}/usin_fix/usin_exact_adds_samestreet.parquet", f"{W}/france_fix/fr_exact_adds.parquet", f"{W}/dr_test_accepted.parquet",
         f"{W}/floor_rescue_test.parquet", f"{W}/hopeless_rescue_new.parquet", f"{W}/hungry_finds_test.parquet"]
other = set()
for f in lists:
    try: other |= set(pl.read_parquet(f)["m_id"].to_list())
    except Exception as e: print("skip", f, e)
new = t.filter(~pl.col("m_id").is_in(list(assigned)) & ~pl.col("m_id").is_in(list(other)))
print("tie records q>=.70:", t.height, "| assigned in e5hyb2:", t.filter(pl.col("m_id").is_in(list(assigned))).height, "| new:", new.height)
for c in ("US", "India", "France"):
    z = new.filter(pl.col("country") == c)
    print(f"   {c:6s} new adds q>=.70 {z.height:5d}  q>=.75 {z.filter(pl.col('q') >= .75).height:5d}  q>=.80 {z.filter(pl.col('q') >= .8).height:5d}  q>=.90 {z.filter(pl.col('q') >= .9).height:5d}")
new.select("s1_id", "m_id", "country", "q", "k").write_parquet(f"{W}/tie_test_adds.parquet")
print("TIEADDS_DONE")
