"""Near-miss band for India: argmax pairs with stage-3 p in [.5, .75) (not kept), on records unassigned in stack_e5hyb2.
Per state: count per 1k S1, and whether they land on S1 SHORT a copy (receiving S1's current count distribution vs the state)
- the same fill test that the DR adds passed. VALID (AP/TS, labels): precision of the same band as a calibration."""
import gzip, io, os
from collections import Counter
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); E = f"{W}/e5hyb2"
def dist(counts):
    n = len(counts); c = Counter(min(x, 6) for x in counts)
    return [c[i] / n for i in range(7)] + [sum(counts) / n]
te = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state"]).filter(pl.col("country") == "India")
ST = dict(zip(te["entity_id"].to_list(), te["addr_state"].to_list()))
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
t = pl.read_parquet(f"{E}/stage3_test_pairs_India.parquet").select("s1_id", "m_id", "p", "grp")
t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r"))
band = t.filter((pl.col("r") == 1) & (pl.col("p") >= 0.5) & (pl.col("p") < 0.75) & ~pl.col("m_id").is_in(list(assigned)) & ~pl.col("grp").is_in(["shift_pure", "shift_ms_only"]))
recv = Counter(band["s1_id"].to_list())
print(f"India band pairs (argmax, p .5-.75, record unassigned, not shift decoys): {band.height}")
print(f"{'':26s} {'n':>7s} " + " ".join(f"{k:>6s}" for k in ("0", "1", "2", "3", "4", "5", "6+", "mean")) + "  band/1k")
for st in ("mh", "dl", "up", "ka", "tn", "gj", "ts", "ap"):
    ids = [s for s, x in ST.items() if x == st]
    rc = [pt.get(s, 0) for s in ids if recv.get(s, 0)]
    print(f"{st + ' state':26s} {len(ids):7d} " + " ".join(f"{x:6.3f}" for x in dist([pt.get(s, 0) for s in ids])) + f"  {1000 * sum(recv.get(s, 0) for s in ids) / len(ids):6.1f}")
    if rc: print(f"{st + ' receiving S1':26s} {len(rc):7d} " + " ".join(f"{x:6.3f}" for x in dist(rc)))
v = pl.read_parquet(f"{E}/stage3_valid_oof_pairs.parquet").filter(pl.col("p").is_not_null() & (pl.col("country") == "India"))
v = v.with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r"))
kept_v = set(v.filter((pl.col("r") == 1) & (pl.col("p") >= 0.75))["m_id"].to_list())
vb = v.filter((pl.col("r") == 1) & (pl.col("p") >= 0.5) & (pl.col("p") < 0.75) & ~pl.col("grp").is_in(["shift_pure", "shift_ms_only"]))
print(f"VALID India (AP/TS) same band: {vb.height} pairs, precision {vb['y'].mean():.3f}; per 1k India VALID S1 {1000 * vb.height / 73301:.1f}")
print("BANDFILL_DONE")
