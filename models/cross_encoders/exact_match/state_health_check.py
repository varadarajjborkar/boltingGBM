"""Per-state health check of the uploaded predictions against the generator's records-per-S1 distribution.
Truth (VALID labels): share of S1 with 0 / 1 / 2 / 3 / 4 / 5 / 6+ true records and mean. Prediction on VALID for reference.
Test (stack_e5hyb2 file): same shares per state for US / India states with >= 15k S1. Big departures (too few zeros = FPs on
singletons, too many zeros/ones = misses) flag states where the pipeline behaves differently from VALID."""
import gzip, io, os
from collections import Counter
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
def dist(counts):
    n = len(counts); c = Counter(min(x, 6) for x in counts)
    return [c[i] / n for i in range(7)] + [sum(counts) / n]
s1t = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
nt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list())}
v = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
d = v.sort("p", descending=True).unique("m_id", keep="first").filter(pl.col("p") >= 0.75)
pv = Counter(d["s1_id"].to_list())
hdr = f"{'':22s} {'n S1':>7s} " + " ".join(f"{k:>6s}" for k in ("0", "1", "2", "3", "4", "5", "6+", "mean"))
print(hdr)
for c, sts in (("US", ["ny"]), ("India", ["ap", "ts"])):
    ids = s1t.filter((pl.col("country") == c) & pl.col("addr_state").is_in(sts))["entity_id"].to_list()
    for lab, cnt in (("truth", [nt.get(s, 0) for s in ids]), ("VALID pred", [pv[s] for s in ids])):
        print(f"{c + ' ' + lab:22s} {len(ids):7d} " + " ".join(f"{x:6.3f}" for x in dist(cnt)))
# all TRAIN states' truth (to see whether the distribution varies by state)
for c in ("US", "India"):
    tt = s1t.filter(pl.col("country") == c)
    for st, grp in tt.group_by("addr_state"):
        ids = grp["entity_id"].to_list()
        if len(ids) < 40000: continue
        print(f"{c + ' TRAIN ' + str(st[0]) + ' truth':22s} {len(ids):7d} " + " ".join(f"{x:6.3f}" for x in dist([nt.get(s, 0) for s in ids])))
te = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state"])
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
for c in ("US", "India", "France"):
    tt = te.filter(pl.col("country") == c)
    rows = []
    for st, grp in tt.group_by("addr_state"):
        ids = grp["entity_id"].to_list()
        if len(ids) >= 15000: rows.append((str(st[0]), ids))
    for st, ids in sorted(rows, key=lambda x: -len(x[1])):
        print(f"{c + ' TEST ' + st:22s} {len(ids):7d} " + " ".join(f"{x:6.3f}" for x in dist([pt.get(s, 0) for s in ids])))
print("STATEDIST_DONE")
