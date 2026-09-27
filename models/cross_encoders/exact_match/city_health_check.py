"""City-level health check (the MH/DL method, finer): per city (most frequent non-state word pair of the S1 address, >= 3,000
S1) predicted records per S1 and 1-record share, vs the country's median city; cities under-predicting by >= 0.03 records per
S1 are candidate blocking gaps. Also how many India DR adds land there (fill check)."""
import gzip, io, os, re
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
te = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state", "business_address"]).fill_null("")
def city(raw):
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    parts = [re.sub(r"[^a-z ]", "", p).strip() for p in parts]
    parts = [p for p in parts if p and len(p) > 2 and not re.fullmatch(r"(no|plot|flat|floor|road|street|st|ave|avenue|suite|ste)", p)]
    return parts[-2] if len(parts) >= 2 else (parts[-1] if parts else "")
CITY = {e: city(a) for e, a in zip(te["entity_id"].to_list(), te["business_address"].to_list())}
CT = dict(zip(te["entity_id"].to_list(), te["country"].to_list()))
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
dr = pl.read_parquet(f"{W}/dr_test_accepted.parquet"); dr = dr.filter(~pl.col("m_id").is_in(list(assigned)))
drc = Counter(CITY.get(s, "") for s in dr["s1_id"].to_list())
g = defaultdict(list)
for s in te["entity_id"].to_list(): g[(CT[s], CITY[s])].append(pt.get(s, 0))
for c in ("US", "India", "France"):
    rows = [(k[1], sum(v) / len(v), sum(1 for x in v if x == 1) / len(v), sum(1 for x in v if x == 0) / len(v), len(v)) for k, v in g.items() if k[0] == c and len(v) >= 3000 and k[1]]
    if not rows: continue
    med = sorted(r[1] for r in rows)[len(rows) // 2]
    print(f"== {c}: {len(rows)} cities with >= 3,000 S1, median mean {med:.3f}")
    for name, mean, one, zero, n in sorted(rows, key=lambda r: r[1])[:10]:
        print(f"   {name[:24]:24s} S1 {n:6d} mean {mean:.3f} ({mean - med:+.3f}) 1-share {one:.3f} 0-share {zero:.3f} | DR adds {drc.get(name, 0)} ({1000 * drc.get(name, 0) / n:.1f}/1k)")
print("CITY_DONE")
