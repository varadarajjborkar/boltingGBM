"""US city-level recall scan (the MH/DL method). City = the comma component next to the state component of the S1's raw
address ('3015 Sheridan Drive, Amherst, NY' -> amherst; 'NY, VALLEY STREAM, 10 PICARD ROAD' -> valley stream). Per (state,
city) with >= 1,500 S1: mean kept records per S1 (stack_e5hyb2), 1-record share, empty share; deficit vs the state's median
city. Cities >= 0.04 below their state are candidate blocking gaps (Mumbai was -0.072)."""
import gzip, io, os, re
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
US = {"al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
      "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"}
def city(raw):
    parts = [re.sub(r"[^a-z ]", "", p.strip().lower()).strip() for p in (raw or "").split(",")]
    parts = [p for p in parts if p]
    idx = [i for i, p in enumerate(parts) if p in US or re.sub(r"\s+\d.*", "", p) in US]
    if not idx: return ""
    i = idx[-1]
    cand = parts[i - 1] if i > 0 else (parts[i + 1] if i + 1 < len(parts) else "")
    return cand if cand and not re.search(r"\d", cand) and len(cand) > 2 else ""
te = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state", "business_address"]).filter(pl.col("country") == "US").fill_null("")
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
g = defaultdict(list); parsed = 0
for e, st, a in zip(te["entity_id"].to_list(), te["addr_state"].to_list(), te["business_address"].to_list()):
    c = city(a)
    if c: parsed += 1; g[(st, c)].append(pt.get(e, 0))
print(f"US S1 {te.height}, city parsed for {parsed}")
bystate = defaultdict(list)
for (st, c), v in g.items():
    if len(v) >= 1500: bystate[st].append((c, sum(v) / len(v), len(v), sum(1 for x in v if x == 1) / len(v), sum(1 for x in v if x == 0) / len(v)))
rows = []
for st, L in bystate.items():
    med = sorted(x[1] for x in L)[len(L) // 2]
    for c, m, n, one, zero in L: rows.append((m - med, st, c, m, n, one, zero, med))
print(f"{len(rows)} (state, city) groups with >= 1,500 S1; most under-predicting:")
for d, st, c, m, n, one, zero, med in sorted(rows)[:15]:
    print(f"   {st} {c[:22]:22s} S1 {n:6d} mean {m:.3f} (state median {med:.3f}, {d:+.3f}) 1-share {one:.3f} empty {zero:.3f}")
print("USCITY_DONE")
