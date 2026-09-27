"""Diagnose US city gaps (Saint Louis MO, Columbus OH, Raleigh NC, Washington DC): for S1 of the city, show normalised addr_text
vs raw; count UNASSIGNED records (stack_e5hyb2) with the same name_core anywhere in the country, by the record's parsed state,
and how many share a house number with the S1. Examples of S1 with 0-1 kept records and their likely missing copies."""
import gzip, io, os, re, random
from collections import Counter, defaultdict
import polars as pl
random.seed(1)
W = os.environ.get("ER_WORK_DIR", "work")
NUM = re.compile(r"\d+")
t = pl.concat([pl.read_parquet(f"{W}/norm_v2/test_source{s}.parquet", columns=["entity_id", "business_name", "business_address", "name_core", "addr_text", "addr_state", "country"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])
t = t.filter(pl.col("country") == "US").fill_null("")
s1 = t.filter(pl.col("src") == 1); rec = t.filter(pl.col("src") != 1)
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
byname = defaultdict(list)
for e, n, a, st, raw, ra in zip(rec["entity_id"].to_list(), rec["name_core"].to_list(), rec["addr_text"].to_list(), rec["addr_state"].to_list(), rec["business_name"].to_list(), rec["business_address"].to_list()):
    if n and e not in assigned: byname[n].append((e, a, st, raw, ra))
for label, pat, state in (("Saint Louis MO", r"s(ain)?t\.?\s+louis", "mo"), ("Columbus OH", r"columbus", "oh"), ("Raleigh NC", r"raleigh", "nc"), ("Washington DC", r"washington", "dc")):
    sub = s1.filter(pl.col("business_address").str.to_lowercase().str.contains(pat) & (pl.col("addr_state") == state))
    st = Counter(); ex = []
    for e, n, a, raw, ra in zip(sub["entity_id"].to_list(), sub["name_core"].to_list(), sub["addr_text"].to_list(), sub["business_name"].to_list(), sub["business_address"].to_list()):
        k = pt.get(e, 0); st["S1"] += 1; st["kept"] += k
        cands = byname.get(n, [])
        na = set(NUM.findall(a))
        same_num = [c for c in cands if na & set(NUM.findall(c[1]))]
        st["unassigned_same_name"] += len(cands); st["unassigned_same_name_same_num"] += len(same_num)
        for c in same_num: st[("rec_state", c[2])] += 1
        if k <= 1 and same_num and len(ex) < 4: ex.append((raw, ra[:60], a[:60], [(c[3], c[4][:60], c[1][:60], c[2]) for c in same_num[:2]]))
    print(f"== {label}: S1 {st['S1']}, mean kept {st['kept'] / max(st['S1'], 1):.3f}; unassigned same-name records {st['unassigned_same_name']}, of which share a number {st['unassigned_same_name_same_num']}")
    print("   parsed state of those records: " + ", ".join(f"{k[1] or '(none)'} {v}" for k, v in sorted(((k, v) for k, v in st.items() if isinstance(k, tuple)), key=lambda x: -x[1])[:6]))
    for r_, ra_, a_, cs in ex:
        print(f"   S1 {r_!r} [{ra_}] -> addr_text [{a_}]")
        for c in cs: print(f"        missing? {c[0]!r} [{c[1]}] -> [{c[2]}] state={c[3] or '(none)'}")
print("CITYGAP_DONE")
