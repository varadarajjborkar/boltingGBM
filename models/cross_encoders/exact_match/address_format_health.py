"""Address-FORMAT health scan (find the next parser-type bug): mean kept records per S1 (stack_e5hyb2) by S1 address format
feature, vs the country mean. Features from the raw S1 address: unit / suite / floor / '#' / po box / hyphen number (97-12) /
fraction (1/2) / number count / first component is the state or city (reversed order) / has zip-like 5-6 digit / comma count.
Also saves the new hopeless-rescue rider list (not already in the exact or DR lists)."""
import gzip, io, os, re
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
def feats(a, c):
    s = (a or "").lower(); parts = [p.strip() for p in s.split(",") if p.strip()]
    f = []
    if re.search(r"\bunit\b", s): f.append("unit")
    if re.search(r"\b(suite|ste)\b", s): f.append("suite")
    if re.search(r"\b(fl|floor)\b", s): f.append("floor")
    if "#" in s: f.append("hash")
    if re.search(r"\bp\.?\s?o\.?\s?box\b|\bpmb\b", s): f.append("pobox/pmb")
    if re.search(r"\d+-\d+", s): f.append("hyphen_num")
    if re.search(r"\d+/\d+", s): f.append("fraction")
    n = len(re.findall(r"\d+", s)); f.append(f"nums={min(n, 4)}")
    if parts and not re.search(r"\d", parts[0]) and len(parts) >= 3: f.append("first_comp_no_digit")
    if re.search(r"\b\d{5,6}\b", s): f.append("zip_like")
    f.append(f"commas={min(s.count(','), 5)}")
    return f
te = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "business_address"]).fill_null("")
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
g = defaultdict(list); tot = defaultdict(list)
for e, c, a in zip(te["entity_id"].to_list(), te["country"].to_list(), te["business_address"].to_list()):
    k = pt.get(e, 0); tot[c].append(k)
    for f in feats(a, c): g[(c, f)].append(k)
for c in ("US", "India", "France"):
    base = sum(tot[c]) / len(tot[c])
    rows = [(sum(v) / len(v) - base, f, len(v), sum(1 for x in v if x == 1) / len(v), sum(1 for x in v if x == 0) / len(v)) for (cc, f), v in g.items() if cc == c and len(v) >= 2000]
    print(f"== {c}: mean {base:.3f}; formats with >= 2,000 S1, most under-predicting first")
    for d, f, n, one, zero in sorted(rows)[:9]:
        print(f"   {f:22s} S1 {n:7d} mean {base + d:.3f} ({d:+.3f}) 1-share {one:.3f} empty {zero:.3f}")
# rider list: hopeless rescue proposals not in exact / DR lists
h = pl.read_parquet(f"{W}/hopeless_rescue_test.parquet")
ex = pl.concat([pl.read_parquet(f"{W}/usin_fix/usin_exact_adds_samestreet.parquet").select("s1_id", "m_id"), pl.read_parquet(f"{W}/france_fix/fr_exact_adds.parquet").select("s1_id", "m_id"),
                pl.read_parquet(f"{W}/dr_test_accepted.parquet").select("s1_id", "m_id")])
new = h.join(ex, on=["s1_id", "m_id"], how="anti"); new.write_parquet(f"{W}/hopeless_rescue_new.parquet")
print("hopeless rescue NEW rider pairs:", new.height, new.group_by("country").len().to_dicts())
print("FMTSCAN_DONE")
