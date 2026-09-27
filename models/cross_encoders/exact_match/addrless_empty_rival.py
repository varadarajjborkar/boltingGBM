"""Address-less 2-way ties with one empty competitor. For an address-less record r whose name_core is shared by exactly 2 S1 of
the country, where one S1 has 0 kept (predicted) records and the other >= 1: propose the non-empty S1.
VALID (labels): restricted to ties whose BOTH S1 are VALID S1 (so both have predicted counts from the stack), records the
stack left unassigned -> precision. TEST (stack_e5hyb2): pocket size by country among unassigned address-less records."""
import gzip, io, os
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")


def run(split, kept_map, label, owner=None, s1_filter=None):
    s1 = pl.read_parquet(f"{W}/norm_v2/{split}_source1.parquet", columns=["entity_id", "name_core", "country"]).fill_null("")
    ties = defaultdict(list)
    for e, n, c in zip(s1["entity_id"].to_list(), s1["name_core"].to_list(), s1["country"].to_list()):
        if n: ties[(c, n)].append(e)
    r = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{k}.parquet", columns=["entity_id", "name_core", "country", "addr_empty"]) for k in (2, 3)]).fill_null("")
    r = r.filter(pl.col("addr_empty"))
    cnt = Counter(kept_map.values())
    st = Counter(); out = []
    for e, n, c in zip(r["entity_id"].to_list(), r["name_core"].to_list(), r["country"].to_list()):
        if e in kept_map or not n: continue
        L = ties.get((c, n), [])
        if len(L) != 2: continue
        if s1_filter is not None and not all(x in s1_filter for x in L): continue
        a, b = L; ca, cb = cnt[a], cnt[b]
        if (ca == 0) == (cb == 0): continue
        x = a if ca > 0 else b
        st[(c, "n")] += 1
        if owner is not None: st[(c, "ok")] += owner.get(e) == x; st[(c, "owned")] += e in owner
        out.append((x, e, c))
    print(f"== {label}")
    for c in ("US", "India", "France"):
        if st[(c, "n")] == 0: continue
        line = f"   {c:6s} proposals {st[(c, 'n')]}"
        if owner is not None: line += f" precision {st[(c, 'ok')] / st[(c, 'n')]:.3f} (record owned by any S1 {st[(c, 'owned')] / st[(c, 'n')]:.3f})"
        print(line)
    return out


gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
s1t = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
V = set(s1t.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list())
v = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
d = v.sort("p", descending=True).unique("m_id", keep="first").filter(pl.col("p") >= 0.75)
run("train", dict(zip(d["m_id"].to_list(), d["s1_id"].to_list())), "VALID (both tied S1 in VALID, stack_s3a2 counts)", owner, V)
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
tmap = {m: s for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()) for m in ids.split(",") if m}
out = run("test", tmap, "TEST (stack_e5hyb2 counts)")
pl.DataFrame({"s1_id": [x[0] for x in out], "m_id": [x[1] for x in out], "country": [x[2] for x in out]}).write_parquet(f"{W}/addrless_2tie_empty_rival.parquet")
print("HUNGRY2_DONE")
