"""Do the dense-retrieval India adds fill the Maharashtra / Delhi record deficit?
For India test states: current kept records per S1 (stack_e5hyb2) and after adding dr_test_accepted pairs (records not yet
assigned). A real missing copy lands on an S1 that is short (fewer kept records than typical); a distractor lands like a
random S1 (incl. zero-record S1, which are ~95% true singletons). Compare: distribution of the receiving S1's current count
vs all S1 of the state, and the state mean / 0-share / 1-share before vs after, against the truth (3.465, 0.055, 0.054).
VALID analogue: DR VALID adds (dr_valid_new_pairs) with labels -> precision by receiving-S1 count."""
import gzip, io, os
from collections import Counter
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
def dist(counts):
    n = len(counts); c = Counter(min(x, 6) for x in counts)
    return [c[i] / n for i in range(7)] + [sum(counts) / n]
te = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state"]).filter(pl.col("country") == "India")
ST = dict(zip(te["entity_id"].to_list(), te["addr_state"].to_list()))
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pt = {s: len([m for m in ids.split(",") if m]) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list())}
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
dr = pl.read_parquet(f"{W}/dr_test_accepted.parquet").filter(pl.col("country") == "India")
dr = dr.filter(~pl.col("m_id").is_in(list(assigned)))
recv = Counter(dr["s1_id"].to_list())
print(f"India DR adds on unassigned records: {dr.height}")
print(f"{'':26s} {'n':>7s} " + " ".join(f"{k:>6s}" for k in ("0", "1", "2", "3", "4", "5", "6+", "mean")))
for st in ("mh", "dl", "up", "ka", "tn", "ts", "ap"):
    ids = [s for s, x in ST.items() if x == st]
    before = [pt.get(s, 0) for s in ids]; after = [pt.get(s, 0) + recv.get(s, 0) for s in ids]
    rc = [pt.get(s, 0) for s in ids if recv.get(s, 0)]
    print(f"{st + ' before':26s} {len(ids):7d} " + " ".join(f"{x:6.3f}" for x in dist(before)))
    print(f"{st + ' after DR adds':26s} {len(ids):7d} " + " ".join(f"{x:6.3f}" for x in dist(after)))
    if rc: print(f"{st + ' receiving S1 (current)':26s} {len(rc):7d} " + " ".join(f"{x:6.3f}" for x in dist(rc)))
# VALID analogue with labels
vp = f"{W}/dr_valid_new_pairs.parquet"
if os.path.exists(vp):
    dv = pl.read_parquet(vp); print("VALID DR new pairs columns:", dv.columns, dv.height)
    v = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
    d = v.sort("p", descending=True).unique("m_id", keep="first").filter(pl.col("p") >= 0.75)
    pv = Counter(d["s1_id"].to_list())
    gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
    owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
    ycol = "y" if "y" in dv.columns else None
    st = Counter()
    for s, m in zip(dv["s1_id"].to_list(), dv["m_id"].to_list()):
        k = min(pv[s], 4); st[(k, "n")] += 1; st[(k, "ok")] += owner.get(m) == s
    print("VALID DR new pairs: precision by the receiving S1's current kept count: " + ", ".join(f"{k}{'+' if k == 4 else ''}: {st[(k, 'ok')] / max(st[(k, 'n')], 1):.2f} (n={st[(k, 'n')]})" for k in range(5)))
print("DRFILL_DONE")
