"""Floor-parser victims (S1 side): US S1 whose parsed state is FL only because a floor component ('Fl 3') came last, while the
address names another state code (the true state). Rescue: records not assigned (and not in the exact / DR add lists) with a
near-exact name (same name_core, or name-token Jaccard >= .6) + IDENTICAL house-number multiset (excludes shift decoys) +
street-word Jaccard >= .5, record parsed in the true state or FL; unique best S1. TRAIN precision (labels, all states) for
the same rule on train floor victims; TEST list."""
import gzip, io, os, re
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
NUM = re.compile(r"\d+")
US = set("al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc".split())
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
def jac(a, b): return len(a & b) / max(len(a | b), 1)
def true_state(raw, parsed):
    if parsed != "fl": return None
    codes = [re.sub(r"\d+", "", p).strip().lower() for p in (raw or "").split(",")]
    other = [c for c in codes if c in US and c != "fl"]
    return other[-1] if other else None


def run(split, assigned, label, owner=None, exclude=set()):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_address", "name_core", "addr_text", "country", "addr_state", "addr_empty"])
                   .with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)]).filter(pl.col("country") == "US").fill_null("")
    s1 = t.filter(pl.col("src") == 1)
    vic = {}
    for e, ra, n, a, st in zip(s1["entity_id"].to_list(), s1["business_address"].to_list(), s1["name_core"].to_list(), s1["addr_text"].to_list(), s1["addr_state"].to_list()):
        ts = true_state(ra, st)
        if ts and n and nk(a): vic[e] = (ts, n, nk(a), street(a))
    idx = defaultdict(list)
    for e, (ts, n, k, sw) in vic.items(): idx[k].append(e)
    r = t.filter((pl.col("src") != 1) & ~pl.col("addr_empty"))
    st = Counter(); out = []
    for e, n, a, rs in zip(r["entity_id"].to_list(), r["name_core"].to_list(), r["addr_text"].to_list(), r["addr_state"].to_list()):
        if e in assigned or e in exclude or not n: continue
        k = nk(a)
        L = idx.get(k) if k else None
        if not L: continue
        sa = street(a); A = set(n.split())
        cands = []
        for x in L:
            ts, xn, _, xs = vic[x]
            if rs not in (ts, "fl"): continue
            if jac(sa, xs) < 0.5: continue
            nj = 1.0 if xn == n else jac(A, set(xn.split()))
            if nj >= 0.6: cands.append((nj, x))
        if not cands: continue
        cands.sort(reverse=True)
        if len(cands) > 1 and cands[1][0] == cands[0][0]: continue
        nj, x = cands[0]; cls = "core_eq" if nj == 1.0 else "core_near"
        st[(cls, "n")] += 1
        if owner is not None: st[(cls, "ok")] += owner.get(e) == x
        out.append((x, e, cls))
    print(f"== {label}: floor victims {len(vic)}")
    for cls in ("core_eq", "core_near"):
        n_ = st[(cls, "n")]
        print(f"   {cls:10s} proposals {n_:6d}" + (f" precision {st[(cls, 'ok')] / max(n_, 1):.3f}" if owner is not None else ""))
    return out, vic


gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
run("train", set(), "TRAIN (labels; nothing assigned - precision of the key itself)", owner)
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
ex = set(pl.concat([pl.read_parquet(f"{W}/usin_fix/usin_exact_adds_samestreet.parquet").select("m_id"), pl.read_parquet(f"{W}/dr_test_accepted.parquet").select("m_id")])["m_id"].to_list())
out, vic = run("test", assigned, "TEST (records unassigned in stack_e5hyb2, not in exact/DR adds)", None, ex)
pt = Counter(s for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()) for m in ids.split(",") if m)
ex_s1 = Counter(pl.read_parquet(f"{W}/usin_fix/usin_exact_adds_samestreet.parquet")["s1_id"].to_list())
still_empty = [v for v in vic if pt[v] + ex_s1[v] == 0]
print(f"test floor victims still EMPTY after the exact adds: {len(still_empty)}; of them receiving a rescue pair: {len({x for x, e, c in out} & set(still_empty))}")
pl.DataFrame({"s1_id": [x[0] for x in out], "m_id": [x[1] for x in out], "cls": [x[2] for x in out]}).write_parquet(f"{W}/floor_rescue_test.parquet")
print("FLOORFIX_DONE")
