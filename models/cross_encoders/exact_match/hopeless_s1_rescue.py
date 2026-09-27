"""Hopeless-S1 rescue: S1 with ZERO kept records; unassigned records that match it with the decoy fingerprint EXCLUDED:
same name_core, same legal-form set (decoys change it 98.9%), identical house-number multiset (decoys change one 99.4%), street
words Jaccard >= .5 (IDF-free). VALID (labels): precision by name strictness; TEST: pocket size by country. Break-even for
an add to an EMPTY S1 is ~59% (a true copy lifts F from 0 to ~0.7-1.0, a false one drops a singleton from 1 to 0)."""
import gzip, io, os, re, unicodedata
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
NUM = re.compile(r"\d+")
LF = {"inc", "llc", "corp", "co", "ltd", "llp", "lp", "pvt", "private", "limited", "company", "corporation", "incorporated", "plc", "pllc", "pc", "opc", "public",
      "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei"}
def fold(s): return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
def lfs(s): return frozenset(x for x in re.findall(r"[a-z0-9]+", fold(s).replace(".", "")) if x in LF)
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
def jac(a, b): return len(a & b) / max(len(a | b), 1)


def run(split, kept_map, label, owner=None, s1_filter=None):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "name_core", "addr_text", "country", "addr_empty"])
                   .with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)]).fill_null("")
    cnt = Counter(kept_map.values())
    s1 = t.filter(pl.col("src") == 1)
    idx = defaultdict(list)
    for e, raw, n, a, c in zip(s1["entity_id"].to_list(), s1["business_name"].to_list(), s1["name_core"].to_list(), s1["addr_text"].to_list(), s1["country"].to_list()):
        if cnt[e] or not n or not nk(a): continue
        if s1_filter is not None and e not in s1_filter: continue
        idx[(c, n, nk(a))].append((e, lfs(raw), street(a)))
    r = t.filter((pl.col("src") != 1) & ~pl.col("addr_empty"))
    st = Counter(); out = []
    for e, raw, n, a, c in zip(r["entity_id"].to_list(), r["business_name"].to_list(), r["name_core"].to_list(), r["addr_text"].to_list(), r["country"].to_list()):
        if e in kept_map or not n: continue
        L = idx.get((c, n, nk(a)))
        if not L: continue
        sa = street(a); lr = lfs(raw)
        L2 = [x for x in L if jac(sa, x[2]) >= 0.5]
        if len(L2) != 1: continue
        x, lx, _ = L2[0]
        lfrel = "lf_same" if lx == lr else ("lf_onemissing" if not lx or not lr else "lf_differ")
        st[(c, lfrel, "n")] += 1
        if owner is not None: st[(c, lfrel, "ok")] += owner.get(e) == x
        if lfrel != "lf_differ": out.append((x, e, c, lfrel))
    print(f"== {label}")
    for c in ("US", "India", "France"):
        for lfrel in ("lf_same", "lf_onemissing", "lf_differ"):
            n_ = st[(c, lfrel, "n")]
            if not n_: continue
            print(f"   {c:6s} {lfrel:14s} proposals {n_:6d}" + (f" precision {st[(c, lfrel, 'ok')] / n_:.3f}" if owner is not None else ""))
    return out


gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
s1t = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
V = set(s1t.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list())
v = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
d = v.sort("p", descending=True).unique("m_id", keep="first").filter(pl.col("p") >= 0.75)
run("train", dict(zip(d["m_id"].to_list(), d["s1_id"].to_list())), "VALID (empty VALID S1, stack_s3a2 decisions)", owner, V)
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
tmap = {m: s for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()) for m in ids.split(",") if m}
out = run("test", tmap, "TEST (empty S1 in stack_e5hyb2)")
pl.DataFrame({"s1_id": [x[0] for x in out], "m_id": [x[1] for x in out], "country": [x[2] for x in out], "lf": [x[3] for x in out]}).write_parquet(f"{W}/hopeless_rescue_test.parquet")
print("HOPELESS_DONE")
