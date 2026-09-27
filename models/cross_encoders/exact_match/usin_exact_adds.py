"""US/India exact-match adds for records the test blocking never compared (same rule as the France fix, unique case only).
Record unassigned in stack_e5hyb2; exactly one S1 of the country with the same name_core + house-number multiset (+ street
Jaccard >= .5); raw body equal after accent folding and legal-form removal; legal forms equal or one side without one;
the record adds no decoy-vocabulary word over the S1's raw name. VALID: same rule restricted to records the stack left
unassigned -> precision (owner == S1) and count; plus precision of the unique key on all VALID records."""
import gzip, io, json, os, re, unicodedata
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/usin_fix"; os.makedirs(OUT, exist_ok=True)
NUM = re.compile(r"\d+")
LF = {"inc", "llc", "corp", "co", "ltd", "llp", "lp", "pvt", "private", "limited", "company", "corporation", "incorporated", "plc", "pllc", "pc", "opc"}
VOC = set(json.load(open(f"{W}/e5hyb2/decoy_vocab_validated.json"))) if os.path.exists(f"{W}/e5hyb2/decoy_vocab_validated.json") else set()


def fold(s): return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
def toks(s): return re.findall(r"[a-z0-9]+", fold(s).replace(".", ""))
def body(s): return tuple(sorted(x for x in toks(s) if x not in LF))
def lfs(s): return frozenset(x for x in toks(s) if x in LF)
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
def jac(a, b): return len(a & b) / max(len(a | b), 1)


def tables(split):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "name_core", "addr_text", "country", "addr_state"])
                   .with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])
    return t.with_columns(pl.col("name_core").fill_null(""), pl.col("addr_text").fill_null(""), pl.col("business_name").fill_null(""))


def pocket(t, assigned, countries, s1ok=None):
    s1 = t.filter((pl.col("src") == 1) & pl.col("country").is_in(countries))
    g = defaultdict(list)
    for e, n, a, c, raw in zip(s1["entity_id"].to_list(), s1["name_core"].to_list(), s1["addr_text"].to_list(), s1["country"].to_list(), s1["business_name"].to_list()):
        if n and nk(a) and (s1ok is None or e in s1ok):
            g[(c, n, nk(a))].append((e, street(a), body(raw), lfs(raw), set(toks(raw))))
    r = t.filter((pl.col("src") != 1) & pl.col("country").is_in(countries))
    out = []; st = Counter()
    for e, n, a, c, raw in zip(r["entity_id"].to_list(), r["name_core"].to_list(), r["addr_text"].to_list(), r["country"].to_list(), r["business_name"].to_list()):
        if e in assigned or not n:
            continue
        k = nk(a)
        if not k:
            continue
        L = g.get((c, n, k))
        if not L or len(L) != 1:
            continue
        B, sb, bb, bf, bt = L[0]
        sa = street(a)
        if not sa or jac(sa, sb) < 0.5:
            st[(c, "street_fail")] += 1
            continue
        if bb != body(raw) or not (bf == lfs(raw) or not bf or not lfs(raw)):
            st[(c, "name_fail")] += 1
            continue
        if (set(toks(raw)) - bt) & VOC:
            st[(c, "decoy_word")] += 1
            continue
        out.append((B, e, c))
    return out, st


# ---- VALID: records the stack left unassigned
t = tables("train")
s1t = t.filter(pl.col("src") == 1)
V = set(s1t.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list())
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
v = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
d = v.sort("p", descending=True).unique("m_id", keep="first").filter(pl.col("p") >= 0.75)
assigned = set(d["m_id"].to_list())
vo, vst = pocket(t, assigned, ["US", "India"], V)
ok = sum(owner.get(m) == b for b, m, c in vo)
print(f"VALID: rule adds {len(vo)} records ({1000 * len(vo) / len(V):.2f}/1k S1) to stack-unassigned records; precision {ok / max(len(vo), 1):.3f}; filtered: {dict(vst)}")
del t
# ---- TEST
t = tables("test")
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
assigned = {m for ids in mr["matched_entity_ids"].to_list() for m in ids.split(",") if m}
n1 = Counter(t.filter(pl.col("src") == 1)["country"].to_list())
to, tst = pocket(t, assigned, ["US", "India"])
cand = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_candidate_pairs.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
C = {(s, m) for s, ids in zip(cand[cand.columns[0]].to_list(), cand[cand.columns[1]].to_list()) for m in ids.split(",") if m}
for c in ("US", "India"):
    x = [(b, m) for b, m, cc in to if cc == c]
    print(f"TEST {c}: {len(x)} adds ({1000 * len(x) / n1[c]:.2f}/1k S1), in candidate set {sum(p in C for p in x) / max(len(x), 1):.2f}; filtered: " + str({k[1]: v for k, v in tst.items() if k[0] == c}))
pl.DataFrame({"s1_id": [b for b, m, c in to], "m_id": [m for b, m, c in to], "country": [c for b, m, c in to]}).write_parquet(f"{OUT}/usin_exact_adds.parquet")
print("USIN_DONE")
