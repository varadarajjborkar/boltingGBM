"""France fix list on top of stack_e5hyb2 (applies to any file whose France rows = stack_ce, e.g. stack_pkf_a1_fr).
ADD  (record unassigned):  unique S1 B with same name_core + numbers + street, raw body equal (accent-folded, legal forms
     removed) and legal forms equal or one side without one;  OR  collision group with a unique same-body same-legal-form member.
MOVE (record assigned to A): same conditions for B, and A's raw body differs from r's while B's is equal (identical-name S1 exists).
Safety diagnostics: (1) VALID: records that are exact raw duplicates (same source, same raw name+address) of an owned record -
same owner? (2) France pocket: B already has an assigned record from r's source with identical raw name+address?"""
import gzip, io, os, re, unicodedata, json
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/france_fix"; os.makedirs(OUT, exist_ok=True)
NUM = re.compile(r"\d+")
LF = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scp", "selarl", "scop", "gie", "earl", "gaec", "sca", "scs", "selas", "sem", "ei", "eirl",
      "inc", "llc", "corp", "co", "ltd", "llp", "lp", "pvt", "private", "limited", "company", "corporation", "incorporated", "plc"}
def fold(s): return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
def toks(s): return re.findall(r"[a-z0-9]+", fold(s).replace(".", ""))
def body(s): return tuple(sorted(x for x in toks(s) if x not in LF))
def lfs(s): return frozenset(x for x in toks(s) if x in LF)
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
def jac(a, b): return len(a & b) / max(len(a | b), 1)
def tables(split):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "business_address", "name_core", "addr_text", "country"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])
    return t.with_columns(pl.col("name_core").fill_null(""), pl.col("addr_text").fill_null(""), pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
# ---- (1) VALID duplicate diagnostic
t = tables("train")
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
r = t.filter(pl.col("src") != 1)
dup = defaultdict(list)
for e, s, n, a in zip(r["entity_id"].to_list(), r["src"].to_list(), r["business_name"].to_list(), r["business_address"].to_list()):
    if a: dup[(s, n.strip().lower(), a.strip().lower())].append(e)
dd = Counter()
for L in dup.values():
    if len(L) < 2: continue
    os_ = [owner.get(x) for x in L]; dd["groups"] += 1
    if all(o is not None for o in os_): dd["all_owned"] += 1; dd["same_owner"] += len(set(os_)) == 1
    elif any(o is not None for o in os_): dd["some_unowned"] += 1
    else: dd["none_owned"] += 1
print(f"TRAIN exact raw duplicates within a source: {dd['groups']} groups; all owned {dd['all_owned']} (same owner {dd['same_owner']}), some unowned {dd['some_unowned']}, none owned {dd['none_owned']}")
del t, r, dup
# ---- test France
t = tables("test")
RAW = dict(zip(t["entity_id"].to_list(), t["business_name"].to_list())); RA = dict(zip(t["entity_id"].to_list(), t["business_address"].to_list()))
SRC = dict(zip(t["entity_id"].to_list(), t["src"].to_list()))
s1 = t.filter((pl.col("src") == 1) & (pl.col("country") == "France"))
g = defaultdict(list)
for e, n, a, raw in zip(s1["entity_id"].to_list(), s1["name_core"].to_list(), s1["addr_text"].to_list(), s1["business_name"].to_list()):
    if n and nk(a): g[(n, nk(a))].append((e, street(a), body(raw), lfs(raw)))
nF = s1.height
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
pred = {m: s for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()) for m in ids.split(",") if m}
members = defaultdict(list)
for m, s in pred.items(): members[s].append(m)
rf = t.filter((pl.col("src") != 1) & (pl.col("country") == "France"))
adds, moves = [], []; st = Counter()
for e, n, a, raw in zip(rf["entity_id"].to_list(), rf["name_core"].to_list(), rf["addr_text"].to_list(), rf["business_name"].to_list()):
    k = nk(a)
    if not n or not k: continue
    L = g.get((n, k))
    if not L: continue
    sa = street(a); L = [x for x in L if sa and jac(sa, x[1]) >= 0.5]
    if not L: continue
    b_, f_ = body(raw), lfs(raw)
    if len(L) == 1:
        B, _, bb, bf = L[0]
        ok = bb == b_ and (bf == f_ or not bf or not f_); why = "unique"
    else:
        M = [x for x in L if x[2] == b_ and x[3] == f_]
        ok = len(M) == 1; B = M[0][0] if ok else None; why = "group"
    if not ok: continue
    A = pred.get(e)
    if A == B: continue
    same_src_dup = any(SRC[x] == SRC[e] and RAW[x].strip().lower() == raw.strip().lower() and RA[x].strip().lower() == RA[e].strip().lower() for x in members.get(B, []))
    if A is None:
        adds.append((B, e, why, same_src_dup)); st[("add", why)] += 1; st[("add_dup", why)] += same_src_dup
    elif body(RAW[A]) != b_ or (why == "group" and lfs(RAW[A]) != f_):
        moves.append((A, B, e, why, same_src_dup)); st[("move", why)] += 1; st[("move_dup", why)] += same_src_dup
for kind in ("add", "move"):
    for why in ("unique", "group"):
        x = st[(kind, why)]; print(f"France {kind:4s} {why:6s}: {x:5d} ({1000 * x / nF:.2f}/1k S1); B already holds an identical same-source record: {st[(kind + '_dup', why)] / max(x, 1):.3f}")
pl.DataFrame({"s1_id": [x[0] for x in adds], "m_id": [x[1] for x in adds], "why": [x[2] for x in adds], "dup_of_assigned": [x[3] for x in adds]}).write_parquet(f"{OUT}/fr_exact_adds.parquet")
pl.DataFrame({"from_s1_id": [x[0] for x in moves], "s1_id": [x[1] for x in moves], "m_id": [x[2] for x in moves], "why": [x[3] for x in moves], "dup_of_assigned": [x[4] for x in moves]}).write_parquet(f"{OUT}/fr_exact_moves.parquet")
print(f"written {len(adds)} adds, {len(moves)} moves; French S1 {nF}")
print("FRFIX_DONE")
