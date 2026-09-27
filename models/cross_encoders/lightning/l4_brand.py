"""Brand dictionary from brand statements ("<Brand> f/k/a|d/b/a|DBA:|a/k/a|aka|doing business as|formerly <legal name>").
Question: does an entity keep the same synthetic brand across its records? If yes, brand-only records ("Veoevo / 50 GUION PL")
can be tied to the S1 through the statement record of the same entity. Measured on TRAIN labels (no test labels exist).
Also: which of the team's VALID misses (stack_s3a2 p < 0.75 or never scored) a brand link would recover, at what precision."""
import os, re, time
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
SEP = re.compile(r"^(?P<b>.+?)\s*(?:\bf/k/a\b|\bd/b/a\b|\bdba\b\s*:?|\ba/k/a\b|\baka\b|\bdoing business as\b|\bformerly\b)\s*(?P<l>.+)$", re.I)
SUF = {"sys", "co", "inc", "llc", "ltd", "corp", "group", "labs", "lab", "hq", "one", "pro"}
def brand_key(s):
    toks = [re.sub(r"[^a-z]", "", t) for t in s.lower().split()]
    toks = [t for t in toks if t and t not in SUF]
    return toks[0] if toks and len(toks[0]) >= 5 else None
rec = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=["entity_id", "business_name", "business_address"]) for s in (2, 3)])
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
own = dict(zip(gt["m_id"].to_list(), gt["s1_id"].to_list()))
names = dict(zip(rec["entity_id"].to_list(), rec["business_name"].fill_null("").to_list()))
brand_s1 = defaultdict(Counter); stmt = set()
for e, n in names.items():
    m = SEP.match(n)
    if m:
        k = brand_key(m.group("b")); stmt.add(e)
        if k: brand_s1[k][own.get(e)] += 1
log(f"statement records {len(stmt)}, distinct brands {len(brand_s1)}; brands used by >1 S1: {sum(1 for c in brand_s1.values() if len([s for s in c if s]) > 1)}")
# brand-only records: first-word brand key present in the dictionary, record is not itself a statement
hit = tot = amb = none = 0; per_true = []
for e, n in names.items():
    if e in stmt: continue
    k = brand_key(n)
    if not k or k not in brand_s1: continue
    c = brand_s1[k]; s1s = [s for s in c if s]
    tot += 1
    if len(s1s) != 1: amb += 1; continue
    t = own.get(e)
    if t is None: none += 1
    elif t == s1s[0]: hit += 1
    per_true.append((e, s1s[0], t))
log(f"brand-only records with a dictionary brand: {tot}; unambiguous {tot - amb}: correct S1 {hit}, distractor (no S1) {none}, "
    f"wrong S1 {tot - amb - hit - none}; precision {hit / max(tot - amb, 1):.3f}")
# the team's VALID misses
s1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
vs = set(s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list())
sc = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "y", "p")
pred = set((a, b) for a, b, p in zip(sc["s1_id"].to_list(), sc["m_id"].to_list(), sc["p"].to_list()) if p >= 0.75)
cand = [(e, s, t) for e, s, t in per_true if s in vs]
new_true = sum(1 for e, s, t in cand if t == s and (s, e) not in pred)
new_false = sum(1 for e, s, t in cand if t != s and (s, e) not in pred)
log(f"VALID: brand links to VALID S1 {len(cand)}; NOT predicted by the team (p<0.75 or unscored): true {new_true}, false {new_false}")
pl.DataFrame({"m_id": [e for e, s, t in per_true], "s1_id": [s for e, s, t in per_true], "true_s1": [t for e, s, t in per_true]}).write_parquet(f"{W}/brand_links_train.parquet")
print("BRAND_DONE", flush=True)
