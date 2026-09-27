"""Locality-alias features (new: the pipeline has no city/locality feature). Generator swaps a suburb for its metro city
(Lackawanna -> Buffalo, Irondequoit -> Rochester). Learn locality co-occurrence from TRUE training pairs and score each pair:
loc_exact (shared locality component), loc_alias (max learned P(record locality | S1 locality) over differing pairs),
loc_conflict (record names a locality that is neither equal nor a known alias), rec_has_loc.
Fair VALID: alias tables for VALID S1 of half h are learned from non-VALID states + VALID half (1-h) only (s1_id.hash(13)%2).
Test: tables from ALL training true pairs. Residual on the team's stack_s3a2 VALID p + e5 v1 (2 folds by the same hash)."""
import json, os, re, time
from collections import Counter, defaultdict
import numpy as np, polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/out_loc"; os.makedirs(OUT, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
US = {"al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv",
      "nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy","dc"}
STATE_WORDS = {"alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia","hawaii","idaho","illinois",
  "indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi","missouri","montana","nebraska",
  "nevada","new hampshire","new jersey","new mexico","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
  "south carolina","south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia","wisconsin","wyoming",
  "telangana","andhra pradesh","tg","ap","ts","maharashtra","mh","karnataka","ka","tamil nadu","tn","delhi","dl","uttar pradesh","up","gujarat","gj",
  "west bengal","wb","kerala","kl","rajasthan","rj","haryana","hr","bihar","br","punjab","pb","madhya pradesh","mp","odisha","or","assam","as",
  "hauts-de-france","pays de la loire","nouvelle-aquitaine","france","india","usa","united states"}
STREETY = re.compile(r"\b(road|rd|street|st|avenue|ave|av|drive|dr|lane|ln|court|ct|boulevard|blvd|way|place|pl|circle|cir|highway|hwy|parkway|pkwy|"
                     r"trail|trl|terrace|ter|rue|allee|chemin|impasse|cours|quai|route|floor|flat|plot|door|house|no|apartment|apt|unit|suite|bldg|room|po box)\b")
def localities(addr):
    out = set()
    for c in (addr or "").lower().split(","):
        c = re.sub(r"[^a-zÀ-ɏ\s-]", " ", c.replace(".", " ")).strip(); c = re.sub(r"\s+", " ", c)
        if not c or len(c) < 3 or any(ch.isdigit() for ch in c) or c in US or c in STATE_WORDS or STREETY.search(c):
            continue
        out.add(c)
    if len(out) > 1:
        out.discard("new york")
    return out
t0 = time.time()
tr1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
rp = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=["entity_id", "business_name", "name_non_latin", "addr_empty", "name_is_domain"]) for s in (2, 3)])
rp = rp.select(pl.col("entity_id").alias("m_id"), pl.col("entity_id").str.starts_with("S3").cast(pl.Float64).alias("rec_s3"),
               pl.col("name_non_latin").cast(pl.Float64).fill_null(0).alias("rec_native"), pl.col("addr_empty").cast(pl.Float64).fill_null(0).alias("rec_noaddr"),
               pl.col("name_is_domain").cast(pl.Float64).fill_null(0).alias("rec_domain"), pl.col("business_name").fill_null("").str.len_chars().cast(pl.Float64).alias("rec_nlen"))
sc = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "y", "p")
new = pl.read_parquet(f"{W}/dr_valid_new_pairs.parquet")
assigned = set(sc.filter(pl.col("p") >= 0.725)["m_id"].to_list())
import lightgbm as lgb
s1v = tr1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"]))).select("entity_id", "country")
gv = gt.join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
ntrue = gv.group_by("s1_id").len().rename({"len": "nt"}); gtp = gv.with_columns(pl.lit(1).alias("t"))
def macro(scx):
    sc1 = scx.sort("p", descending=True).unique("m_id", keep="first"); best = None
    for t in np.arange(0.4, 0.96, 0.025).round(3):
        pr = sc1.filter(pl.col("p") >= t).join(gtp, on=["s1_id", "m_id"], how="left").with_columns(pl.col("t").fill_null(0))
        agg = pr.group_by("s1_id").agg(pl.len().alias("np"), pl.col("t").sum().alias("tp"))
        u = (s1v.rename({"entity_id": "s1_id"}).join(agg, on="s1_id", how="left").join(ntrue, on="s1_id", how="left").fill_null(0)
               .with_columns(pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
                             .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))).alias("f"),
                             (pl.col("s1_id").hash(13) % 2).alias("h")))
        r = (float(t), u["f"].mean(), u.filter(pl.col("h") == 0)["f"].mean(), u.filter(pl.col("h") == 1)["f"].mean(),
             u.filter(pl.col("country") == "US")["f"].mean(), u.filter(pl.col("country") == "India")["f"].mean())
        if best is None or r[1] > best[1]: best = r
    return dict(zip(["t", "macro", "h0", "h1", "US", "India"], best))
def oof(X, yv, grp):
    p = np.zeros(len(yv)); prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8, seed=7)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[grp != k], yv[grp != k]), 300); p[grp == k] = m.predict(X[grp == k])
    return p
base = macro(sc.select("s1_id", "m_id", pl.col("p").cast(pl.Float64)))
log(f"base {json.dumps({k: round(float(v), 5) for k, v in base.items()})}")
un = new.filter(~pl.col("m_id").is_in(list(assigned)))
log(f"new pairs from unassigned records: {un.height} ({un.height / 175615:.2f} per S1); records {un['m_id'].n_unique()}")
un = un.with_columns(pl.col("cos").rank("ordinal", descending=True).over("m_id").alias("r"))
import re
t = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{k}.parquet", columns=["entity_id", "name_core", "addr_text"]) for k in (1, 2, 3)])
NC = dict(zip(t["entity_id"].to_list(), t["name_core"].fill_null("").to_list())); AT = dict(zip(t["entity_id"].to_list(), t["addr_text"].fill_null("").to_list()))
NUM = re.compile(r"\d+")
GEN = {"private", "limited", "pvt", "ltd", "services", "service", "center", "centre", "group", "india", "enterprises", "inc", "llc", "corp", "company", "the", "and", "of"}
def share(a, b):
    na = set(NUM.findall(AT.get(a, ""))); nb = set(NUM.findall(AT.get(b, "")))
    wa = {w for w in NC.get(a, "").split() if len(w) >= 4 and w not in GEN}; wb = {w for w in NC.get(b, "").split() if len(w) >= 4 and w not in GEN}
    return (bool(na & nb), bool(wa & wb))
cand = un.filter(pl.col("r") <= 2)
sh = [share(a, b) for a, b in zip(cand["s1_id"].to_list(), cand["m_id"].to_list())]
cand = cand.with_columns(pl.Series("snum", [x[0] for x in sh]), pl.Series("sword", [x[1] for x in sh]))
for name, cond in (("share number", pl.col("snum")), ("share word", pl.col("sword")), ("number AND word", pl.col("snum") & pl.col("sword")),
                   ("number OR word", pl.col("snum") | pl.col("sword")), ("top1 & (number OR word)", (pl.col("r") == 1) & (pl.col("snum") | pl.col("sword")))):
    f = cand.filter(cond); add = f.filter(pl.col("ce") >= 0.998)
    r = macro(pl.concat([sc.select("s1_id", "m_id", pl.col("p").cast(pl.Float64)), add.select("s1_id", "m_id", pl.col("ce").cast(pl.Float64).alias("p"))]))
    log(f"{name:24s}: judged {f.height} ({f.height / 175615:.3f} per S1), added {add.height} prec {add['y'].mean() if add.height else 0:.3f} -> "
        f"{r['macro'] - base['macro']:+.5f} | halves {r['h0'] - base['h0']:+.5f} {r['h1'] - base['h1']:+.5f} | US {r['US'] - base['US']:+.5f} India {r['India'] - base['India']:+.5f}")
print("DREVAL3_DONE", flush=True)
