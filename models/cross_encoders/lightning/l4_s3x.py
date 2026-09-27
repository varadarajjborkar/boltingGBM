"""Locality-alias features (new: the pipeline has no city/locality feature). Generator swaps a suburb for its metro city
(Lackawanna -> Buffalo, Irondequoit -> Rochester). Learn locality co-occurrence from TRUE training pairs and score each pair:
loc_exact (shared locality component), loc_alias (max learned P(record locality | S1 locality) over differing pairs),
loc_conflict (record names a locality that is neither equal nor a known alias), rec_has_loc.
Fair VALID: alias tables for VALID S1 of half h are learned from non-VALID states + VALID half (1-h) only (s1_id.hash(13)%2).
Test: tables from ALL training true pairs. Residual on the team's stack_s3a2 VALID p + e5 v1 (2 folds by the same hash)."""
import json, os, re, time
from collections import Counter, defaultdict
import numpy as np, polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/out_s3x"; os.makedirs(OUT, exist_ok=True)
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
cols = ["entity_id", "business_address", "country", "addr_state"]
tr1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=cols)
trr = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=cols) for s in (2, 3)])
L = {}
for df in (tr1, trr):
    for e, a in zip(df["entity_id"].to_list(), df["business_address"].to_list()):
        L[e] = localities(a)
log(f"train localities parsed: {len(L)} records ({(time.time() - t0) / 60:.1f} min)")
s1state = dict(zip(tr1["entity_id"].to_list(), zip(tr1["country"].to_list(), tr1["addr_state"].fill_null("").to_list())))
isvalid = lambda s: s1state.get(s, ("", ""))[0] == "US" and s1state[s][1] == "ny" or s1state.get(s, ("", ""))[0] == "India" and s1state[s][1] in ("ap", "ts")
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
half = dict(zip(tr1["entity_id"].to_list(), (tr1["entity_id"].hash(13) % 2).to_list()))
# counts: bucket 'all' (non-VALID + VALID), 'h0'/'h1' (VALID halves); table for half h = nonvalid + valid(1-h)
cnt = {k: Counter() for k in ("nv", "v0", "v1")}; tot = {k: Counter() for k in ("nv", "v0", "v1")}
for s, m in zip(gt["s1_id"].to_list(), gt["m_id"].to_list()):
    a, b = L.get(s, set()), L.get(m, set())
    if not a or not b: continue
    k = ("v0" if half[s] == 0 else "v1") if isvalid(s) else "nv"
    for x in a:
        tot[k][x] += 1
        for y in b:
            if y != x: cnt[k][(x, y)] += 1
log(f"alias counts built ({(time.time() - t0) / 60:.1f} min): nv {len(cnt['nv'])} pairs")
def table(keys):
    c = Counter(); t = Counter()
    for k in keys: c.update(cnt[k]); t.update(tot[k])
    return c, t
TAB = {0: table(["nv", "v1"]), 1: table(["nv", "v0"]), "all": table(["nv", "v0", "v1"])}
def feats(s1s, ms, Lmap, tabsel):
    ex, al, cf, hl = [], [], [], []
    for s, m in zip(s1s, ms):
        a, b = Lmap.get(s, set()), Lmap.get(m, set()); c, t = TAB[tabsel(s)]
        e = bool(a & b); best = 0.0
        for x in a:
            tx = t.get(x, 0)
            if tx >= 3:
                for y in b - a:
                    best = max(best, c.get((x, y), 0) / tx)
        ex.append(float(e)); al.append(best); hl.append(float(bool(b)))
        cf.append(float(bool(b) and bool(a) and not e and best < 0.01))
    return ex, al, cf, hl

def comp_prof(d, recprof):
    d = d.join(recprof, on="m_id", how="left", maintain_order="left")
    d = d.with_columns(pl.col("c1").fill_null(pl.col("p")).alias("c1"))
    g = "m_id"
    d = d.with_columns(pl.len().over(g).cast(pl.Float64).alias("rec_ncand"), pl.col("p").rank("ordinal", descending=True).over(g).cast(pl.Float64).alias("p_rank_rec"),
                       pl.col("c1").rank("ordinal", descending=True).over(g).cast(pl.Float64).alias("c_rank_rec"), pl.col("p").max().over(g).alias("_pmax"), pl.col("c1").max().over(g).alias("_cmax"))
    two = d.group_by(g).agg(pl.col("p").sort(descending=True).head(2).alias("_tp"), pl.col("c1").sort(descending=True).head(2).alias("_tc"))
    two = two.with_columns(pl.col("_tp").list.get(1, null_on_oob=True).alias("_p2"), pl.col("_tc").list.get(1, null_on_oob=True).alias("_c2")).drop("_tp", "_tc")
    d = d.join(two, on=g, how="left", maintain_order="left")
    d = d.with_columns(pl.when(pl.col("p") >= pl.col("_pmax")).then(pl.col("_p2")).otherwise(pl.col("_pmax")).fill_null(0.0).alias("p_best_other"),
                       pl.when(pl.col("c1") >= pl.col("_cmax")).then(pl.col("_c2")).otherwise(pl.col("_cmax")).fill_null(0.0).alias("c_best_other"))
    d = d.with_columns((pl.col("p") - pl.col("p_best_other")).alias("p_margin"), (pl.col("c1") - pl.col("c_best_other")).alias("c_margin"))
    return d.drop("_pmax", "_cmax", "_p2", "_c2")
def recprof(split):
    rp = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "name_non_latin", "addr_empty", "name_is_domain"]) for s in (2, 3)])
    return rp.select(pl.col("entity_id").alias("m_id"), pl.col("entity_id").str.starts_with("S3").cast(pl.Float64).alias("rec_s3"),
                     pl.col("name_non_latin").cast(pl.Float64).fill_null(0).alias("rec_native"), pl.col("addr_empty").cast(pl.Float64).fill_null(0).alias("rec_noaddr"),
                     pl.col("name_is_domain").cast(pl.Float64).fill_null(0).alias("rec_domain"), pl.col("business_name").fill_null("").str.len_chars().cast(pl.Float64).alias("rec_nlen"))
COLS = ["rec_ncand", "p_rank_rec", "c_rank_rec", "p_best_other", "c_best_other", "p_margin", "c_margin", "rec_s3", "rec_native", "rec_noaddr",
        "rec_domain", "rec_nlen", "loc_exact", "loc_alias", "loc_conflict", "rec_has_loc"]
# VALID (locality tables cross-fitted by S1-hash half, as measured)
sc = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
d = sc.join(pl.read_parquet(f"{W}/ce_team_valid.parquet").rename({"p": "c1"}), on=["s1_id", "m_id"], how="left", maintain_order="left")
d = comp_prof(d, recprof("train"))
ex, al, cf, hl = feats(d["s1_id"].to_list(), d["m_id"].to_list(), L, lambda s: half[s])
d = d.with_columns(pl.Series("loc_exact", ex), pl.Series("loc_alias", al), pl.Series("loc_conflict", cf), pl.Series("rec_has_loc", hl))
d.select(["s1_id", "m_id"] + COLS).write_parquet(f"{OUT}/s3x_valid.parquet"); log(f"VALID features: {d.height} pairs")
# TEST (locality tables from ALL training true pairs)
cols = ["entity_id", "business_address"]
LT = {}
for s in (1, 2, 3):
    t = pl.read_parquet(f"{W}/norm_v2/test_source{s}.parquet", columns=cols)
    for e, a in zip(t["entity_id"].to_list(), t["business_address"].to_list()):
        LT[e] = localities(a)
log(f"test localities parsed: {len(LT)}")
rpt = recprof("test")
for c in ["France", "India", "US"]:
    t = pl.read_parquet(f"{W}/best_test_scored_{c}.parquet").select("s1_id", "m_id", "p")
    t = t.join(pl.read_parquet(f"{W}/ce_team_test_{c}.parquet").rename({"p": "c1"}), on=["s1_id", "m_id"], how="left", maintain_order="left")
    t = comp_prof(t, rpt)
    ex, al, cf, hl = feats(t["s1_id"].to_list(), t["m_id"].to_list(), LT, lambda s: "all")
    t = t.with_columns(pl.Series("loc_exact", ex), pl.Series("loc_alias", al), pl.Series("loc_conflict", cf), pl.Series("rec_has_loc", hl))
    t.select(["s1_id", "m_id"] + COLS).write_parquet(f"{OUT}/s3x_test_{c}.parquet"); log(f"test {c}: {t.height} pairs")
print("S3X_DONE", flush=True)
