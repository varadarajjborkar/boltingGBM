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
ce = pl.read_parquet(f"{W}/ce_team_valid.parquet").rename({"p": "ce1"})
ce2 = pl.read_parquet(f"{W}/ce2_team_valid.parquet").rename({"p": "ce2"})
loc = pl.read_parquet(f"{W}/out_loc/loc_valid.parquet")
d = (sc.join(ce, on=["s1_id", "m_id"], how="left", maintain_order="left").join(ce2, on=["s1_id", "m_id"], how="left", maintain_order="left")
       .join(rp, on="m_id", how="left", maintain_order="left").join(loc, on=["s1_id", "m_id"], how="left", maintain_order="left"))
d = d.with_columns(pl.col("ce1").is_not_null().cast(pl.Float64).alias("ce_flag"), pl.col("ce1").fill_null(pl.col("p")).alias("c1"), pl.col("ce2").fill_null(pl.col("p")).alias("c2")).fill_null(0.0)
g = "m_id"
d = d.with_columns(pl.len().over(g).cast(pl.Float64).alias("rec_ncand"), pl.col("p").rank("ordinal", descending=True).over(g).cast(pl.Float64).alias("p_rank_rec"),
                   pl.col("c1").rank("ordinal", descending=True).over(g).cast(pl.Float64).alias("c_rank_rec"), pl.col("p").max().over(g).alias("_pmax"), pl.col("c1").max().over(g).alias("_cmax"))
two = d.group_by(g).agg(pl.col("p").sort(descending=True).head(2).alias("_tp"), pl.col("c1").sort(descending=True).head(2).alias("_tc"))
two = two.with_columns(pl.col("_tp").list.get(1, null_on_oob=True).alias("_p2"), pl.col("_tc").list.get(1, null_on_oob=True).alias("_c2")).drop("_tp", "_tc")
d = d.join(two, on=g, how="left", maintain_order="left")
d = d.with_columns(pl.when(pl.col("p") >= pl.col("_pmax")).then(pl.col("_p2")).otherwise(pl.col("_pmax")).fill_null(0.0).alias("p_best_other"),
                   pl.when(pl.col("c1") >= pl.col("_cmax")).then(pl.col("_c2")).otherwise(pl.col("_cmax")).fill_null(0.0).alias("c_best_other"))
d = d.with_columns((pl.col("p") - pl.col("p_best_other")).alias("p_margin"), (pl.col("c1") - pl.col("c_best_other")).alias("c_margin"))
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
log(f"SANITY raw stack_s3a2 macro (must be ~0.9873): {macro(sc.select('s1_id', 'm_id', 'p'))}")
def oof(X, yv, grp):
    p = np.zeros(len(yv)); prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8, seed=7)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[grp != k], yv[grp != k]), 300); p[grp == k] = m.predict(X[grp == k])
    return p
pp = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
col = lambda cs: [d[c].cast(pl.Float64).to_numpy() for c in cs]
BASE = [lp] + col(["c1", "c2", "ce_flag"])
COMP = col(["rec_ncand", "p_rank_rec", "c_rank_rec", "p_best_other", "c_best_other", "p_margin", "c_margin"])
PROF = col(["rec_s3", "rec_native", "rec_noaddr", "rec_domain", "rec_nlen"])
LOC = col(["loc_exact", "loc_alias", "loc_conflict", "rec_has_loc"])
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
res = {}
for name, X in (("A stack+e5v1+v2", BASE), ("B +comp", BASE + COMP), ("C +profile", BASE + PROF), ("D +locality", BASE + LOC), ("E +ALL", BASE + COMP + PROF + LOC)):
    res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(np.column_stack(X), yy, grp))))
    log(f"{name:18s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
A = res["A stack+e5v1+v2"]
for k in list(res)[1:]:
    B = res[k]; log(f"{k} vs A: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
json.dump(res, open(f"{W}/combo_result.json", "w"), default=float, indent=1)
print("COMBO_DONE", flush=True)
