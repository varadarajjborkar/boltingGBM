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
sc = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "y", "p")
ce = pl.read_parquet(f"{W}/ce_team_valid.parquet").rename({"p": "ce"})
d = sc.join(ce, on=["s1_id", "m_id"], how="left", maintain_order="left")
ex, al, cf, hl = feats(d["s1_id"].to_list(), d["m_id"].to_list(), L, lambda s: half[s])
d = d.with_columns(pl.Series("loc_exact", ex), pl.Series("loc_alias", al), pl.Series("loc_conflict", cf), pl.Series("rec_has_loc", hl))
log(f"VALID features: exact {np.mean(ex):.3f}, alias>0 {np.mean(np.array(al) > 0):.3f}, conflict {np.mean(cf):.3f}; "
    f"P(y | conflict) {d.filter(pl.col('loc_conflict') == 1)['y'].mean():.3f}, P(y | alias>0.05, not exact) "
    f"{d.filter((pl.col('loc_alias') > 0.05) & (pl.col('loc_exact') == 0))['y'].mean()}")
# residual (same machinery as l4_ce4.py)
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
cef = d["ce"].fill_null(-1.0).to_numpy(); fl = d["ce"].is_not_null().cast(pl.Float64).to_numpy()
LOC = np.column_stack([d[c].to_numpy() for c in ("loc_exact", "loc_alias", "loc_conflict", "rec_has_loc")])
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
res = {}
for name, X in (("A stack+e5", np.column_stack([lp, cef, fl])), ("B stack+e5+locality", np.column_stack([lp, cef, fl, LOC]))):
    res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(X, yy, grp))))
    log(f"{name:22s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
A, B = res["A stack+e5"], res["B stack+e5+locality"]
log(f"LOCALITY over stack+e5: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
json.dump(res, open(f"{OUT}/loc_result.json", "w"), default=float, indent=1)
d.select("s1_id", "m_id", "loc_exact", "loc_alias", "loc_conflict", "rec_has_loc").write_parquet(f"{OUT}/loc_valid.parquet")
print("LOC_VALID_DONE", flush=True)
