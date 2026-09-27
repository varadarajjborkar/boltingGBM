"""Address-less NAME-TIE breaker from RAW-name evidence (legal form, accents, case, punctuation, token order) that name_core
throws away. A record without an address whose country+name_core is shared by k >= 2 S1 is scored against every tied S1.
LightGBM trained ONLY on tie groups that contain no VALID S1 (US ny, India ap/ts) -> no VALID labels used. Evaluated on VALID
with the team's exact decision (stack_e5hyb2 stage-3 OOF + rules): add (unassigned record) / move (assigned elsewhere) when
the top tied S1 has q >= tau; macro F0.5 on both halves. Then scores TEST ties -> tie_test_proposals.parquet."""
import os, re, json, unicodedata, time
from collections import defaultdict
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz.distance import Levenshtein
W = os.environ.get("ER_WORK_DIR", "work"); E = f"{W}/e5hyb2"; t0 = time.time()
def log(s): print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
def fold(s): return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()
def toks(s): return re.findall(r"[a-z0-9]+", fold(s))
PUN = re.compile(r"[^\w\s]")
FEATS = ["k", "raw_eq", "cf_eq", "fold_eq", "tok_eq", "tokset_eq", "legal_eq", "rec_legal", "s1_legal", "lev_raw", "lev_fold", "lev_gap",
         "lev_unique_min", "jac", "jac_gap", "jac_unique_max", "n_tok_eq", "n_legal_eq", "n_fold_eq", "wdiff", "ds", "acc_eq", "pun_eq",
         "same_raw_in_group", "ntok_rec", "order_eq"]
def build(split, s1cols, reccols):
    s1 = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source1.parquet", columns=s1cols)]).fill_null("")
    rec = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{k}.parquet", columns=reccols) for k in (2, 3)]).filter(pl.col("addr_empty")).fill_null("")
    grp = defaultdict(list); S = {}
    for e, n, c, core, lg in zip(s1["entity_id"].to_list(), s1["business_name"].to_list(), s1["country"].to_list(), s1["name_core"].to_list(), s1["name_legal"].to_list()):
        if core: grp[(c, core)].append(e); S[e] = (n, lg, fold(n), toks(n))
    rows = []
    for m, n, c, core, lg in zip(rec["entity_id"].to_list(), rec["business_name"].to_list(), rec["country"].to_list(), rec["name_core"].to_list(), rec["name_legal"].to_list()):
        G = grp.get((c, core))
        if not G or len(G) < 2 or len(G) > 40: continue
        fn, tk = fold(n), toks(n); lgs = set(lg.split()); ds = float("  " in n); acc = any(ord(ch) > 127 for ch in n); pun = set(PUN.findall(n))
        per = []
        for x in G:
            xn, xl, xf, xt = S[x]
            per.append((x, float(n == xn), float(n.lower() == xn.lower()), float(fn == xf), float(tk == xt), float(set(tk) == set(xt)),
                        float(set(xl.split()) == lgs), float(bool(lgs)), float(bool(xl)), float(Levenshtein.distance(n, xn)), float(Levenshtein.distance(fn, xf)),
                        len(set(tk) & set(xt)) / max(len(set(tk) | set(xt)), 1), float(len(tk) - len(xt)), float(acc == any(ord(ch) > 127 for ch in xn)),
                        float(pun == set(PUN.findall(xn))), float(tk == xt or [w for w in tk if w in set(xt)] == [w for w in xt if w in set(tk)])))
        lmin = min(r[10] for r in per); jmax = max(r[11] for r in per)
        nl = sum(1 for r in per if r[10] == lmin); nj = sum(1 for r in per if r[11] == jmax)
        ntok = sum(r[4] for r in per); nleg = sum(r[6] for r in per); nfold = sum(r[3] for r in per)
        rawc = defaultdict(int)
        for x in G: rawc[S[x][0]] += 1
        for r in per:
            x = r[0]
            rows.append((m, x, c, len(G), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[10] - lmin, float(r[10] == lmin and nl == 1),
                         r[11], jmax - r[11], float(r[11] == jmax and nj == 1), ntok, nleg, nfold, r[12], ds, r[13], r[14], float(rawc[S[x][0]]), float(len(tk)), r[15]))
    return pl.DataFrame(rows, schema=["m_id", "s1_id", "country"] + FEATS, orient="row")
S1C = ["entity_id", "business_name", "country", "name_core", "name_legal"]; RC = S1C + ["addr_empty"]
tr = build("train", S1C, RC); log(f"train tie rows {tr.height}")
s1t = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
VS = set(s1t.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list())
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
tr = tr.with_columns(pl.Series("y", [int(owner.get(m) == s) for m, s in zip(tr["m_id"].to_list(), tr["s1_id"].to_list())]),
                     pl.col("s1_id").is_in(list(VS)).any().over("m_id").alias("hasV"))
fit = tr.filter(~pl.col("hasV")); ev = tr.filter(pl.col("hasV"))
log(f"fit rows {fit.height} (records {fit['m_id'].n_unique()}, pos {fit['y'].sum()}), VALID-group rows {ev.height} (records {ev['m_id'].n_unique()})")
PR = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9, verbose=-1, num_threads=12, seed=7)
X = fit.select(FEATS).to_numpy()
# fold-check on fit groups (hash of record) for calibration of the top candidate
h = (fit["m_id"].hash(5) % 2).to_numpy(); oof = np.zeros(fit.height)
for k in (0, 1):
    mdl = lgb.train(PR, lgb.Dataset(X[h != k], fit["y"].to_numpy()[h != k]), 400); oof[h == k] = mdl.predict(X[h == k])
model = lgb.train(PR, lgb.Dataset(X, fit["y"].to_numpy()), 400)
def top(d, p):
    d = d.with_columns(pl.Series("p", p)).with_columns((pl.col("p") / pl.max_horizontal(pl.col("p").sum().over("m_id"), pl.lit(1.0))).alias("q"))
    return d.sort("q", descending=True).unique("m_id", keep="first")
tf = top(fit, oof)
for c in ("US", "India"):
    for tau in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        z = tf.filter((pl.col("country") == c) & (pl.col("q") >= tau))
        log(f"   fit-OOF {c:6s} q>={tau:.2f}: records {z.height:6d} of {tf.filter(pl.col('country') == c).height} precision {z['y'].mean() if z.height else 0:.3f}")
imp = sorted(zip(model.feature_importance("gain"), FEATS), reverse=True)[:12]; log("top features: " + ", ".join(f"{f}:{g:.0f}" for g, f in imp))
te_ = top(ev, model.predict(ev.select(FEATS).to_numpy())).filter(pl.col("s1_id").is_in(list(VS)))
# team decision on VALID
BASE = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98}; T = 0.75
def thr_expr():
    key = pl.when(pl.col("sup") == "corr").then(pl.col("grp") + "|corr").otherwise(pl.col("grp")); e = pl.lit(T)
    for g_, x in BASE.items():
        if "|" in g_: e = pl.when(key == g_).then(pl.lit(x)).otherwise(e)
    for g_, x in BASE.items():
        if "|" not in g_: e = pl.when((pl.col("grp") == g_) & ~key.is_in([k for k in BASE if "|" in k])).then(pl.lit(x)).otherwise(e)
    return e
flags = pl.read_parquet(f"{E}/rule_flags_pairs.parquet", columns=["s1_id", "m_id", "dtok", "split"]).filter(pl.col("split") == "valid").drop("split")
v = pl.read_parquet(f"{E}/stage3_valid_oof_pairs.parquet").filter(pl.col("p").is_not_null()).join(flags, on=["s1_id", "m_id"], how="left")
v = v.with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("_r"))
keep = (pl.col("_r") == 1) & (pl.col("p") >= thr_expr()) & ~pl.col("dtok").fill_null(False)
restore = (pl.col("country") == "India") & pl.col("grp").is_in(["shift_pure", "shift_ms_only"]) & (pl.col("p") >= 0.98) & (pl.col("_r") == 1) & ~pl.col("dtok").fill_null(False)
base = v.filter(keep | restore).select("s1_id", "m_id")
s1v = s1t.filter(pl.col("entity_id").is_in(list(VS))).rename({"entity_id": "s1_id"})
g2 = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id"))).join(s1v.select("s1_id"), on="s1_id")
nt = g2.group_by("s1_id").len().rename({"len": "nt"}); TP = set(zip(g2["s1_id"].to_list(), g2["m_id"].to_list()))
def macro(pairs):
    d = pairs.with_columns(pl.Series("y", [int((s, m) in TP) for s, m in zip(pairs["s1_id"].to_list(), pairs["m_id"].to_list())]))
    a = d.group_by("s1_id").agg(pl.len().alias("np"), pl.col("y").sum().alias("tp"))
    u = (s1v.join(a, on="s1_id", how="left").join(nt, on="s1_id", how="left").fill_null(0)
           .with_columns(pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
                         .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))).alias("f"),
                         (pl.col("s1_id").hash(13) % 2).alias("h")))
    return {"macro": u["f"].mean(), "h0": u.filter(pl.col("h") == 0)["f"].mean(), "h1": u.filter(pl.col("h") == 1)["f"].mean(),
            "US": u.filter(pl.col("country") == "US")["f"].mean(), "India": u.filter(pl.col("country") == "India")["f"].mean()}
B0 = macro(base); log("team decision VALID " + json.dumps({k: round(x, 5) for k, x in B0.items()}))
cur = dict(zip(base["m_id"].to_list(), base["s1_id"].to_list()))
st = te_.with_columns(pl.Series("cur", [cur.get(m) for m in te_["m_id"].to_list()]))
log(f"VALID tie records whose top tied S1 is VALID: {st.height}; currently assigned {st['cur'].is_not_null().sum()} (to the top {(st['cur'] == st['s1_id']).sum()}); top is owner {st['y'].mean():.3f}")
for tau in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    for mode in ("add", "add+move"):
        z = st.filter((pl.col("q") >= tau) & (pl.col("cur").is_null() | ((pl.col("cur") != pl.col("s1_id")) if mode == "add+move" else pl.lit(False))))
        new = pl.concat([base.filter(~pl.col("m_id").is_in(z["m_id"].to_list())), z.select("s1_id", "m_id")])
        R = macro(new)
        log(f"tau {tau:.2f} {mode:8s} changes {z.height:5d} (prec {z['y'].mean() if z.height else 0:.3f}) gain {R['macro'] - B0['macro']:+.5f} halves {R['h0'] - B0['h0']:+.5f} {R['h1'] - B0['h1']:+.5f} US {R['US'] - B0['US']:+.5f} India {R['India'] - B0['India']:+.5f}")
# TEST
tt = build("test", S1C, RC); log(f"test tie rows {tt.height}")
tt = top(tt, model.predict(tt.select(FEATS).to_numpy()))
tt.select("s1_id", "m_id", "country", "q", "p", "k").write_parquet(f"{W}/tie_test_top.parquet")
for c in ("US", "India", "France"):
    log(f"TEST {c}: tie records {tt.filter(pl.col('country') == c).height}; q>=.75 {tt.filter((pl.col('country') == c) & (pl.col('q') >= .75)).height}; q>=.9 {tt.filter((pl.col('country') == c) & (pl.col('q') >= .9)).height}")
print("TIEMODEL_DONE", flush=True)
