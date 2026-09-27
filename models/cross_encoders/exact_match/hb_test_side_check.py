"""Hard-band CE: TEST-side checks used in review. New p only for argmax pairs in the decision zone (team p in
[.10, .995], where test hb scores exist); team p elsewhere; same rules. Stage-3 combiner [logit p, hb] fit on VALID (the team's
stage-3 practice), cross-fit halves for the VALID number, full fit for test.
VALID: gain by half with the zone restriction; changed pairs per 1k S1 (true/false adds, true/false removals) in NY and AP+TS.
TEST: adds / removals per 1k S1 in the SAME states (NY; AP+TS) and in all states -> implied precision = VALID true adds per 1k /
test adds per 1k (state-matched); P(k=1 | k>=1) of S1 receiving adds (random ~6.4%); after-removal mean k of S1 losing a pair
vs general mean. Writes hb_test_diff.parquet."""
import gzip, io, os, json
from collections import Counter
import numpy as np, polars as pl, lightgbm as lgb
W = os.environ.get("ER_WORK_DIR", "work"); E = f"{W}/e5hyb2"
BASE = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98}; T = 0.75
def thr_expr():
    key = pl.when(pl.col("sup") == "corr").then(pl.col("grp") + "|corr").otherwise(pl.col("grp")); e = pl.lit(T)
    for g_, x in BASE.items():
        if "|" in g_: e = pl.when(key == g_).then(pl.lit(x)).otherwise(e)
    for g_, x in BASE.items():
        if "|" not in g_: e = pl.when((pl.col("grp") == g_) & ~key.is_in([k for k in BASE if "|" in k])).then(pl.lit(x)).otherwise(e)
    return e
def decide(d, pcol):
    d = d.with_columns(pl.col(pcol).rank("ordinal", descending=True).over("m_id").alias("_r"))
    keep = (pl.col("_r") == 1) & (pl.col(pcol) >= thr_expr()) & ~pl.col("dtok").fill_null(False)
    restore = (pl.col("country") == "India") & pl.col("grp").is_in(["shift_pure", "shift_ms_only"]) & (pl.col(pcol) >= 0.98) & (pl.col("_r") == 1) & ~pl.col("dtok").fill_null(False)
    return d.filter(keep | restore).select("s1_id", "m_id")
flags = pl.read_parquet(f"{E}/rule_flags_pairs.parquet", columns=["s1_id", "m_id", "dtok", "split"])
v = pl.read_parquet(f"{E}/stage3_valid_oof_pairs.parquet").filter(pl.col("p").is_not_null()).join(flags.filter(pl.col("split") == "valid").drop("split"), on=["s1_id", "m_id"], how="left")
v = v.join(pl.read_parquet(f"{W}/out_hb/hb_team_valid.parquet").rename({"p": "hb"}), on=["s1_id", "m_id"], how="left", maintain_order="left")
v = v.with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r0"))
v = v.with_columns(((pl.col("r0") == 1) & (pl.col("p") >= 0.10) & (pl.col("p") <= 0.995) & pl.col("hb").is_not_null()).alias("zone"))
yv = v["y"].to_numpy(); hv = (v["s1_id"].hash(13) % 2).to_numpy()
pp = np.clip(v["p"].cast(pl.Float64).to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp)); hbv = v["hb"].fill_null(0.5).to_numpy()
Z = v["zone"].to_numpy()
PR = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8, seed=7)
X = np.c_[lp, hbv]
o = pp.copy()
for k in (0, 1):
    tr = (hv != k) & Z; m = lgb.train(PR, lgb.Dataset(X[tr], yv[tr]), 300); te = (hv == k) & Z; o[te] = m.predict(X[te])
full = lgb.train(PR, lgb.Dataset(X[Z], yv[Z]), 300)
v = v.with_columns(pl.Series("pn", o))
s1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
s1v = s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"]))).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
g2 = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id"))).join(s1v.select("s1_id"), on="s1_id")
nt = g2.group_by("s1_id").len().rename({"len": "nt"}); TP = set(zip(g2["s1_id"].to_list(), g2["m_id"].to_list()))
def macro(p_):
    d = p_.with_columns(pl.Series("y", [int((s, m) in TP) for s, m in zip(p_["s1_id"].to_list(), p_["m_id"].to_list())]))
    a = d.group_by("s1_id").agg(pl.len().alias("np"), pl.col("y").sum().alias("tp"))
    u = (s1v.join(a, on="s1_id", how="left").join(nt, on="s1_id", how="left").fill_null(0)
           .with_columns(pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
                         .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))).alias("f"),
                         (pl.col("s1_id").hash(13) % 2).alias("h")))
    return {"macro": u["f"].mean(), "h0": u.filter(pl.col("h") == 0)["f"].mean(), "h1": u.filter(pl.col("h") == 1)["f"].mean(),
            "US": u.filter(pl.col("country") == "US")["f"].mean(), "India": u.filter(pl.col("country") == "India")["f"].mean()}
b0, b1 = decide(v, "p"), decide(v, "pn"); A, B = macro(b0), macro(b1)
print(f"VALID zone-only hb stack: gain {B['macro'] - A['macro']:+.5f} halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
adds = b1.join(b0, on=["s1_id", "m_id"], how="anti"); rems = b0.join(b1, on=["s1_id", "m_id"], how="anti")
CT = dict(zip(s1v["s1_id"].to_list(), s1v["country"].to_list())); NV = Counter(s1v["country"].to_list())
def per1k(df, lab):
    c = Counter((CT[s], int((s, m) in TP)) for s, m in zip(df["s1_id"].to_list(), df["m_id"].to_list()))
    return {k: {"true": round(1000 * c[(k, 1)] / NV[k], 3), "false": round(1000 * c[(k, 0)] / NV[k], 3)} for k in ("US", "India")}
va, vr = per1k(adds, "adds"), per1k(rems, "rems")
print("VALID per 1k S1  adds:", va, " removals:", vr)
# TEST
te1 = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state"])
res = {}; diffs = []
for c, states in (("US", ["ny"]), ("India", ["ap", "ts"])):
    t = pl.read_parquet(f"{E}/stage3_test_pairs_{c}.parquet").join(flags.filter(pl.col("split") == "test").drop("split"), on=["s1_id", "m_id"], how="left")
    t = t.join(pl.read_parquet(f"{W}/out_hb/hb_team_test_{c}.parquet").rename({"p": "hb"}), on=["s1_id", "m_id"], how="left", maintain_order="left")
    q = np.clip(t["p"].cast(pl.Float64).to_numpy(), 1e-6, 1 - 1e-6); lq = np.log(q / (1 - q)); z = t["hb"].is_not_null().to_numpy()
    pn = q.copy(); pn[z] = full.predict(np.c_[lq[z], t["hb"].to_numpy()[z]])
    t = t.with_columns(pl.Series("pn", pn))
    d0, d1 = decide(t, "p"), decide(t, "pn")
    ad = d1.join(d0, on=["s1_id", "m_id"], how="anti"); rm = d0.join(d1, on=["s1_id", "m_id"], how="anti")
    diffs += [ad.with_columns(pl.lit(c).alias("country"), pl.lit("add").alias("action")), rm.with_columns(pl.lit(c).alias("country"), pl.lit("remove").alias("action"))]
    tc = te1.filter(pl.col("country") == c); ST = dict(zip(tc["entity_id"].to_list(), tc["addr_state"].to_list()))
    n_all = tc.height; n_st = tc.filter(pl.col("addr_state").is_in(states)).height
    a_st = sum(1 for s in ad["s1_id"].to_list() if ST.get(s) in states); r_st = sum(1 for s in rm["s1_id"].to_list() if ST.get(s) in states)
    kc = Counter(d0["s1_id"].to_list()); kc1 = Counter(d1["s1_id"].to_list())
    recv = set(ad["s1_id"].to_list()); pk1 = sum(1 for s in recv if kc1[s] == 1) / max(len(recv), 1)
    rnd = sum(1 for s, k in kc1.items() if k == 1) / max(len(kc1), 1)
    lost = set(rm["s1_id"].to_list()); after = np.mean([kc1[s] for s in lost]) if lost else 0; gen = np.mean(list(kc.values()))
    ta, tr_ = 1000 * a_st / n_st, 1000 * r_st / n_st
    imp = va[c]["true"] / ta if ta else float("nan")
    print(f"TEST {c}: adds {ad.height} removals {rm.height} | same-state per 1k S1: adds {ta:.3f} (VALID true {va[c]['true']}, false {va[c]['false']}) -> implied add precision {imp:.2f}; removals {tr_:.3f} (VALID true {vr[c]['true']}, false {vr[c]['false']})")
    print(f"      all-state per 1k: adds {1000 * ad.height / n_all:.3f} removals {1000 * rm.height / n_all:.3f} | P(k=1 | k>=1) receivers {pk1:.3f} vs all S1 {rnd:.3f} | after-removal mean k {after:.2f} vs general {gen:.2f}")
pl.concat(diffs).write_parquet(f"{W}/hb_test_diff.parquet")
print("HBTEST_DONE")
