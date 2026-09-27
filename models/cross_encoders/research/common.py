"""Shared residual harness (copied from l4_combo.py; num_threads=6)."""
import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, polars as pl
pl.Config.set_tbl_rows(200)
W = os.environ.get("ER_WORK_DIR", "work"); R = os.path.dirname(os.path.abspath(__file__))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
tr1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
s1v = tr1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"]))).select("entity_id", "country")
gv = gt.join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
ntrue = gv.group_by("s1_id").len().rename({"len": "nt"}); gtp = gv.with_columns(pl.lit(1).alias("t"))
def macro(scx, ts=None):
    sc1 = scx.sort("p", descending=True).unique("m_id", keep="first"); best = None
    for t in (np.arange(0.4, 0.96, 0.025).round(3) if ts is None else ts):
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
def oof(X, yv, grp, seed=7):
    import lightgbm as lgb
    p = np.zeros(len(yv)); prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=6, seed=seed)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[grp != k], yv[grp != k]), 300); p[grp == k] = m.predict(X[grp == k])
    return p
def load_base():
    """d: s1_id, m_id, y, p, c1, c2, ce_flag (VALID pair order of best_valid_scored)."""
    f = f"{R}/base_d.parquet"
    if os.path.exists(f): return pl.read_parquet(f)
    sc = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "y", "p")
    ce = pl.read_parquet(f"{W}/ce_team_valid.parquet").rename({"p": "ce1"})
    ce2 = pl.read_parquet(f"{W}/ce2_team_valid.parquet").rename({"p": "ce2"})
    d = sc.join(ce, on=["s1_id", "m_id"], how="left", maintain_order="left").join(ce2, on=["s1_id", "m_id"], how="left", maintain_order="left")
    d = d.with_columns(pl.col("ce1").is_not_null().cast(pl.Float64).alias("ce_flag"), pl.col("ce1").fill_null(pl.col("p")).cast(pl.Float64).alias("c1"),
                       pl.col("ce2").fill_null(pl.col("p")).cast(pl.Float64).alias("c2")).drop("ce1", "ce2")
    d.write_parquet(f); return d
def base_X(d):
    pp = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
    return [lp] + [d[c].cast(pl.Float64).to_numpy() for c in ("c1", "c2", "ce_flag")]
def run_test(d, feats, name, base_res=None):
    """feats: dict name -> np array aligned with d. Returns (A, B) macro dicts."""
    yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
    X0 = base_X(d)
    if base_res is None:
        bf = f"{R}/base_res.json"
        if os.path.exists(bf): base_res = json.load(open(bf))
        else:
            base_res = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(np.column_stack(X0), yy, grp)))); json.dump(base_res, open(bf, "w"))
    X1 = np.column_stack(X0 + [np.asarray(v, dtype=np.float64) for v in feats.values()])
    B = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(X1, yy, grp))))
    A = base_res
    log(f"BASE {json.dumps({k: round(float(v), 5) for k, v in A.items()})}")
    log(f"{name} {json.dumps({k: round(float(v), 5) for k, v in B.items()})}")
    log(f"RESULT {name} vs base: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
    return A, B
