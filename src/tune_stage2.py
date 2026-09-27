"""Hyperparameter search for the stage-2 LightGBM (random search on the real competition metric).

Selection protocol (guards against picking a lucky configuration):
  VALID S1 entities (held-out states) are split in two halves by entity hash. Each configuration is scored with
  macro F0.5 on half A (threshold chosen on A) and then on half B at that same threshold. We rank by A and report B;
  a winner must also beat the baseline on B.

Stage 1 is fixed (the fold models in ER_MODEL_DIR). Design matrices are built once and cached under
<model dir>/tune/ (about 0.8 GB), so each configuration only trains one stage-2 model (~30-60 s).
Results are appended to <model dir>/tune/stage2_results.jsonl; a restarted run skips finished configurations.

Usage: ER_MODEL_DIR=.../model_v2s ER_FAST_PREDICT=1 python tune_stage2.py --n 40 [--threads 7]
"""
import argparse
import hashlib
import json
import random
import time

import numpy as np
import polars as pl

import p1_train as t
from utils import announce_pid, log, write_parquet_atomic

T = t.M / "tune"
THRESHOLDS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
SPACE = dict(learning_rate=[0.02, 0.03, 0.05, 0.08, 0.1], num_leaves=[15, 31, 63, 127, 255],
             min_data_in_leaf=[20, 50, 100, 300, 1000], feature_fraction=[0.5, 0.6, 0.7, 0.8, 0.9],
             bagging_fraction=[0.6, 0.7, 0.8, 0.9, 1.0], lambda_l2=[0.0, 0.1, 1.0, 5.0, 10.0],
             min_gain_to_split=[0.0, 0.01, 0.1])
BASELINE = dict(learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                lambda_l2=1.0, min_gain_to_split=0.0)


def prepare():
    """Build and cache: stage-2 design matrices for FIT (out-of-fold p1) and VALID (fold-average p1), truth, halves."""
    if (T / "DONE").exists():
        return
    from stack import SIB_COLS, sibling_features, stage2_features
    T.mkdir(exist_ok=True)
    cols = json.loads((t.M / "feature_cols.json").read_text())
    allc = json.loads((t.M / "stage2_cols.json").read_text())
    s2cols = allc[len(cols):]
    sib = any(c in SIB_COLS for c in s2cols)
    texts = t.train_texts() if sib else None

    def design(p1tab):
        s2 = stage2_features(p1tab).filter(pl.col("p1") >= t.P1_MIN)
        if sib:
            s2 = s2.join(sibling_features(p1tab, texts, t.P1_MIN), on=["s1_id", "m_id"], how="left")
        return s2

    p1f = pl.read_parquet(t.M / "fit_oof_p1.parquet").select("s1_id", "m_id", "p1")
    Xf, mf = t.load_role_numpy("fit", cols, keep=design(p1f), extra_cols=s2cols)
    np.save(T / "X_fit.npy", Xf)
    np.save(T / "y_fit.npy", mf["y"].to_numpy().astype(np.int8))
    np.save(T / "es_fit.npy", (mf["s1_grp"].to_numpy() % 7 == 3))
    del Xf
    Xv, mv = t.load_role_numpy("valid", cols)
    p1v = t.predict_p1(Xv)
    del Xv
    p1tab = mv.select("s1_id", "m_id").with_columns(pl.Series("p1", p1v.astype(np.float32)))
    Xv2, mv2 = t.load_role_numpy("valid", cols, keep=design(p1tab), extra_cols=s2cols)
    np.save(T / "X_val.npy", Xv2)
    write_parquet_atomic(mv2.select("s1_id", "m_id"), T / "val_ids.parquet")
    s1v = pl.concat([pl.read_parquet(t.WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state"])
                     .filter(t.role_mask(c, "valid")) for c in t.COUNTRIES]).select(pl.col("entity_id").alias("s1_id"))
    s1v = s1v.with_columns((pl.col("s1_id").hash(21) % 2).cast(pl.Int8).alias("half"))
    write_parquet_atomic(s1v, T / "val_universe.parquet")
    write_parquet_atomic(t.labels().join(s1v.select("s1_id"), on="s1_id").select("s1_id", "m_id"), T / "val_truth.parquet")
    (T / "DONE").write_text("ok")
    log(f"prepared: FIT {len(mf)} rows, VALID {len(mv2)} rows, {s1v.height} VALID S1")


def macro_by_half(scored: pl.DataFrame, truth: pl.DataFrame, uni: pl.DataFrame, thr: float) -> dict:
    """Exact competition metric (see metric.entity_f05), vectorised: one S1 per record, then threshold."""
    pred = (scored.filter(pl.col("p") == pl.col("p").max().over("m_id")).unique(subset=["m_id"], keep="first")
                  .filter(pl.col("p") >= thr).select("s1_id", "m_id"))
    n_p = pred.group_by("s1_id").len().rename({"len": "np"})
    n_t = truth.group_by("s1_id").len().rename({"len": "nt"})
    n_tp = pred.join(truth, on=["s1_id", "m_id"]).group_by("s1_id").len().rename({"len": "tp"})
    d = (uni.join(n_p, on="s1_id", how="left").join(n_t, on="s1_id", how="left").join(n_tp, on="s1_id", how="left")
            .with_columns(pl.col("np", "nt", "tp").fill_null(0)))
    f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))))
    r = d.with_columns(f.alias("f")).group_by("half").agg(pl.col("f").mean())
    return {int(h): float(v) for h, v in r.iter_rows()}


def cfg_id(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="random configurations (plus the baseline)")
    ap.add_argument("--threads", type=int, default=t.THREADS)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    announce_pid("tune_stage2")
    prepare()
    Xf, yf, es = np.load(T / "X_fit.npy"), np.load(T / "y_fit.npy"), np.load(T / "es_fit.npy")
    Xv = np.load(T / "X_val.npy")
    ids = pl.read_parquet(T / "val_ids.parquet")
    truth, uni = pl.read_parquet(T / "val_truth.parquet"), pl.read_parquet(T / "val_universe.parquet")
    dp = {"feature_pre_filter": False, "max_bin": 255, "verbose": -1}   # binned once, reused by every configuration
    dtr = lgb.Dataset(Xf[~es], yf[~es], params=dp, free_raw_data=False).construct()
    des = lgb.Dataset(Xf[es], yf[es], reference=dtr, params=dp).construct()
    out = T / "stage2_results.jsonl"
    seen = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rng = random.Random(a.seed)
    cfgs = [BASELINE] + [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(a.n)]
    for n, cfg in enumerate(cfgs):
        cid = cfg_id(cfg)
        if cid in seen:
            continue
        t0 = time.time()
        params = {**t.PARAMS, **cfg, "num_threads": a.threads}
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        sc = ids.with_columns(pl.Series("p", m.predict(Xv, num_threads=a.threads)))
        res = {thr: macro_by_half(sc, truth, uni, thr) for thr in THRESHOLDS}
        best_t = max(THRESHOLDS, key=lambda x: res[x][0])
        rec = dict(id=cid, cfg=cfg, iters=m.best_iteration, secs=round(time.time() - t0), best_t_on_A=best_t,
                   A=round(res[best_t][0], 5), B_at_A_t=round(res[best_t][1], 5),
                   B_best=round(max(res[x][1] for x in THRESHOLDS), 5), baseline=(cfg == BASELINE))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"[{n + 1}/{len(cfgs)}] A {rec['A']:.5f} B {rec['B_at_A_t']:.5f} t={best_t} iters {m.best_iteration} "
            f"{rec['secs']}s  {cfg}")
    rs = [json.loads(l) for l in out.read_text().splitlines()]
    base = next((r for r in rs if r["baseline"]), None)
    log("TOP 5 by half A (B = confirmation on the other half):")
    for r in sorted(rs, key=lambda r: -r["A"])[:5]:
        log(f"  A {r['A']:.5f}  B {r['B_at_A_t']:.5f}  t={r['best_t_on_A']}  {r['cfg']}")
    if base:
        log(f"  baseline: A {base['A']:.5f}  B {base['B_at_A_t']:.5f}")


if __name__ == "__main__":
    main()
