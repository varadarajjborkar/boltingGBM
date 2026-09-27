"""Hyperparameter search for the stage-1 LightGBM (the main pair model), random search on the real metric.

Protocol (same as tune_stage2.py): each configuration trains on the FIT fold-0 split (early stopping on its own
entity slice), predicts every VALID pair, and is scored with macro F0.5 of the stage-1 decision alone
(one S1 per record + threshold) on VALID half A (threshold chosen on A) and half B (confirmation).
Stage 2 is left out on purpose: it would need a full out-of-fold refit per configuration; a better stage 1 feeds
a better stage 2. Also logged: VALID log loss.

Results: <model dir>/tune/stage1_results.jsonl (a restarted run skips finished configurations).
Usage: ER_MODEL_DIR=.../model ER_THREADS=8 python tune_stage1.py --n 12
"""
import argparse
import json
import random
import time

import numpy as np
import polars as pl

import p1_train as t
from tune_stage2 import cfg_id, macro_by_half
from utils import announce_pid, log, write_parquet_atomic

T = t.M / "tune"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SPACE = dict(learning_rate=[0.03, 0.05, 0.1], num_leaves=[63, 127, 255, 511], min_data_in_leaf=[50, 100, 300, 1000],
             feature_fraction=[0.6, 0.7, 0.8, 0.9], bagging_fraction=[0.7, 0.8, 1.0], lambda_l2=[0.0, 1.0, 10.0])
BASELINE = dict(learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                lambda_l2=1.0)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="random configurations (plus the baseline)")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    announce_pid("tune_stage1")
    T.mkdir(exist_ok=True)
    cols = json.loads((t.M / "feature_cols.json").read_text())
    X, meta = t.load_role_numpy("fit", cols)
    y, grp = meta["y"].to_numpy(), meta["s1_grp"].to_numpy()
    tr = grp % 2 != 0                        # fold-0 training part, exactly as in p1_train stage 1
    es = tr & (grp % 7 == 3)
    trn = tr & ~es
    dp = {"feature_pre_filter": False, "max_bin": 255, "verbose": -1}
    dtr = lgb.Dataset(X[trn], y[trn], params=dp, free_raw_data=True).construct()
    des = lgb.Dataset(X[es], y[es], reference=dtr, params=dp).construct()
    del X
    log(f"FIT fold-0: train {trn.sum()} / early-stop {es.sum()} rows")
    Xv, mv = t.load_role_numpy("valid", cols)
    yv = mv["y"].to_numpy()
    ids = mv.select("s1_id", "m_id")
    s1v = pl.concat([pl.read_parquet(t.WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state"])
                     .filter(t.role_mask(c, "valid")) for c in t.COUNTRIES]).select(pl.col("entity_id").alias("s1_id"))
    uni = s1v.with_columns((pl.col("s1_id").hash(21) % 2).cast(pl.Int8).alias("half"))
    truth = t.labels().join(uni.select("s1_id"), on="s1_id").select("s1_id", "m_id")
    out = T / "stage1_results.jsonl"
    seen = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rng = random.Random(a.seed)
    cfgs = [BASELINE] + [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(a.n)]
    for n, cfg in enumerate(cfgs):
        cid = cfg_id(cfg)
        if cid in seen:
            continue
        t0 = time.time()
        m = lgb.train({**t.PARAMS, **cfg}, dtr, num_boost_round=6000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(Xv, num_threads=t.THREADS)
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        ll = float(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))
        sc = ids.with_columns(pl.Series("p", p)).filter(pl.col("p") >= 0.05)
        res = {thr: macro_by_half(sc, truth, uni, thr) for thr in THRESHOLDS}
        best_t = max(THRESHOLDS, key=lambda x: res[x][0])
        rec = dict(id=cid, cfg=cfg, iters=m.best_iteration, secs=round(time.time() - t0), valid_logloss=round(ll, 6),
                   best_t_on_A=best_t, A=round(res[best_t][0], 5), B_at_A_t=round(res[best_t][1], 5),
                   baseline=(cfg == BASELINE))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"[{n + 1}/{len(cfgs)}] A {rec['A']:.5f} B {rec['B_at_A_t']:.5f} logloss {ll:.5f} t={best_t} "
            f"iters {m.best_iteration} {rec['secs']}s  {cfg}")
    rs = [json.loads(l) for l in out.read_text().splitlines()]
    log("TOP 5 by half A (B = confirmation on the other half):")
    for r in sorted(rs, key=lambda r: -r["A"])[:5]:
        log(f"  A {r['A']:.5f}  B {r['B_at_A_t']:.5f}  logloss {r['valid_logloss']:.5f}  {r['cfg']}")
    base = next((r for r in rs if r["baseline"]), None)
    if base:
        log(f"  baseline: A {base['A']:.5f}  B {base['B_at_A_t']:.5f}  logloss {base['valid_logloss']:.5f}")


if __name__ == "__main__":
    main()
