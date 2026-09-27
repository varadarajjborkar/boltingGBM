"""XGBoost stage-2 hyperparameter sweep that reuses a finished stage 1 (ER_MODEL_DIR, ER_GBM=xgb).
Builds the stage-2 train design (as p1_train.stage_stage2) and the VALID stage-2 design (as stage_valid, cached in
<model>/s2sweep/), then for each config trains stage 2, scores VALID (one S1 per record, threshold grid) and reports
macro F0.5 with S1-hash halves. --save NAME writes the best config's booster to <model>_<NAME>/stage2.json (with the
stage-1 files linked) for p1_score.py --stage rescore --alt.
Usage (worker, cwd src, same env as the model's queue): python xgb_s2_sweep.py [--only c1,c3] [--save t]"""
import argparse
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

import p1_train as P
from decide import by_threshold, one_s1_per_record, to_sets
from metric import entity_f05
from stack import SIB_COLS, sibling_features, stage2_features

B = dict(P.XGB_PARAMS)
CONFIGS = {
    "base": {},
    "c1": {"eta": 0.05},
    "c2": {"eta": 0.05, "max_leaves": 255, "min_child_weight": 5},
    "c3": {"eta": 0.05, "max_leaves": 63},
    "c4": {"eta": 0.05, "colsample_bytree": 0.5},
    "c5": {"eta": 0.05, "reg_lambda": 5.0, "min_child_weight": 3},
    "c6": {"eta": 0.03, "max_leaves": 255, "colsample_bytree": 0.6, "subsample": 0.7},
    "c7": {"eta": 0.05, "grow_policy": "depthwise", "max_depth": 8, "max_leaves": 0},
    "c8": {"eta": 0.05, "max_leaves": 511, "min_child_weight": 10},
    "c9": {"eta": 0.05, "max_leaves": 127, "gamma": 0.5, "reg_alpha": 1.0},
}
TS = [0.6, 0.65, 0.7, 0.75, 0.8]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def train_design(cols, s2cols):
    p1 = pl.read_parquet(P.M / "s2fit_p1.parquet")
    s2 = stage2_features(p1.select("s1_id", "m_id", "p1")).filter(pl.col("p1") >= P.P1_MIN)
    if any(c in SIB_COLS for c in s2cols):
        s2 = s2.join(sibling_features(p1.select("s1_id", "m_id", "p1"), P.train_texts(), P.P1_MIN), on=["s1_id", "m_id"], how="left")
    X, meta = P.load_role_numpy("s2fit", cols, keep=s2, extra_cols=s2cols)
    return X, meta["y"].to_numpy(), meta["s1_grp"].to_numpy() % 7 == 3


def valid_design(cols, s2cols, C):
    if (C / "X2v.npy").exists():
        return np.load(C / "X2v.npy"), pl.read_parquet(C / "valid_meta.parquet")
    X, meta = P.load_role_numpy("valid", cols)
    meta = meta.with_row_index("row").with_columns(pl.Series("p1", P.predict_p1(X)))
    s2 = stage2_features(meta.select("row", "s1_id", "m_id", "p1")).filter(pl.col("p1") >= P.P1_MIN)
    if any(c in SIB_COLS for c in s2cols):
        s2 = s2.join(sibling_features(meta.select("s1_id", "m_id", "p1"), P.train_texts(), P.P1_MIN), on=["s1_id", "m_id"], how="left")
    s2 = s2.sort("row")
    rows = s2["row"].to_numpy()
    X2 = np.hstack([X[rows], s2.select([pl.col(c).cast(pl.Float32) for c in s2cols]).to_numpy()]).astype(np.float32)
    vm = meta[rows].select("s1_id", "m_id", "y")
    np.save(C / "X2v.npy", X2)
    vm.write_parquet(C / "valid_meta.parquet")
    return X2, vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    C = P.M / "s2sweep"
    C.mkdir(exist_ok=True)
    cols = json.loads((P.M / "feature_cols.json").read_text())
    allc = json.loads((P.M / "stage2_cols.json").read_text())
    s2cols = allc[len(cols):]
    X2, vm = valid_design(cols, s2cols, C)
    log(f"VALID stage-2 design {X2.shape}")
    X, y, es = train_design(cols, s2cols)
    log(f"train stage-2 design {X.shape}, es {es.sum()}")
    dtr = xgb.DMatrix(X[~es], y[~es], feature_names=allc, nthread=P.THREADS)
    des = xgb.DMatrix(X[es], y[es], feature_names=allc, nthread=P.THREADS)
    dv = xgb.DMatrix(X2, feature_names=allc, nthread=P.THREADS)
    del X
    s1v = pl.concat([pl.read_parquet(P.WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state", "country"])
                     .filter(P.role_mask(c, "valid")) for c in P.COUNTRIES])
    gt = P.labels().join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    truth = to_sets(gt)
    ids = s1v["entity_id"].to_list()
    half = (s1v["entity_id"].hash(13) % 2).to_numpy()
    ctry = s1v["country"].to_numpy()

    def score(pred):
        Pd = to_sets(pred)
        f = np.array([entity_f05(Pd.get(s, set()), truth.get(s, set())) for s in ids])
        return {"macro": f.mean(), "h0": f[half == 0].mean(), "h1": f[half == 1].mean(),
                "US": f[ctry == "US"].mean(), "India": f[ctry == "India"].mean()}

    res = {}
    only = set(filter(None, a.only.split(",")))
    for name, cfg in CONFIGS.items():
        if only and name not in only:
            continue
        prm = {**B, **cfg, "nthread": P.THREADS}
        t0 = time.time()
        m = xgb.train(prm, dtr, num_boost_round=4000, evals=[(des, "es")],
                      early_stopping_rounds=100 if prm["eta"] < 0.1 else 50, verbose_eval=False)
        best = m.best_iteration
        m = m[: best + 1]
        sc1 = one_s1_per_record(vm.with_columns(pl.Series("p", m.predict(dv))))
        r = {t: score(by_threshold(sc1, t)) for t in TS}
        tb = max(r, key=lambda t: r[t]["macro"])
        res[name] = {"cfg": cfg, "iters": best + 1, "t_best": tb, **{k: round(float(v), 5) for k, v in r[tb].items()},
                     "t70": round(float(r[0.7]["macro"]), 5), "secs": round(time.time() - t0)}
        log(f"{name}: {json.dumps(res[name])}")
        m.save_model(str(C / f"stage2_{name}.json"))
        (C / "sweep.json").write_text(json.dumps(res, indent=1))
    if "base" in res:
        b = res["base"]
        for n, r in sorted(res.items(), key=lambda kv: -kv[1]["macro"]):
            log(f"{n:5s} macro {r['macro']:.5f} ({r['macro'] - b['macro']:+.5f}) h0 {r['h0'] - b['h0']:+.5f} h1 {r['h1'] - b['h1']:+.5f} "
                f"US {r['US']:.5f} India {r['India']:.5f} t {r['t_best']} iters {r['iters']}")
    if a.save and res:
        win = max(res, key=lambda n: res[n]["macro"])
        D = P.M.parent / f"{P.M.name}_{a.save}"
        D.mkdir(exist_ok=True)
        for f in ["stage1_fold0.json", "feature_cols.json", "stage2_cols.json", "roles.parquet"]:
            if (P.M / f).exists() and not (D / f).exists():
                os.symlink(P.M / f, D / f)
        (D / "stage2.json").write_bytes((C / f"stage2_{win}.json").read_bytes())
        log(f"saved {win} -> {D / 'stage2.json'}")


if __name__ == "__main__":
    main()
