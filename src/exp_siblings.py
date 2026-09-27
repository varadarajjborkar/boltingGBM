"""Experiment: "sibling" (collective) evidence for stage 2.

Idea: a noisy record m that closely resembles another record m' we are already confident belongs to the same S1
entity (p1(s, m') >= ANCHOR_P) probably belongs too, even when m itself looks weak against the clean S1 record.
S2/S3 duplicates of one business often share typos, abbreviations or a missing address with each other.

The features are built by stack.sibling_features (see its docstring). This script retrains ONLY stage 2 on the v1
stage-1 out-of-fold scores, with and without those features, and compares VALID macro F0.5.
Result (25 Sep): baseline 0.97213 -> with siblings 0.97393 at t=0.70 (adopted in p1_train / p1_score).

Run (v1 model, train split): python exp_siblings.py   -> prints VALID macro F0.5 of stage 2 with/without siblings.
"""
import json
import time

import numpy as np
import polars as pl

import p1_train as t
from stack import SIB_COLS, sibling_features, stage2_features


def main():
    import lightgbm as lgb
    from decide import by_threshold, one_s1_per_record, to_sets
    from metric import entity_f05
    t0 = time.time()
    cols = json.loads((t.M / "feature_cols.json").read_text())
    texts = t.train_texts()
    # ---- FIT: out-of-fold p1 (as used for the real stage 2)
    p1f = pl.read_parquet(t.M / "fit_oof_p1.parquet").select("s1_id", "m_id", "p1")
    s2f = stage2_features(p1f).filter(pl.col("p1") >= t.P1_MIN)
    sibf = sibling_features(p1f, texts, t.P1_MIN)
    s2f = s2f.join(sibf, on=["s1_id", "m_id"], how="left")
    s2cols = [c for c in s2f.columns if c.startswith("s2_")] + ["p1"]
    print(f"FIT stage-2 pairs {s2f.height}, sibling coverage {(s2f['s3_sib_n'] > 0).mean():.3f}  ({time.time() - t0:.0f}s)")
    Xf, mf = t.load_role_numpy("fit", cols, keep=s2f, extra_cols=s2cols + SIB_COLS)
    yf = mf["y"].to_numpy()
    es = mf["s1_grp"].to_numpy() % 7 == 3
    # ---- VALID: p1 for all pairs, then stage-2 set
    Xv, mv = t.load_role_numpy("valid", cols)
    p1v = t.predict_p1(Xv)
    p1tab = mv.select("s1_id", "m_id").with_columns(pl.Series("p1", p1v.astype(np.float32)))
    s2v = stage2_features(p1tab).filter(pl.col("p1") >= t.P1_MIN).join(sibling_features(p1tab, texts, t.P1_MIN), on=["s1_id", "m_id"], how="left")
    del Xv
    Xv2, mv2 = t.load_role_numpy("valid", cols, keep=s2v, extra_cols=s2cols + SIB_COLS)
    print(f"VALID stage-2 pairs {len(Xv2)}  ({time.time() - t0:.0f}s)")
    s1v = pl.concat([pl.read_parquet(t.WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state", "country"])
                     .filter(t.role_mask(c, "valid")) for c in t.COUNTRIES])
    truth = to_sets(t.labels().join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id"))
    ids = s1v["entity_id"].to_list()
    base_n = len(cols) + len(s2cols)
    for name, ncol in [("baseline (no siblings)", base_n), ("with siblings", base_n + len(SIB_COLS))]:
        m = lgb.train(t.PARAMS, lgb.Dataset(Xf[~es, :ncol], yf[~es]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(Xf[es, :ncol], yf[es])], callbacks=[lgb.early_stopping(50, verbose=False)])
        sc = mv2.select("s1_id", "m_id").with_columns(pl.Series("p", m.predict(Xv2[:, :ncol], num_threads=t.THREADS)))
        sc1 = one_s1_per_record(sc)
        res = {}
        for th in [0.6, 0.65, 0.7, 0.75, 0.8]:
            P = to_sets(by_threshold(sc1, th))
            res[th] = round(float(np.mean([entity_f05(P.get(s, set()), truth.get(s, set())) for s in ids])), 5)
        print(f"{name:24s} iters {m.best_iteration:4d}  macro by threshold {res}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
