"""Phase 1 training + full-universe validation (train split).

Stages (each checkpointed under work/p1/model/; re-run skips finished stages; nothing auto-resumes):
  roles   : pick FIT and VALID states per country (sorted by size, alternating, so both include big crowded states).
            ER_FIT_MODE=entities: FIT = ER_FIT_PCT% of S1 entities from EVERY non-VALID state (incl. stateless S1),
            which gives the model all regional patterns instead of 1-2 states. VALID is unchanged either way.
  feats   : full-universe features for FIT and VALID S1 (p1_features.save_features)
  stage1  : 2-fold LightGBM ensemble on FIT (folds split by S1 entity: FIT holds only 1-2 big states per
            country, too few for state folds); out-of-fold p1 for FIT
  stage2  : stage-2 context features from p1 (over all pairs), trained on FIT pairs with p1 >= P1_MIN
  valid   : score VALID (p1 = fold average, same as test), decisions tuned on VALID, entity model cross-fitted by state
Usage: caffeinate -i -s -m python p1_train.py --stages roles,feats,stage1,stage2,valid
Env: ER_MODEL_DIR (model folder), ER_THREADS (default 7), ER_FIT_MODE (states|entities), ER_FIT_PCT (default 10),
     ER_SIBLINGS (default 1: stage-2 sibling evidence), ER_FAST_PREDICT (1: lleaves compiled prediction)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl

from pathlib import Path

from utils import WORK, announce_pid, done, log, resource_guard, rss_gb, timed, write_parquet_atomic

M = Path(os.environ.get("ER_MODEL_DIR", str(WORK / "p1" / "model")))
COUNTRIES = ["US", "India"]
FIT_FRAC, VALID_FRAC = 0.10, 0.06
THREADS = int(os.environ.get("ER_THREADS", "7"))
FIT_MODE = os.environ.get("ER_FIT_MODE", "states")          # states | entities
FIT_PCT = int(os.environ.get("ER_FIT_PCT", "10"))            # entities mode: % of S1 entities per state
FIT_DIR = "fit" if FIT_MODE == "states" else f"fit_ent{FIT_PCT}"   # feature folder name of the FIT role
SIBLINGS = os.environ.get("ER_SIBLINGS", "1") == "1"         # stage-2 sibling evidence (v1 was trained with 0)
# Stage-2 training on WHOLE states (v5): stage 2 uses "competition" features (how many other S1s want this record).
# If its training S1s are an entity sample, each record sees ~2 competitors instead of ~13 as at test time
# (measured: 2.13 vs 12.81). With ER_S2_STATES set, stage 1 is trained WITHOUT these states and stage 2 is trained
# on all pairs of these states (complete competition), using p1 from stage-1 models that never saw them.
S2_STATES = [x for x in os.environ.get("ER_S2_STATES", "").split(",") if x]
P1_MIN = 0.005
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=7,
              deterministic=True, force_row_wise=True)
S1_PARAMS = {**PARAMS, **json.loads(os.environ.get("ER_S1_PARAMS", "{}"))}   # tuned stage-1 knobs (JSON)
S1_SINGLE = os.environ.get("ER_S1_SINGLE") == "1"     # one stage-1 model on all FIT rows (needs ER_S2_STATES)
S1_PATIENCE = int(os.environ.get("ER_S1_PATIENCE", "50"))   # early-stopping patience; low learning rates need more
S1_MMAP = os.environ.get("ER_S1_MMAP") == "1"          # single mode: stream FIT parts to .npy and bin from memory maps
GBM = os.environ.get("ER_GBM", "lgb")                   # lgb | xgb (XGBoost track: stage 1 memory-mapped single model + stage 2)
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", grow_policy="lossguide",
                  max_depth=0, max_leaves=127, eta=0.1, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, max_bin=256, nthread=THREADS, seed=7)
XGB_S1_PARAMS = {**XGB_PARAMS, **json.loads(os.environ.get("ER_XGB_S1_PARAMS", "{}"))}


def model_file(name, d=None):
    """Saved booster of this model folder: <name>.json (XGBoost) or <name>.txt (LightGBM)."""
    d = d or M
    return d / f"{name}.json" if (d / f"{name}.json").exists() else d / f"{name}.txt"


def merged_state(col="addr_state"):
    return pl.col(col).replace("ts", "ap")


def stage_roles(force):
    out = M / "roles.parquet"
    if done(out, force):
        log("skip roles"); return
    rows = []
    for c in COUNTRIES:
        s1 = pl.read_parquet(WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state"])
        sizes = (s1.with_columns(merged_state().alias("st")).filter(pl.col("st") != "")
                   .group_by("st").len().sort("len", descending=True))
        total = s1.height
        fit, val, nf, nv = [], [], 0, 0
        for k, (st, n) in enumerate(sizes.iter_rows()):
            # alternate: largest -> FIT, 2nd largest -> VALID, then fill each to its target fraction
            want_val = (k % 2 == 1)
            if want_val and nv < VALID_FRAC * total and n < 1.6 * VALID_FRAC * total:
                val.append(st); nv += n
            elif nf < FIT_FRAC * total and n < 1.2 * FIT_FRAC * total:
                fit.append(st); nf += n
        log(f"{c}: FIT states {fit} ({nf} S1, {nf / total:.1%}) | VALID states {val} ({nv} S1, {nv / total:.1%})")
        rows += [(c, st, "fit") for st in fit] + [(c, st, "valid") for st in val]
    write_parquet_atomic(pl.DataFrame(rows, schema=["country", "st", "role"], orient="row"), out)


def role_mask(country, role):
    r = pl.read_parquet(M / "roles.parquet").filter(pl.col("country") == country)
    if role == "s2fit":                   # whole states reserved for stage-2 training
        return merged_state().is_in(S2_STATES)
    if role == "fit" and FIT_MODE == "entities":
        out = list(r.filter(pl.col("role") == "valid")["st"]) + S2_STATES
        return (~merged_state().fill_null("").is_in(out)) & (pl.col("entity_id").hash(5) % 100 < FIT_PCT)
    r = r.filter(pl.col("role") == role)
    return merged_state().is_in(r["st"].implode())


def labels():
    gt = pl.read_parquet(WORK / "parquet" / "train_ground_truth.parquet").filter(pl.col("matched_entity_ids") != "")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m")
              .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id"), pl.lit(1, pl.Int8).alias("y")))


def stage_feats(force):
    from p1_features import save_features
    lab = labels()
    for c in COUNTRIES:
        for role in ["fit", "valid"] + (["s2fit"] if S2_STATES else []):
            dname = {"fit": FIT_DIR, "s2fit": "fit"}.get(role, role)
            d = WORK / "p1" / "train" / c / f"feats_{dname}"
            if (d / "DONE").exists() and not force:
                log(f"skip feats {c}/{dname}"); continue
            resource_guard(f"feats {c}/{dname}")
            with timed(f"features train/{c}/{dname}"):
                save_features("train", c, dname, role_mask(c, role), labels=lab)
            (d / "DONE").write_text("ok")


def load_role(role):
    return pl.concat([pl.scan_parquet(str(WORK / "p1" / "train" / c / f"feats_{role}" / "part_*.parquet"))
                      .with_columns(pl.lit(c).alias("country")) for c in COUNTRIES], how="diagonal").collect()


def role_parts(role):
    role = {"fit": FIT_DIR, "s2fit": "fit"}.get(role, role)     # s2fit = whole FIT states (v1 feature folder)
    return [(c, p) for c in COUNTRIES for p in sorted((WORK / "p1" / "train" / c / f"feats_{role}").glob("part_*.parquet"))]


def load_role_numpy(role, cols, keep: pl.DataFrame = None, extra_cols=None):
    """Stream feature parts into ONE preallocated float32 matrix (no second full copy in RAM).

    keep: optional table [s1_id, m_id, ...extra_cols] to inner-join per part (e.g. stage-2 rows + s2 features).
    Returns X, meta (s1_id, m_id, y, s1_state_hash, s1_grp, country). s1_grp = S1 entity hash used for folds.
    """
    import pyarrow.parquet as pq
    parts = role_parts(role)
    extra_cols = extra_cols or []
    n = sum(pq.ParquetFile(p).metadata.num_rows for _, p in parts) if keep is None else None
    Xs, metas, pos = [], [], 0
    X = np.empty((n, len(cols) + len(extra_cols)), np.float32) if n is not None else None
    for c, p in parts:
        df = pl.read_parquet(p)
        if keep is not None:
            df = df.join(keep, on=["s1_id", "m_id"], how="inner")
        df = attach_state(df.with_columns(pl.lit(c).alias("country")))
        block = f32(df, cols + extra_cols)
        if X is not None:
            X[pos:pos + len(block)] = block
            pos += len(block)
        else:
            Xs.append(block)
        metas.append(df.select("s1_id", "m_id", "y", "s1_state_hash", "s1_st", "country")
                     .with_columns(pl.col("s1_id").hash(13).alias("s1_grp")))
        del df, block
    if X is None:
        X = np.concatenate(Xs) if Xs else np.empty((0, len(cols) + len(extra_cols)), np.float32)
    return X, pl.concat(metas)


class _NpySeq:
    """lightgbm.Sequence over one memory-mapped .npy block (rows are read in batches while binning)."""
    batch_size = 65536

    def __init__(self, path):
        self.a = np.load(path, mmap_mode="r")

    def __getitem__(self, idx):
        return np.asarray(self.a[idx], dtype=np.float64)     # the Sequence path bins from float64 batches

    def __len__(self):
        return self.a.shape[0]


def stream_role_npy(role, cols, drop_states=()):
    """Stage-1 rows streamed to WORK/tmp/s1mm as .npy blocks, split at write time into train and early-stopping rows
    (es = S1 entity hash(13) % 7 == 3, as in stage_stage1). Returns (train seqs, y_train), (es seqs, y_es); RAM holds
    only the labels, so no raw matrix, no string meta and no Dataset.subset copy."""
    import shutil
    import lightgbm as lgb
    seq_cls = type("NpySeq", (_NpySeq, lgb.Sequence), {})
    d = WORK / "tmp" / "s1mm"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    out = {"trn": ([], []), "es": ([], [])}
    for k, (c, p) in enumerate(role_parts(role)):
        df = attach_state(pl.read_parquet(p).with_columns(pl.lit(c).alias("country")))
        if drop_states:
            df = df.filter(~pl.col("s1_st").is_in(list(drop_states)))
        if df.height == 0:
            continue
        es = (df["s1_id"].hash(13) % 7 == 3).to_numpy()
        X, y = f32(df, cols), df["y"].to_numpy().astype(np.int8)
        del df
        for name, m in (("trn", ~es), ("es", es)):
            if m.any():
                f = d / f"{name}_{k:04d}.npy"
                np.save(f, X[m])
                out[name][0].append(seq_cls(f))
                out[name][1].append(y[m])
        del X
    return tuple((seqs, np.concatenate(ys)) for seqs, ys in (out["trn"], out["es"]))


def stage1_mmap(cols):
    """Single-model stage 1 on memory-mapped blocks (ER_S1_MMAP=1): bin train rows, bin es rows against them, train."""
    import shutil
    import lightgbm as lgb
    (tseq, ytr), (eseq, yes) = stream_role_npy("fit", cols, S2_STATES)
    if GBM == "xgb":
        return stage1_mmap_xgb(cols, tseq, ytr, eseq, yes)
    log(f"stage 1 memory-mapped ({len(tseq)} + {len(eseq)} blocks, stage-2 states {S2_STATES} excluded): train {len(ytr)} / es {len(yes)}, "
        f"positive rate {ytr.mean():.4f}, RSS {rss_gb():.1f} GB")
    with timed(f"stage-1 fold 0 (memory-mapped): train {len(ytr)} / es {len(yes)}"):
        dtr = lgb.Dataset(tseq, ytr, feature_name=cols).construct()
        des = lgb.Dataset(eseq, yes, feature_name=cols, reference=dtr).construct()
        tseq = eseq = None
        shutil.rmtree(WORK / "tmp" / "s1mm", ignore_errors=True)
        log(f"binned; blocks removed, RSS {rss_gb():.1f} GB")
        m = lgb.train(S1_PARAMS, dtr, num_boost_round=6000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(S1_PATIENCE, verbose=False), lgb.log_evaluation(200)])
    m.save_model(str(M / "stage1_fold0.txt"))
    log(f"  fold 0 best iter {m.best_iteration}")
    del m, dtr, des


def stage1_mmap_xgb(cols, tseq, ytr, eseq, yes):
    """XGBoost stage 1 on the same memory-mapped blocks: QuantileDMatrix from a block iterator (1 byte per value),
    early stopping on the es rows, booster cut at the best iteration and saved as stage1_fold0.json."""
    import shutil
    import xgboost as xgb

    class It(xgb.DataIter):
        def __init__(self, seqs, ys):
            self.seqs, self.ys, self.k = seqs, ys, 0
            self.off = np.concatenate([[0], np.cumsum([len(q) for q in seqs])])
            super().__init__()

        def next(self, input_data):
            if self.k == len(self.seqs):
                return 0
            a = np.asarray(self.seqs[self.k].a, dtype=np.float32)
            input_data(data=a, label=self.ys[self.off[self.k]:self.off[self.k + 1]], feature_names=cols)
            self.k += 1
            return 1

        def reset(self):
            self.k = 0

    log(f"XGBoost stage 1 memory-mapped ({len(tseq)} + {len(eseq)} blocks): train {len(ytr)} / es {len(yes)}, params {XGB_S1_PARAMS}")
    with timed(f"XGBoost stage-1 fold 0: train {len(ytr)} / es {len(yes)}"):
        dtr = xgb.QuantileDMatrix(It(tseq, ytr), max_bin=XGB_S1_PARAMS["max_bin"])
        des = xgb.QuantileDMatrix(It(eseq, yes), ref=dtr)
        shutil.rmtree(WORK / "tmp" / "s1mm", ignore_errors=True)
        log(f"quantised; blocks removed, RSS {rss_gb():.1f} GB")
        m = xgb.train(XGB_S1_PARAMS, dtr, num_boost_round=6000, evals=[(des, "es")], early_stopping_rounds=S1_PATIENCE,
                      verbose_eval=200)
    best = m.best_iteration
    m = m[: best + 1]
    m.save_model(str(M / "stage1_fold0.json"))
    log(f"  fold 0 best iter {best}")


def feat_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


def f32(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def attach_state(df):
    s = pl.concat([pl.read_parquet(WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state"])
                   .with_columns(pl.lit(c).alias("country")) for c in COUNTRIES])
    s = s.select(pl.col("entity_id").alias("s1_id"), merged_state().alias("s1_st"),
                 (pl.col("country") + pl.lit(":") + merged_state()).hash(11).alias("s1_state_hash"))
    return df.join(s, on="s1_id", how="left")


def compact_rows(X, keep, block=1_000_000):
    """X[keep] without a second full copy: kept rows move down in place (sources always lie at or after targets)."""
    idx = np.flatnonzero(keep)
    for s in range(0, len(idx), block):
        j = idx[s:s + block]
        X[s:s + len(j)] = X[j]
    return X[:len(idx)]


def stage_stage1(force):
    import lightgbm as lgb
    if done(model_file("stage1_fold0") if S1_SINGLE else M / "stage1_fold1.txt", force):
        log("skip stage1"); return
    cols = feat_cols(pl.read_parquet(role_parts("fit")[0][1], n_rows=5))
    (M / "feature_cols.json").write_text(json.dumps(cols))
    if S1_MMAP and S1_SINGLE:              # memory-mapped path: no fit_oof_p1 (unused with ER_S2_STATES), then s2fit p1
        stage1_mmap(cols)
        X2, m2 = load_role_numpy("s2fit", cols)
        write_parquet_atomic(m2.select("s1_id", "m_id", "y").with_columns(pl.Series("p1", predict_p1(X2).astype(np.float32))),
                             M / "s2fit_p1.parquet")
        log(f"stage-2 training states {S2_STATES}: p1 for {len(m2)} pairs")
        return
    X, meta = load_role_numpy("fit", cols)
    if S2_STATES:                          # these states are reserved for stage-2 training
        keep = ~meta["s1_st"].is_in(S2_STATES).to_numpy()
        if not keep.all():
            X, meta = compact_rows(X, keep), meta.filter(pl.Series(keep))
        log(f"stage 1 excludes stage-2 states {S2_STATES}: {len(meta)} rows left")
    y = meta["y"].to_numpy()
    grp = meta["s1_grp"].to_numpy()
    fold = grp % 2                    # all pairs of one S1 entity stay in the same fold
    log(f"FIT: {(len(y), len(cols))} pairs x feats, positive rate {y.mean():.4f}, RSS {rss_gb():.1f} GB")
    oof = np.zeros(len(y), np.float32)
    for k in ((0,) if S1_SINGLE else (0, 1)):
        # single mode: one model on everything except the early-stopping slice (no out-of-fold scores needed,
        # stage 2 is trained on the reserved whole states)
        tr, te = (np.ones(len(y), bool), np.zeros(len(y), bool)) if S1_SINGLE else (fold != k, fold == k)
        # early stopping on a slice of the training fold's own entities (no peeking at the other fold)
        es = tr & (grp % 7 == 3)
        trn = tr & ~es
        with timed(f"stage-1 fold {k}: train {trn.sum()} / es {es.sum()} / oof {te.sum()}"):
            if S1_SINGLE:   # bin once, release the raw matrix, then train on index subsets
                full = lgb.Dataset(X, y, feature_name=cols).construct()
                X = None
                dtr, des = full.subset(np.where(trn)[0]), full.subset(np.where(es)[0])
                log(f"binned; raw matrix released, RSS {rss_gb():.1f} GB")
            else:
                dtr, des = lgb.Dataset(X[trn], y[trn], feature_name=cols), lgb.Dataset(X[es], y[es], feature_name=cols)
            m = lgb.train(S1_PARAMS, dtr, num_boost_round=6000, valid_sets=[des],
                          callbacks=[lgb.early_stopping(S1_PATIENCE, verbose=False), lgb.log_evaluation(200)])
        if te.any():
            oof[te] = m.predict(X[te], num_threads=THREADS)
        m.save_model(str(M / f"stage1_fold{k}.txt"))
        log(f"  fold {k} best iter {m.best_iteration}")
    write_parquet_atomic(meta.select("s1_id", "m_id", "y").with_columns(pl.Series("p1", oof)), M / "fit_oof_p1.parquet")
    if S2_STATES:                          # p1 for every pair of the whole stage-2 states, as at test time
        del X
        X2, m2 = load_role_numpy("s2fit", cols)
        write_parquet_atomic(m2.select("s1_id", "m_id", "y").with_columns(pl.Series("p1", predict_p1(X2).astype(np.float32))),
                             M / "s2fit_p1.parquet")
        log(f"stage-2 training states {S2_STATES}: p1 for {len(m2)} pairs")


def train_texts():
    from stack import pool_texts
    return pool_texts([WORK / "p1" / "train" / c / "pool.parquet" for c in COUNTRIES])


def stage2_design(df_feats, p1_table, cols):
    from stack import stage2_features
    s2 = stage2_features(p1_table)                         # context over ALL pairs of the role
    s2 = s2.filter(pl.col("p1") >= P1_MIN)                 # stage-2 input set (= candidate_pairs at test time)
    d = s2.join(df_feats, on=["s1_id", "m_id"], how="inner")
    s2cols = [c for c in s2.columns if c.startswith("s2_")] + ["p1"]
    return d, s2cols


def stage_stage2(force):
    import lightgbm as lgb
    if done(model_file("stage2"), force):
        log("skip stage2"); return
    from stack import SIB_COLS, sibling_features, stage2_features
    cols = json.loads((M / "feature_cols.json").read_text())
    p1 = pl.read_parquet(M / ("s2fit_p1.parquet" if S2_STATES else "fit_oof_p1.parquet"))
    lost = p1.filter((pl.col("y") == 1) & (pl.col("p1") < P1_MIN)).height
    log(f"stage-2 filter p1>={P1_MIN}: drops {lost} of {p1['y'].sum()} true FIT pairs ({lost / p1['y'].sum():.4%})")
    s2 = stage2_features(p1.select("s1_id", "m_id", "p1")).filter(pl.col("p1") >= P1_MIN)
    s2cols = [c for c in s2.columns if c.startswith("s2_")] + ["p1"]
    if SIBLINGS:
        s2 = s2.join(sibling_features(p1.select("s1_id", "m_id", "p1"), train_texts(), P1_MIN), on=["s1_id", "m_id"], how="left")
        s2cols += SIB_COLS
    allc = cols + s2cols
    (M / "stage2_cols.json").write_text(json.dumps(allc))
    X, meta = load_role_numpy("s2fit" if S2_STATES else "fit", cols, keep=s2, extra_cols=s2cols)
    y = meta["y"].to_numpy()
    es = meta["s1_grp"].to_numpy() % 7 == 3
    if GBM == "xgb":
        import xgboost as xgb
        with timed(f"XGBoost stage-2 train on {len(y)} pairs ({y.mean():.3f} positive)"):
            m = xgb.train(XGB_PARAMS, xgb.DMatrix(X[~es], y[~es], feature_names=allc), num_boost_round=3000,
                          evals=[(xgb.DMatrix(X[es], y[es], feature_names=allc), "es")], early_stopping_rounds=50, verbose_eval=200)
        best = m.best_iteration
        m = m[: best + 1]
        m.save_model(str(M / "stage2.json"))
        imp = sorted(m.get_score(importance_type="gain").items(), key=lambda x: -x[1])
        log(f"stage-2 best iter {best}; top: {[c for c, _ in imp[:12]]}")
        return
    with timed(f"stage-2 train on {len(y)} pairs ({y.mean():.3f} positive)"):
        m = lgb.train(PARAMS, lgb.Dataset(X[~es], y[~es], feature_name=allc), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X[es], y[es], feature_name=allc)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    m.save_model(str(M / "stage2.txt"))
    imp = sorted(zip(allc, m.feature_importance("gain")), key=lambda x: -x[1])
    log(f"stage-2 best iter {m.best_iteration}; top: {[c for c, _ in imp[:12]]}")


def load_predictor(path, fast=None):
    """Returns f(X) -> probabilities for a saved LightGBM model.

    With ER_FAST_PREDICT=1 and lleaves installed, the trees are compiled to native code once (cached next to the
    model, keyed by the model's checksum). Outputs match LightGBM to ~1e-15 and scoring runs ~10x faster.
    Otherwise plain LightGBM prediction is used. fast=False forces LightGBM (worth it for small row counts, where
    the one-off compile would take longer than predicting)."""
    if str(path).endswith(".json"):          # XGBoost booster (saved already cut at its best iteration)
        import xgboost as xgb
        bx = xgb.Booster(model_file=str(path))
        bx.set_param({"nthread": THREADS})
        return lambda X: bx.inplace_predict(np.ascontiguousarray(X, dtype=np.float32))
    import lightgbm as lgb
    if fast is None:
        fast = os.environ.get("ER_FAST_PREDICT") == "1"
    if fast:
        try:
            import lleaves
        except ImportError:
            log("lleaves not installed: using LightGBM prediction")
        else:
            import hashlib
            import platform
            tag = hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]
            m = lleaves.Model(model_file=str(path))
            m.compile(cache=f"{path}.{tag}.{platform.system()}-{platform.machine()}.o")
            return lambda X: m.predict(np.ascontiguousarray(X, dtype=np.float64), n_jobs=THREADS)
    b = lgb.Booster(model_file=str(path))
    return lambda X: b.predict(X, num_threads=THREADS)


_P1_MODELS = []


def predict_p1(X):
    """Stage-1 probability = average of the two fold models (loaded once per process)."""
    if not _P1_MODELS:
        _P1_MODELS.extend(load_predictor(p) for p in sorted(M.glob("stage1_fold?.txt")) + sorted(M.glob("stage1_fold?.json")))
    return np.mean([f(X) for f in _P1_MODELS], axis=0)


def stage_valid(force):
    import lightgbm as lgb
    from decide import by_expected_f_entity, by_threshold, crossfit_entity_prob, entity_table, one_s1_per_record, to_sets
    from metric import entity_f05
    out = M / "valid_report.json"
    if done(out, force):
        log("skip valid"); return
    from stack import stage2_features
    cols = json.loads((M / "feature_cols.json").read_text())
    allc = json.loads((M / "stage2_cols.json").read_text())
    s2cols = allc[len(cols):]
    X, meta = load_role_numpy("valid", cols)
    p1 = predict_p1(X)
    meta = meta.with_row_index("row").with_columns(pl.Series("p1", p1))
    s2 = stage2_features(meta.select("row", "s1_id", "m_id", "p1")).filter(pl.col("p1") >= P1_MIN)
    if any(c.startswith("s3_sib") for c in s2cols):
        from stack import sibling_features
        s2 = s2.join(sibling_features(meta.select("s1_id", "m_id", "p1"), train_texts(), P1_MIN), on=["s1_id", "m_id"], how="left")
    s2 = s2.sort("row")
    rows = s2["row"].to_numpy()
    X2 = np.hstack([X[rows], s2.select([pl.col(c).cast(pl.Float32) for c in s2cols]).to_numpy()])
    m2 = load_predictor(model_file("stage2"), fast=False)
    sc = meta[rows].select("s1_id", "m_id", "y", "s1_state_hash", "s1_grp").with_columns(pl.Series("p", m2(X2)))
    ent_feats = [c for c in ["f_cos_comb", "f_cos_name", "f_cos_addr", "f_n_same_name_s1", "f_num_share"] if c in cols]
    d = meta[rows].select("s1_id", "m_id").with_columns([pl.Series(c, X[rows, cols.index(c)]) for c in ent_feats])
    del X, X2
    write_parquet_atomic(sc, M / "valid_scored.parquet")
    # evaluation universe: every VALID S1 (including those with no candidates)
    s1v = pl.concat([pl.read_parquet(WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state", "country"])
                     .filter(role_mask(c, "valid")) for c in COUNTRIES])
    gt = labels().join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    truth = to_sets(gt)
    lost = gt.join(sc.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti").height
    log(f"VALID: {s1v.height} S1, {gt.height} true pairs; not reaching stage 2 (blocking+filter): {lost} ({lost / gt.height:.3%})")

    def score(pred):
        P = to_sets(pred)
        f = [entity_f05(P.get(s, set()), truth.get(s, set())) for s in s1v["entity_id"].to_list()]
        df = s1v.with_columns(pl.Series("f", f))
        return {"macro": round(float(np.mean(f)), 5),
                **{f"f05_{c}": round(df.filter(pl.col("country") == c)["f"].mean(), 5) for c in COUNTRIES},
                "singleton_acc": round(df.join(gt.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="anti")["f"].mean(), 4)}

    sc1 = one_s1_per_record(sc)
    rep = {"threshold": {}, "entity": {}}
    for t in np.arange(0.3, 0.91, 0.05).round(2):
        rep["threshold"][float(t)] = score(by_threshold(sc1, t))
    has = {s: s in truth for s in s1v["entity_id"].to_list()}
    ent = entity_table(sc1, d)
    ent = ent.join(sc.select("s1_id", (pl.col("s1_grp") % 2).alias("fold")).unique("s1_id"), on="s1_id", how="left")
    pm = crossfit_entity_prob(ent, has)   # folds = S1 entities (VALID has 1 state per country, too few for state folds)
    for mu in [0.0, 0.1, 0.3]:
        rep["entity"][mu] = score(by_expected_f_entity(sc1, pm, mu))
    best_t = max(rep["threshold"], key=lambda k: rep["threshold"][k]["macro"])
    best_mu = max(rep["entity"], key=lambda k: rep["entity"][k]["macro"])
    rep["best"] = {"threshold": [best_t, rep["threshold"][best_t]], "entity": [best_mu, rep["entity"][best_mu]]}
    rep["valid_prior"] = float(sc["y"].mean())
    out.write_text(json.dumps(rep, indent=1))
    log(f"BEST threshold t={best_t}: {rep['threshold'][best_t]}")
    log(f"BEST entity mu={best_mu}: {rep['entity'][best_mu]}")


STAGES = {"roles": stage_roles, "feats": stage_feats, "stage1": stage_stage1, "stage2": stage_stage2, "valid": stage_valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="roles,feats,stage1,stage2,valid")
    ap.add_argument("--force", default="")
    a = ap.parse_args()
    M.mkdir(parents=True, exist_ok=True)
    stages, force = a.stages.split(","), set(filter(None, a.force.split(",")))
    if len(stages) > 1:   # one fresh process per stage (OpenMP runtimes must not mix)
        for st in stages:
            r = subprocess.run([sys.executable, __file__, "--stages", st] + (["--force", st] if st in force else []))
            if r.returncode:
                log(f"stage {st} FAILED ({r.returncode}); stopping"); sys.exit(r.returncode)
        return
    announce_pid(f"p1_train[{stages[0]}]")
    resource_guard(stages[0])
    STAGES[stages[0]](stages[0] in force)


if __name__ == "__main__":
    main()
