"""Phase 0 loop on the regional dev samples: block -> feats -> train -> eval.

Each stage writes to work/phase0/<tag>/ and is skipped if its output exists (use --force STAGE to redo).
Nothing restarts automatically. Stop with Ctrl-C / kill <pid>; re-run to continue from the last finished stage.
Usage: python run_phase0.py --tag v1 [--stages block,feats,train,eval] [--force feats,train]
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import polars as pl

from blocking import BlockCfg, BlockCfgV2, block, block_v2
from decide import (by_expected_f, by_expected_f_entity, by_threshold, crossfit_entity_prob, entity_table,
                    one_s1_per_record, to_sets)
from features import add_m_context, build_features, name_freq_tables, token_idf
from metric import entity_f05, oracle_f05
from utils import WORK, announce_pid, done, log, resource_guard, rss_gb, timed, write_parquet_atomic

BLOCKER = "v1"
USE_FREQ = True
FREQ = None
SETS = ["dev_train", "dev_valid"]   # dev_valid_dense dropped after review #1 (wrong distractor profile)


def load_set(name):
    d = WORK / "phase0" / name
    return pl.read_parquet(d / "s1.parquet"), pl.read_parquet(d / "pool.parquet"), pl.read_parquet(d / "gt.parquet")


def truth_sets(gt):
    return to_sets(gt.rename({"s1_id": "s1_id", "m_id": "m_id"}))


def tied_distractors(s1, pool, gt):
    """k = unmatched pool records with the same state and same core name as the S1 (review #1: test has ~2x)."""
    un = pool.join(gt.select(pl.col("m_id").alias("entity_id")), on="entity_id", how="anti")
    k = (s1.select("entity_id", "addr_state", "name_core")
           .join(un.group_by(["addr_state", "name_core"]).len(), on=["addr_state", "name_core"], how="left")
           .select("entity_id", pl.col("len").fill_null(0).alias("k")))
    return dict(k.iter_rows())


def score_report(pred_pairs: pl.DataFrame, s1: pl.DataFrame, gt: pl.DataFrame, kmap=None) -> dict:
    truth, pred = truth_sets(gt), to_sets(pred_pairs)
    rows = []
    for sid, c in s1.select("entity_id", "country").iter_rows():
        t, p = truth.get(sid, set()), pred.get(sid, set())
        rows.append((c, len(t) == 0, entity_f05(p, t), (kmap or {}).get(sid, 0)))
    df = pl.DataFrame(rows, schema=["country", "singleton", "f", "k"], orient="row")
    df = df.with_columns((2.0 ** pl.col("k").clip(0, 3)).alias("w"))
    rep = {"macro_f05": round(df["f"].mean(), 5),
           "weighted_f05": round((df["f"] * df["w"]).sum() / df["w"].sum(), 5),
           "lookalike_group_f05": round(df.filter(pl.col("k") >= 1)["f"].mean(), 5),
           "lookalike_group_share": round((df["k"] >= 1).mean(), 4)}
    for c in df["country"].unique().sort().to_list():
        rep[f"f05_{c}"] = round(df.filter(pl.col("country") == c)["f"].mean(), 5)
    rep["singleton_acc"] = round(df.filter(pl.col("singleton"))["f"].mean(), 4)
    rep["nonsingleton_f05"] = round(df.filter(~pl.col("singleton"))["f"].mean(), 4)
    rep["pred_pairs"] = pred_pairs.height
    return rep


def stage_block(tag, force):
    for name in SETS:
        out = WORK / "phase0" / tag / name / "cands.parquet"
        if done(out, force):
            log(f"skip block {name}"); continue
        s1, pool, gt = load_set(name)
        with timed(f"block {name} ({s1.height} S1, {pool.height} pool) with {BLOCKER}"):
            c = block_v2(s1, pool, BlockCfgV2()) if BLOCKER == "v2" else block(s1, pool, BlockCfg())
        write_parquet_atomic(c, out)
        truth = truth_sets(gt)
        cand = to_sets(c)
        hit = c.join(gt, on=["s1_id", "m_id"], how="semi").height
        rep = {"pairs": c.height, "cands_per_s1": round(c.height / s1.height, 2), "pair_recall": round(hit / gt.height, 5),
               "oracle_f05": round(oracle_f05(cand, truth, s1["entity_id"].to_list()), 5)}
        for via in ["name", "phon", "addr", "comb", "rev", "cat", "acr", "sl_name", "sl_phon", "sl_cat"]:
            only = c.filter(pl.col("via") == via).join(gt, on=["s1_id", "m_id"], how="semi").height
            rep[f"true_found_only_by_{via}"] = only
        (out.parent / "block_report.json").write_text(json.dumps(rep, indent=1))
        log(f"block {name}: {rep}")


def stage_feats(tag, force):
    for name in SETS:
        out = WORK / "phase0" / tag / name / "feats.parquet"
        if done(out, force):
            log(f"skip feats {name}"); continue
        s1, pool, gt = load_set(name)
        c = pl.read_parquet(WORK / "phase0" / tag / name / "cands.parquet")
        idf = {}
        for country in s1["country"].unique().to_list():
            a = s1.filter(pl.col("country") == country); b = pool.filter(pl.col("country") == country)
            idf[country] = (token_idf(a["name_core"].to_list() + b["name_core"].to_list()),
                            token_idf(a["addr_text"].to_list() + b["addr_text"].to_list()))
        c = add_m_context(c)
        global FREQ
        FREQ = name_freq_tables(s1, pool) if USE_FREQ else None
        n_chunks = max(1, -(-c.height // 1_500_000))
        part_dir = out.parent / "feats_parts"
        part_dir.mkdir(exist_ok=True)
        yl = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
        with timed(f"features {name} ({c.height} pairs in {n_chunks} chunks)"):
            for b in range(n_chunks):
                part = part_dir / f"part_{b:03d}.parquet"
                if part.exists():
                    continue          # chunk-level checkpoint
                sub = c.filter((pl.col("s1_id").hash(1) % n_chunks) == b)
                f = build_features(sub, s1, pool, idf, freq_tables=FREQ)
                f = f.join(yl, on=["s1_id", "m_id"], how="left").with_columns(pl.col("y").fill_null(0))
                write_parquet_atomic(f, part)
                log(f"  chunk {b + 1}/{n_chunks}: {f.height} pairs, RSS {rss_gb():.1f} GB")
                del f, sub
        tmp = out.with_suffix(".parquet.tmp")
        pl.scan_parquet(str(part_dir / "part_*.parquet")).sink_parquet(tmp)
        import os, shutil
        os.replace(tmp, out)
        shutil.rmtree(part_dir)          # parts no longer needed (keeps disk tidy)


def feat_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


def stage_train(tag, force):
    import lightgbm as lgb  # imported only in this stage's own process (OpenMP clash with sparse_dot_topn)
    mpath = WORK / "phase0" / tag / "model.txt"
    if done(mpath, force):
        log("skip train"); return
    tr = pl.read_parquet(WORK / "phase0" / tag / "dev_train" / "feats.parquet")
    ids = tr["s1_id"].unique().sort()
    es_ids = ids.sample(fraction=0.15, seed=7)
    es_mask = tr["s1_id"].is_in(es_ids.implode())
    cols = feat_cols(tr)
    f32 = [pl.col(c).cast(pl.Float32) for c in cols]          # float32 straight from polars: half the RAM
    Xtr, ytr = tr.filter(~es_mask).select(f32).to_numpy(), tr.filter(~es_mask)["y"].to_numpy()
    Xes, yes = tr.filter(es_mask).select(f32).to_numpy(), tr.filter(es_mask)["y"].to_numpy()
    del tr
    log(f"train matrix {Xtr.shape} float32, RSS {rss_gb():.1f} GB")
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=7, verbose=-1, seed=7)
    ckpt = WORK / "phase0" / tag / "model_ckpt.txt"

    def save_ckpt(env):
        if env.iteration % 200 == 0 and env.iteration > 0:
            env.model.save_model(str(ckpt))

    with timed(f"train LightGBM on {len(ytr)} pairs ({ytr.mean():.3f} positive), early-stop on {len(yes)}"):
        m = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=cols), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(Xes, yes, feature_name=cols)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200), save_ckpt])
    m.save_model(str(mpath))
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
    (WORK / "phase0" / tag / "importance.json").write_text(json.dumps([(c, round(float(v), 1)) for c, v in imp], indent=1))
    log(f"best iter {m.best_iteration}; top features: {[c for c, _ in imp[:12]]}")


def odds_scale(sc, r):
    return sc.with_columns((pl.col("p") * r / (pl.col("p") * r + 1 - pl.col("p"))).alias("p"))


def evaluate(sc, f, s1, pool, gt, label):
    """Decision grids tuned on the look-alike-weighted F0.5 of dev_valid; returns a report dict."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    kmap = tied_distractors(s1, pool, gt)
    rep = {"pair_auc": round(roc_auc_score(sc["y"], sc["p"]), 5), "pair_ap": round(average_precision_score(sc["y"], sc["p"]), 5)}
    sc1 = one_s1_per_record(sc)
    key = "weighted_f05"
    grid_t = {float(t): score_report(by_threshold(sc1, t), s1, gt, kmap)[key] for t in np.arange(0.2, 0.96, 0.05).round(2)}
    t_best = max(grid_t, key=grid_t.get)
    grid_ef = {}
    for r in [1.0, 0.7, 0.5]:
        for mu in [0.0, 0.1, 0.3]:
            grid_ef[f"r={r},mu={mu}"] = score_report(by_expected_f(one_s1_per_record(odds_scale(sc, r)), mu), s1, gt, kmap)[key]
    ef_best = max(grid_ef, key=grid_ef.get)
    r_best, mu_best = [float(x.split("=")[1]) for x in ef_best.split(",")]
    rep["threshold_grid"], rep["expected_f_grid"] = grid_t, grid_ef
    rep["threshold"] = {"t": t_best, **score_report(by_threshold(sc1, t_best), s1, gt, kmap)}
    rep["expected_f"] = {"r": r_best, "mu": mu_best,
                         **score_report(by_expected_f(one_s1_per_record(odds_scale(sc, r_best)), mu_best), s1, gt, kmap)}
    truth = truth_sets(gt)
    has = {sid: len(truth.get(sid, ())) > 0 for sid in s1["entity_id"].to_list()}
    pm = crossfit_entity_prob(entity_table(sc1, f), has)
    from sklearn.metrics import roc_auc_score as _auc
    j = pm.join(pl.DataFrame({"s1_id": list(has), "y": [int(v) for v in has.values()]}), on="s1_id")
    rep["entity_auc"] = round(_auc(j["y"], j["p_match"]), 5)
    grid_e = {mu: score_report(by_expected_f_entity(sc1, pm, mu), s1, gt, kmap)[key] for mu in [0.0, 0.1, 0.3]}
    mu_e = max(grid_e, key=grid_e.get)
    rep["expected_f_entity"] = {"mu": mu_e, **score_report(by_expected_f_entity(sc1, pm, mu_e), s1, gt, kmap)}
    log(f"[{label}] auc={rep['pair_auc']} ap={rep['pair_ap']} entity_auc={rep['entity_auc']}")
    for k in ["threshold", "expected_f", "expected_f_entity"]:
        log(f"  [{label}] {k}: {rep[k]}")
    return rep


def f32_matrix(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def stage_eval(tag, force):
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(WORK / "phase0" / tag / "model.txt"))
    s1, pool, gt = load_set("dev_valid")
    f = pl.read_parquet(WORK / "phase0" / tag / "dev_valid" / "feats.parquet")
    p = m.predict(f32_matrix(f, feat_cols(f)), num_threads=7)
    sc = f.select("s1_id", "m_id", "y").with_columns(pl.Series("p", p))
    write_parquet_atomic(sc, WORK / "phase0" / tag / "dev_valid" / "scored.parquet")
    rep = evaluate(sc, f, s1, pool, gt, "stage1")
    (WORK / "phase0" / tag / "eval_report.json").write_text(json.dumps(rep, indent=1, default=str))


def stage_stack(tag, force):
    """Stage 2: out-of-fold stage-1 scores in context (stack.py), second LightGBM, evaluated like stage 1."""
    import lightgbm as lgb
    from stack import oof_predict, stage2_features
    out = WORK / "phase0" / tag / "eval_stack_report.json"
    if done(out, force):
        log("skip stack"); return
    m1 = lgb.Booster(model_file=str(WORK / "phase0" / tag / "model.txt"))
    rounds = m1.best_iteration if (m1.best_iteration or 0) > 0 else m1.num_trees()   # loaded models report -1
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=7, verbose=-1, seed=7)
    tr = pl.read_parquet(WORK / "phase0" / tag / "dev_train" / "feats.parquet")
    cols = feat_cols(tr)
    X, y = f32_matrix(tr, cols), tr["y"].to_numpy()
    groups = tr.select(pl.col("s1_id").hash(3)).to_series().to_numpy().astype(np.int64)
    with timed(f"stage-1 out-of-fold scores on dev_train ({len(y)} pairs, {rounds} rounds, 2 folds)"):
        oof = oof_predict(X, y, groups, params, rounds, n_folds=2)
    s2_tr = stage2_features(tr.select("s1_id", "m_id").with_columns(pl.Series("p1", oof)))
    s2cols = [c for c in s2_tr.columns if c.startswith("s2_")] + ["p1"]
    X2 = np.hstack([X, s2_tr.select([pl.col(c).cast(pl.Float32) for c in s2cols]).to_numpy()])
    del X, s2_tr
    es = (np.abs(groups) % 7) == 0
    with timed(f"train stage-2 LightGBM ({X2.shape[1]} features)"):
        m2 = lgb.train(params, lgb.Dataset(X2[~es], y[~es], feature_name=cols + s2cols), num_boost_round=2000,
                       valid_sets=[lgb.Dataset(X2[es], y[es], feature_name=cols + s2cols)],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    m2.save_model(str(WORK / "phase0" / tag / "model_stage2.txt"))
    imp = sorted(zip(cols + s2cols, m2.feature_importance("gain")), key=lambda x: -x[1])
    log(f"stage-2 best iter {m2.best_iteration}; top: {[c for c, _ in imp[:10]]}")
    del X2, tr
    s1, pool, gt = load_set("dev_valid")
    f = pl.read_parquet(WORK / "phase0" / tag / "dev_valid" / "feats.parquet")
    Xv = f32_matrix(f, cols)
    p1 = m1.predict(Xv, num_threads=7)
    s2_v = stage2_features(f.select("s1_id", "m_id").with_columns(pl.Series("p1", p1)))
    p2 = m2.predict(np.hstack([Xv, s2_v.select([pl.col(c).cast(pl.Float32) for c in s2cols]).to_numpy()]), num_threads=7)
    sc = f.select("s1_id", "m_id", "y").with_columns(pl.Series("p", p2))
    write_parquet_atomic(sc, WORK / "phase0" / tag / "dev_valid" / "scored_stage2.parquet")
    rep = evaluate(sc, f, s1, pool, gt, "stage2")
    out.write_text(json.dumps(rep, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--stages", default="block,feats,train,eval")
    ap.add_argument("--force", default="")
    ap.add_argument("--blocker", default="v1")
    a = ap.parse_args()
    global BLOCKER
    BLOCKER = a.blocker
    force = set(a.force.split(",")) if a.force else set()
    stages = a.stages.split(",")
    if len(stages) > 1:
        # each stage in a fresh process: separate OpenMP runtimes (LightGBM vs sparse_dot_topn) segfault together
        for st in stages:
            cmd = [sys.executable, __file__, "--tag", a.tag, "--stages", st, "--blocker", a.blocker] + (["--force", st] if st in force else [])
            r = subprocess.run(cmd)
            if r.returncode != 0:
                log(f"stage {st} FAILED with exit code {r.returncode}; stopping (re-run to continue from here)")
                sys.exit(r.returncode)
        return
    st = stages[0]
    announce_pid(f"run_phase0[{a.tag}:{st}]")
    resource_guard(st)
    {"block": stage_block, "feats": stage_feats, "train": stage_train, "eval": stage_eval,
     "stack": stage_stack}[st](a.tag, st in force)
    log(f"stage {st} finished, peak-ish RSS {rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
