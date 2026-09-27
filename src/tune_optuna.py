"""Stage-1 hyperparameter search with Optuna (TPE Bayesian search + early pruning), all knobs at once.

Why not a full grid: 11 knobs with even 3-4 values each is thousands of combinations (days of compute). TPE
proposes each new configuration from the results so far and concentrates on the promising region; pruning stops
weak configurations after a few hundred trees by comparing their early-stopping log loss with earlier trials.

Protocol:
  train  : a random TRAIN_FRAC share of the FIT fold-0 training rows (speed); early stopping on the fold's own
           entity slice (as in p1_train stage 1)
  score  : macro F0.5 of the stage-1 decision (one S1 per record + best threshold) on VALID half A (objective)
  confirm: VALID half B at the same threshold is stored as a user attribute (reported, never optimised)
Several processes (and machines) can run at once; processes on one machine share a journal file, so TPE learns
from all of them. Seed trials (baseline, random-search winners) are enqueued first.

Usage: ER_MODEL_DIR=.../model python tune_optuna.py --study s1_<machine> --minutes 90 --threads 4 --seed 1
"""
import argparse
import json
import time

import numpy as np
import polars as pl

import p1_train as t
from tune_stage2 import macro_by_half
from utils import announce_pid, log

T = t.M / "tune"
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
TRAIN_FRAC = 0.5
SEEDS = [dict(learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              lambda_l1=0.001, lambda_l2=1.0, min_gain_to_split=0.0, max_depth=-1, extra_trees=False,
              min_sum_hessian_in_leaf=0.001),
         dict(learning_rate=0.03, num_leaves=511, min_data_in_leaf=300, feature_fraction=0.7, bagging_fraction=0.8,
              lambda_l1=0.001, lambda_l2=0.001, min_gain_to_split=0.0, max_depth=-1, extra_trees=False,
              min_sum_hessian_in_leaf=0.001)]


def space(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 1023, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 50.0, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        max_depth=trial.suggest_categorical("max_depth", [-1, 8, 12, 16]),
        extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
        min_sum_hessian_in_leaf=trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True))


def main():
    import lightgbm as lgb
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--enqueue", action="store_true", help="add the seed configurations (do this in ONE process)")
    a = ap.parse_args()
    announce_pid(f"tune_optuna[{a.study}]")
    T.mkdir(exist_ok=True)
    cols = json.loads((t.M / "feature_cols.json").read_text())
    X, meta = t.load_role_numpy("fit", cols)
    y, grp = meta["y"].to_numpy(), meta["s1_grp"].to_numpy()
    tr = grp % 2 != 0
    es = tr & (grp % 7 == 3)
    rng = np.random.default_rng(123)                      # same training subsample in every process
    trn = tr & ~es & (rng.random(len(y)) < TRAIN_FRAC)
    dp = {"feature_pre_filter": False, "max_bin": 255, "verbose": -1}
    dtr = lgb.Dataset(X[trn], y[trn], params=dp).construct()
    des = lgb.Dataset(X[es], y[es], reference=dtr, params=dp).construct()
    del X
    Xv, mv = t.load_role_numpy("valid", cols)
    uni = (pl.concat([pl.read_parquet(t.WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state"])
                      .filter(t.role_mask(c, "valid")) for c in t.COUNTRIES])
             .select(pl.col("entity_id").alias("s1_id"))
             .with_columns((pl.col("s1_id").hash(21) % 2).cast(pl.Int8).alias("half")))
    truth = t.labels().join(uni.select("s1_id"), on="s1_id").select("s1_id", "m_id")
    ids = mv.select("s1_id", "m_id")
    log(f"train rows {trn.sum()} (TRAIN_FRAC {TRAIN_FRAC}), early-stop rows {es.sum()}, VALID rows {len(ids)}")

    def objective(trial):
        cfg = space(trial)
        t0 = time.time()

        def prune_cb(env):                                # report -logloss every 50 rounds; stop weak trials early
            if env.iteration % 50 == 0 and env.iteration > 0:
                trial.report(-env.evaluation_result_list[0][2], env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        params = {**t.PARAMS, **cfg, "num_threads": a.threads, "metric": "binary_logloss"}
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["es"],
                      callbacks=[lgb.early_stopping(50, verbose=False), prune_cb])
        p = m.predict(Xv, num_threads=a.threads)
        sc = ids.with_columns(pl.Series("p", p)).filter(pl.col("p") >= 0.05)
        res = {thr: macro_by_half(sc, truth, uni, thr) for thr in THRESHOLDS}
        best_t = max(THRESHOLDS, key=lambda x: res[x][0])
        trial.set_user_attr("B", round(res[best_t][1], 5))
        trial.set_user_attr("t", best_t)
        trial.set_user_attr("iters", m.best_iteration)
        trial.set_user_attr("secs", round(time.time() - t0))
        log(f"trial {trial.number}: A {res[best_t][0]:.5f} B {res[best_t][1]:.5f} t={best_t} iters {m.best_iteration} "
            f"{time.time() - t0:.0f}s {cfg}")
        return res[best_t][0]

    storage = JournalStorage(JournalFileBackend(str(T / f"optuna_{a.study}.log")))
    study = optuna.create_study(study_name=a.study, storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=a.seed, multivariate=True, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=200))
    if a.enqueue:
        for s in SEEDS:
            study.enqueue_trial(s)
    study.optimize(objective, timeout=a.minutes * 60, catch=(Exception,))
    done = [tr for tr in study.trials if tr.value is not None]
    log(f"{a.study}: {len(study.trials)} trials ({len(done)} complete). TOP 5 by half A:")
    for tr in sorted(done, key=lambda tr: -tr.value)[:5]:
        log(f"  A {tr.value:.5f} B {tr.user_attrs.get('B')} t={tr.user_attrs.get('t')} iters {tr.user_attrs.get('iters')} {tr.params}")


if __name__ == "__main__":
    main()
