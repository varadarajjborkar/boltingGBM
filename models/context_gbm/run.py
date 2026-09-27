"""Arch2 track runner: s0 (frozen v10 stage-1 p1) + new pair features + competition + peer support -> GBDT matcher.

Stages (each writes under work_v4/arch2/, skipped when its output exists unless --force):
  p1     : all-pair s0 tables. train = OOF p1 of the stage-2 states (s2fit_p1), valid = v10 stage-1 booster over
           train/<c>/feats_valid (fair tr columns), test = test/<c>/p1_model_v10
  feats  : arch2 pair features (features.py) for the stage-2 pairs (p1 >= 0.005) of every split-country
  context: competition + peer support (context.py) from the frozen s0 tables
  train  : LightGBM matchers for the ablation variants (early stopping on an S1-hash group split)
  valid  : VALID macro F0.5 per variant (one S1 per record, threshold sweep, both halves, singleton accuracy)
  test   : test scores of the chosen variant -> work_v4/p1/test/<c>/stage2_scored_arch2.parquet
Usage: python run.py --stages p1,feats,context,train,valid [--force feats] [--variants base,feat,comp,peer]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl

from common import (COUNTRIES, OUT, P1_MIN, TEST_COUNTRIES, THREADS, WORK, fmt, labels, log, rss_gb, split_dir, sweep,
                    texts, valid_universe, write_atomic)

V10 = WORK / "p1" / "model_v10"


def p1_path(split, c):
    return OUT / f"p1_{split}_{c}.parquet"


def splits():
    return [("train", c) for c in COUNTRIES] + [("valid", c) for c in COUNTRIES] + [("test", c) for c in TEST_COUNTRIES]


# ---------------------------------------------------------------- p1 (s0) tables
def stage_p1(force):
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    s2 = None
    for c in COUNTRIES:
        out = p1_path("train", c)
        if out.exists() and not force:
            continue
        if s2 is None:
            s2 = pl.read_parquet(V10 / "s2fit_p1.parquet")
        ids = pl.read_parquet(split_dir("train", c) / "s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"})
        d = s2.join(ids, on="s1_id", how="semi")
        write_atomic(d, out)
        log(f"train {c}: {d.height} pairs, {d.filter(pl.col('p1') >= P1_MIN).height} stage-2 pairs, {d['y'].sum()} true")
    cols = json.loads((V10 / "feature_cols.json").read_text())
    booster = None
    for c in COUNTRIES:
        out = p1_path("valid", c)
        if out.exists() and not force:
            continue
        if booster is None:
            booster = lgb.Booster(model_file=str(V10 / "stage1_fold0.txt"))
        parts = sorted((split_dir("train", c) / "feats_valid").glob("part_*.parquet"))
        res = []
        for p in parts:
            df = pl.read_parquet(p, columns=["s1_id", "m_id", "y"] + cols)
            X = df.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
            res.append(df.select("s1_id", "m_id", pl.col("y").cast(pl.Int8)).with_columns(
                pl.Series("p1", booster.predict(X, num_threads=THREADS).astype(np.float32))))
            log(f"valid {c} {p.name}: {df.height} rows, RSS {rss_gb():.1f} GB")
            del df, X
        d = pl.concat(res).unique(["s1_id", "m_id"], keep="first")
        write_atomic(d, out)
        log(f"valid {c}: {d.height} pairs, {d.filter(pl.col('p1') >= P1_MIN).height} stage-2 pairs")
    for c in TEST_COUNTRIES:
        out = p1_path("test", c)
        if out.exists() and not force:
            continue
        d = pl.read_parquet(split_dir("test", c) / "p1_model_v10.parquet").with_columns(pl.lit(0, pl.Int8).alias("y"))
        write_atomic(d, out)
        log(f"test {c}: {d.height} pairs, {d.filter(pl.col('p1') >= P1_MIN).height} stage-2 pairs")

# ---------------------------------------------------------------- arch2 pair features
def data_split(split):
    return "test" if split == "test" else "train"


def selected(a):
    only = set(filter(None, a.only.split(",")))
    return [(s, c) for s, c in splits() if not only or f"{s}:{c}" in only]


def stage_feats(force, a):
    import features as F
    corp = {}
    for split, c in selected(a):
        out = OUT / f"feats_{split}_{c}.parquet"
        if out.exists() and not force:
            continue
        ds = data_split(split)
        s1, pool = texts(ds, c)
        if (ds, c) not in corp:
            corp.clear()
            log(f"corpus {ds}/{c}: {s1.height} S1 + {pool.height} pool")
            corp[(ds, c)] = F.Corpus(s1, pool, cache_dir=OUT / "corpus" / f"{ds}_{c}")
        pairs = pl.read_parquet(p1_path(split, c)).filter(pl.col("p1") >= P1_MIN)
        log(f"feats {split}/{c}: {pairs.height} pairs, RSS {rss_gb():.1f} GB")
        f = F.build(pairs, s1, pool, corp[(ds, c)], log=log)
        write_atomic(f, out)
        log(f"feats {split}/{c} written: {f.shape}, RSS {rss_gb():.1f} GB")


# ---------------------------------------------------------------- context (competition + peer support)
def stage_context(force, a):
    from context import competition, peer
    for split, c in selected(a):
        out = OUT / f"ctx_{split}_{c}.parquet"
        if out.exists() and not force:
            continue
        p1 = pl.read_parquet(p1_path(split, c), columns=["s1_id", "m_id", "p1"])
        _, pool = texts(data_split(split), c)
        pool = pool.select("entity_id", "name_core", "addr_text", "addr_nums")
        d = competition(p1)
        for tau, pre in TAUS:
            d = d.join(peer(p1, pool, tau, pre), on=["s1_id", "m_id"], how="left")
            log(f"context {split}/{c}: peer tau {tau} done, RSS {rss_gb():.1f} GB")
        write_atomic(d.drop("p1"), out)
        log(f"context {split}/{c}: {d.shape}")


TAUS = [(0.9, "pe_"), (0.5, "pe5_")]


# ---------------------------------------------------------------- v10 f_ columns and extra seed thresholds
FEATS_DIR = {"train": ("train", "feats_fit"), "valid": ("train", "feats_valid"), "test": ("test", "feats_all")}


def stage_fcols(force, a):
    """v10 stage-1 f_* columns of the stage-2 pairs (train: feats_fit of tx/up/ka, valid: fair feats_valid,
    test: feats_all or a pre-filtered fcols file pulled from worker 3)."""
    cols = json.loads((V10 / "feature_cols.json").read_text())
    for split, c in selected(a):
        out = OUT / f"fcols_{split}_{c}.parquet"
        if out.exists() and not force:
            continue
        ds, folder = FEATS_DIR[split]
        parts = sorted((split_dir(ds, c) / folder).glob("part_*.parquet"))
        if not parts:
            log(f"fcols {split}/{c}: no {folder} parts, skipped"); continue
        keys = pl.read_parquet(p1_path(split, c), columns=["s1_id", "m_id", "p1"]).filter(pl.col("p1") >= P1_MIN).drop("p1")
        res = [pl.read_parquet(p, columns=["s1_id", "m_id"] + cols).join(keys, on=["s1_id", "m_id"], how="semi") for p in parts]
        d = pl.concat(res).unique(["s1_id", "m_id"], keep="first").with_columns(pl.col(cols).cast(pl.Float32))
        write_atomic(d, out)
        log(f"fcols {split}/{c}: {d.height} of {keys.height} stage-2 pairs")


XTAUS = [(0.95, "pe95_"), (0.85, "pe85_")]


def stage_ctxtau(force, a):
    from context import peer
    for split, c in selected(a):
        out = OUT / f"ctxt_{split}_{c}.parquet"
        if out.exists() and not force:
            continue
        p1 = pl.read_parquet(p1_path(split, c), columns=["s1_id", "m_id", "p1"])
        _, pool = texts(data_split(split), c)
        pool = pool.select("entity_id", "name_core", "addr_text", "addr_nums")
        d = None
        for tau, pre in XTAUS:
            x = peer(p1, pool, tau, pre)
            d = x if d is None else d.join(x, on=["s1_id", "m_id"], how="left")
        write_atomic(d, out)
        log(f"ctxtau {split}/{c}: {d.shape}")


# ---------------------------------------------------------------- design matrices
def load_design(split, countries, need=("feats", "ctx", "fcols", "ctxt")):
    parts = []
    for c in countries:
        d = pl.read_parquet(p1_path(split, c)).filter(pl.col("p1") >= P1_MIN)
        for k in need:
            f = OUT / f"{k}_{split}_{c}.parquet"
            if f.exists():
                d = d.join(pl.read_parquet(f), on=["s1_id", "m_id"], how="left")
            elif k in ("feats", "ctx"):
                raise FileNotFoundError(f)
        parts.append(d.with_columns(pl.lit(float(c == "India"), pl.Float32).alias("is_india"),
                                    pl.lit(float(c == "France"), pl.Float32).alias("is_france")))
    return pl.concat(parts, how="diagonal_relaxed")


def variant_cols(d, name):
    name = name.split("-")[0].split("_b")[0]
    feat = [k for k in d.columns if k.split("_")[0] in ("nm", "ph", "ad", "num", "hn")]
    comp = [k for k in d.columns if k.startswith("cm_")]
    pre = lambda p: [k for k in d.columns if k.startswith(p)]
    fc = json.loads((V10 / "feature_cols.json").read_text())
    base = ["p1", "is_india"] + feat + comp
    v = {"base": ["p1"],
         "feat": ["p1", "is_india"] + feat,
         "comp": base,
         "peer": base + pre("pe_"),
         "peer2": base + pre("pe_") + pre("pe5_"),
         "peer95": base + pre("pe95_"),
         "peer85": base + pre("pe85_"),
         "peerf": base + pre("pe_") + fc,
         "peerf95": base + pre("pe95_") + fc,
         "peerf85": base + pre("pe85_") + fc,
         "nofeat": ["p1", "is_india"] + comp + pre("pe_") + pre("pe5_")}
    return v[name]


def n_seeds(v):
    return int(v.split("_b")[1]) if "_b" in v else 1


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=THREADS, verbose=-1, seed=7,
              deterministic=True, force_row_wise=True)
PARAMS.update(json.loads(os.environ.get("A2_PARAMS", "{}")))     # ad hoc override for every model of the run
SEEDS = [7, 11, 23, 41]
# hyperparameter configs: variant "<name>-<tag>" (e.g. peer-h3) trains <name>'s columns with PARAMS + HP[tag]
HP = {"h1": dict(num_leaves=63), "h2": dict(num_leaves=255), "h3": dict(num_leaves=511, min_data_in_leaf=300),
      "h4": dict(min_data_in_leaf=30), "h5": dict(min_data_in_leaf=300), "h6": dict(learning_rate=0.03),
      "h7": dict(feature_fraction=0.6), "h8": dict(feature_fraction=1.0), "h9": dict(lambda_l2=0.0),
      "h10": dict(lambda_l2=5.0), "h11": dict(bagging_fraction=1.0, bagging_freq=0),
      "h12": dict(num_leaves=255, min_data_in_leaf=30, feature_fraction=0.6, learning_rate=0.03, lambda_l2=5.0),
      "h13": dict(num_leaves=511, min_data_in_leaf=100, feature_fraction=0.6, learning_rate=0.03),
      "h14": dict(num_leaves=31), "h15": dict(num_leaves=63, min_data_in_leaf=300),
      "h16": dict(num_leaves=63, learning_rate=0.03)}


def params_for(v):
    tag = v.split("-")[1] if "-" in v else None
    p = {**PARAMS, **(HP[tag] if tag else {})}
    return p, (150 if p["learning_rate"] < 0.05 else 100)
VARIANTS = ["base", "feat", "comp", "peer", "peer2"]


def model_files(v):
    md = OUT / "models"
    return [md / f"{v}.txt"] if n_seeds(v) == 1 else [md / f"{v}_s{k}.txt" for k in range(n_seeds(v))]


def stage_train(force, a):
    import lightgbm as lgb
    md = OUT / "models"
    md.mkdir(parents=True, exist_ok=True)
    d = load_design("train", COUNTRIES)
    y = d["y"].to_numpy()
    es = (d["s1_id"].hash(13) % 7 == 3).to_numpy()
    log(f"train design {d.shape}, positives {y.mean():.4f}, es rows {es.sum()}")
    for v in (a.variants.split(",") if a.variants else VARIANTS):
        if v == "base" or (all(f.exists() for f in model_files(v)) and not force):
            continue            # base = s0 itself (a one-feature matcher is a monotone map of p1)
        cols = variant_cols(d, v)
        miss = d.select([pl.col(k).is_null().mean() for k in cols if k.startswith("f_")]).row(0) if any(k.startswith("f_") for k in cols) else [0]
        log(f"{v}: {len(cols)} cols, max null share of f_ cols {max(miss):.4f}")
        X = d.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
        dtr = lgb.Dataset(X[~es], y[~es], feature_name=cols, free_raw_data=False)
        des = lgb.Dataset(X[es], y[es], feature_name=cols, reference=dtr, free_raw_data=False)
        for k, f in enumerate(model_files(v)):
            if f.exists() and not force:
                continue
            prm, pat = params_for(v)
            m = lgb.train({**prm, "seed": SEEDS[k]}, dtr, num_boost_round=8000, valid_sets=[des],
                          callbacks=[lgb.early_stopping(pat, verbose=False), lgb.log_evaluation(500)])
            m.save_model(str(f))
            imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda t: -t[1])
            log(f"{f.stem}: best iter {m.best_iteration}, es logloss {m.best_score['valid_0']['binary_logloss']:.5f}; "
                f"top {[c for c, _ in imp[:10]]}")
        (md / f"{v}.json").write_text(json.dumps(cols))
        del X, dtr, des


def predict_variant(d, v):
    import lightgbm as lgb
    if v == "base":
        return d["p1"].to_numpy().astype(np.float64)
    cols = json.loads((OUT / "models" / f"{v}.json").read_text())
    X = d.select([pl.col(k).cast(pl.Float32) for k in cols]).to_numpy()
    return np.mean([lgb.Booster(model_file=str(f)).predict(X, num_threads=THREADS) for f in model_files(v)], axis=0)


def stage_valid(force, a):
    u = valid_universe()
    d = load_design("valid", COUNTRIES)
    log(f"VALID design {d.shape}; {u.height} VALID S1")
    rep = {}
    ref = pl.read_parquet(WORK / "p1" / "model_v10_vx" / "valid_scored.parquet").select("s1_id", "m_id", "y", "p")
    best, _ = sweep(ref, u)
    rep["v10_vx"] = best
    log(f"v10_vx (reference)  {fmt(best)}")
    sc = d.select("s1_id", "m_id", "y")
    for v in (a.variants.split(",") if a.variants else VARIANTS):
        if v != "base" and not all(f.exists() for f in model_files(v)):
            continue
        p = predict_variant(d, v)
        s = sc.with_columns(pl.Series("p", p))
        write_atomic(s, OUT / f"valid_scored_{v}.parquet")
        best, rows = sweep(s, u)
        rep[v] = best
        log(f"{v:8s} {fmt(best)}")
        at70 = [r for r in rows if abs(r["t"] - 0.70) < 1e-9][0]
        log(f"{v:8s} @0.70 {fmt(at70)}")
    (OUT / f"valid_report_{a.variants or 'all'}.json").write_text(json.dumps(rep, indent=1))


def stage_test(force, a):
    v = a.variants or "peer"
    for c in TEST_COUNTRIES:
        out = split_dir("test", c) / f"stage2_scored_{a.name}.parquet"
        if out.exists() and not force:
            continue
        d = load_design("test", [c])
        p = predict_variant(d, v)
        s = d.select("s1_id", "m_id").with_columns(pl.Series("p", p))
        write_atomic(s, out)
        log(f"test {c}: {s.height} pairs scored with {v}; p>=0.7: {(p >= 0.7).sum()}")


def stage_blend(force, a):
    """Probability blend of an arch2 variant with v10 (pairs missing on one side count as p = 0 there)."""
    u = valid_universe()
    v = a.variants or "peer2"
    s = pl.read_parquet(OUT / f"valid_scored_{v}.parquet").rename({"p": "pa"})
    ref = pl.read_parquet(WORK / "p1" / "model_v10_vx" / "valid_scored.parquet").select("s1_id", "m_id", pl.col("y").alias("yr"), pl.col("p").alias("pb"))
    d = s.join(ref, on=["s1_id", "m_id"], how="full", coalesce=True).with_columns(
        pl.coalesce("y", "yr").alias("y"), pl.col("pa").fill_null(0), pl.col("pb").fill_null(0))
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        best, rows = sweep(d.with_columns((w * pl.col("pa") + (1 - w) * pl.col("pb")).alias("p")), u)
        log(f"blend w_arch2={w}: {fmt(best)}")


STAGES = {"p1": stage_p1, "feats": stage_feats, "context": stage_context, "train": stage_train, "valid": stage_valid,
          "test": stage_test, "blend": stage_blend,
          "fcols": stage_fcols, "ctxtau": stage_ctxtau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="p1")
    ap.add_argument("--force", default="")
    ap.add_argument("--only", default="", help="comma list of split:country to restrict feats/context")
    ap.add_argument("--variants", default="")
    ap.add_argument("--name", default="arch2", help="test output: stage2_scored_<name>.parquet")
    a = ap.parse_args()
    log(f"arch2 run.py PID {os.getpid()} stages {a.stages}")
    force = set(filter(None, a.force.split(",")))
    for st in a.stages.split(","):
        STAGES[st](st in force) if st in ("p1",) else STAGES[st](st in force, a)


if __name__ == "__main__":
    main()
