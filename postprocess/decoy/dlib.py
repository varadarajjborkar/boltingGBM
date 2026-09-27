"""Director shared helpers: VALID/test loaders, per-S1 F0.5, macro by country, bootstrap over S1, shift-rule flag.

Purpose: one place for the numbers every director analysis needs, so each wake reuses the same definitions.
Usage:   import sys; sys.path.insert(0, "postprocess/decoy"); import dlib
Inputs:  VALID analysis table (tools/valid_explorer.py output), train/test s1.parquet + pool.parquet (addr_nums),
         test stage-2 scores (work_fr/p1/test/<c>/stage2_scored_model_v2b.parquet = the v4 submission's scores).
Outputs: polars frames; nothing is written by this module.
Light by design: polars, 3 threads, a few hundred MB.
"""
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "3")
import numpy as np
import polars as pl

ROOT = "."
sys.path.insert(0, f"{ROOT}/src")
N_S1_VALID = {"US": 102314, "India": 73301}
RULE_OFFSETS = [3, 4, 5, 7, 9, 11, 13, 21]
BETA2 = 0.25


def f05_expr(tp, npred, ntrue):
    p = tp / npred
    r = tp / ntrue
    return (pl.when(ntrue == 0).then(pl.when(npred == 0).then(1.0).otherwise(0.0))
              .when(tp == 0).then(0.0)
              .otherwise((1 + BETA2) * p * r / (BETA2 * p + r)))


def hn_rule_flag(a_nums, b_nums):
    """1 if the closest differing house number is S1 number + a pure rule offset (v4_rule definition)."""
    from features import _hnum_dir
    f = _hnum_dir(list(a_nums), list(b_nums))
    return (f["f_hn_shift"] == 1) & np.isin(f["f_hn_sdiff"], RULE_OFFSETS), f


def load_valid(model="v2b", with_rule=True, extra_cols=()):
    """VALID table (every true and every scored pair). Adds 'rule' (v4_rule drop flag) and 'pred_r' (pred and not rule)."""
    cols = ["s1_id", "m_id", "y", "p", "in_blocking", "country", "src", "addr_empty", "name_non_latin",
            "p_best_any_s1", "is_argmax", "pred", "error", *extra_cols]
    d = pl.read_parquet(f"{ROOT}/work/advisor/valid_pairs_model_{model}.parquet", columns=cols)
    if with_rule:
        parts = []
        for c in ["US", "India"]:
            s1 = pl.read_parquet(f"{ROOT}/work/p1/train/{c}/s1.parquet", columns=["entity_id", "addr_nums"])
            dc = d.filter(pl.col("country") == c)
            pool = (pl.scan_parquet(f"{ROOT}/work/p1/train/{c}/pool.parquet").select("entity_id", "addr_nums")
                      .join(dc.select(pl.col("m_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect())
            x = (dc.join(s1.rename({"entity_id": "s1_id", "addr_nums": "a"}), on="s1_id", how="left")
                   .join(pool.rename({"entity_id": "m_id", "addr_nums": "b"}), on="m_id", how="left"))
            fl, f = hn_rule_flag(x["a"].fill_null("").to_list(), x["b"].fill_null("").to_list())
            x = x.with_columns(pl.Series("rule", fl), pl.Series("hn_sd", f["f_hn_sdiff"]), pl.Series("hn_shift", f["f_hn_shift"]),
                               pl.Series("hn_typo", f["f_hn_typo"])).drop("a", "b")
            parts.append(x)
        d = pl.concat(parts)
        d = d.with_columns((pl.col("pred") & ~pl.col("rule")).alias("pred_r"))
    return d


def entity_scores(d, pred_col="pred", truth_col="y"):
    """Per-S1 tp, npred, ntrue, f (only S1 present in d)."""
    e = d.group_by("s1_id", "country").agg(
        (pl.col(pred_col) & (pl.col(truth_col) == 1)).sum().alias("tp"),
        pl.col(pred_col).sum().alias("npred"),
        (pl.col(truth_col) == 1).sum().alias("ntrue"))
    return e.with_columns(f05_expr(pl.col("tp"), pl.col("npred"), pl.col("ntrue")).alias("f"))


def macro(e, n_s1=N_S1_VALID, col="f"):
    """Macro over all VALID S1: S1 absent from e score 1.0 (singletons with nothing scored)."""
    out, tot = {}, 0.0
    for c, n in n_s1.items():
        s = e.filter(pl.col("country") == c)[col].sum() + (n - e.filter(pl.col("country") == c).height)
        out[c] = s / n
        tot += s
    out["all"] = tot / sum(n_s1.values())
    return out


def bootstrap_gain(e_base, e_new, n_s1=N_S1_VALID, B=400, seed=0):
    """Bootstrap over S1 (absent S1 have zero gain) of the macro gain new - base. Returns mean, 2.5%, 97.5%,
    and the gain on each S1-hash half."""
    g = (e_base.select("s1_id", "country", pl.col("f").alias("f0"))
         .join(e_new.select("s1_id", pl.col("f").alias("f1")), on="s1_id", how="full", coalesce=True)
         .with_columns((pl.col("f1").fill_null(1.0) - pl.col("f0").fill_null(1.0)).alias("g")))
    N = sum(n_s1.values())
    gv = np.zeros(N)
    gv[: g.height] = g["g"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([gv[rng.integers(0, N, N)].mean() for _ in range(B)])
    h = g.with_columns((pl.col("s1_id").hash(7) % 2).alias("h"))
    halves = [h.filter(pl.col("h") == k)["g"].sum() / (N / 2) for k in (0, 1)]
    return {"gain": gv.mean(), "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975), "half0": halves[0], "half1": halves[1]}


def load_test_scores(model="v2b", countries=("US", "India", "France"), with_rule=True):
    """Stage-2 scores on test (the v4 submission) with the shift-rule flag and the v4/v4_rule decisions.
    v4 decision = one S1 per record (argmax) and p >= 0.75."""
    out = []
    for c in countries:
        t = pl.read_parquet(f"{ROOT}/work_fr/p1/test/{c}/stage2_scored_model_{model}.parquet").with_columns(pl.lit(c).alias("country"))
        t = t.with_columns((pl.col("p") == pl.col("p").max().over("m_id")).alias("is_argmax"))
        t = t.with_columns((pl.col("is_argmax") & (pl.col("p") >= 0.75)).alias("pred"))
        if with_rule:
            s1 = pl.read_parquet(f"{ROOT}/work_fr/p1/test/{c}/s1.parquet", columns=["entity_id", "addr_nums"]) if os.path.exists(
                f"{ROOT}/work_fr/p1/test/{c}/s1.parquet") else pl.read_parquet(f"{ROOT}/work_v4/p1/test/{c}/s1.parquet", columns=["entity_id", "addr_nums"])
            pp = f"{ROOT}/work_fr/p1/test/{c}/pool.parquet" if os.path.exists(f"{ROOT}/work_fr/p1/test/{c}/pool.parquet") else f"{ROOT}/work_v4/p1/test/{c}/pool.parquet"
            pool = (pl.scan_parquet(pp).select("entity_id", "addr_nums")
                      .join(t.select(pl.col("m_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect())
            x = (t.join(s1.rename({"entity_id": "s1_id", "addr_nums": "a"}), on="s1_id", how="left")
                  .join(pool.rename({"entity_id": "m_id", "addr_nums": "b"}), on="m_id", how="left"))
            fl, f = hn_rule_flag(x["a"].fill_null("").to_list(), x["b"].fill_null("").to_list())
            t = x.with_columns(pl.Series("rule", fl), pl.Series("hn_sd", f["f_hn_sdiff"]), pl.Series("hn_shift", f["f_hn_shift"]),
                               pl.Series("hn_typo", f["f_hn_typo"])).drop("a", "b")
            t = t.with_columns((pl.col("pred") & ~pl.col("rule")).alias("pred_r"))
        out.append(t)
    return pl.concat(out, how="diagonal")


def expected_f_mc(d, pred_col, p_col="p", n_sims=20, mu=None, seed=0):
    """Expected per-S1 F0.5 if pair scores were calibrated and independent (Monte Carlo over pair labels).
    mu: optional frame (s1_id, lam) or float = expected true matches outside the scored set (Poisson).
    Returns frame s1_id, country, ef."""
    rng = np.random.default_rng(seed)
    d = d.select("s1_id", "country", p_col, pred_col).sort("s1_id")
    sid = d["s1_id"].to_numpy()
    _, idx = np.unique(sid, return_index=True)
    grp = np.zeros(len(sid), np.int64)
    grp[idx[1:]] = 1
    grp = np.cumsum(grp)
    G = grp.max() + 1
    p = np.clip(d[p_col].to_numpy(), 0, 1)
    pr = d[pred_col].to_numpy().astype(bool)
    npred = np.bincount(grp, weights=pr, minlength=G)
    acc = np.zeros(G)
    lam = np.zeros(G) if mu is None else (np.full(G, mu) if np.isscalar(mu) else mu)
    for _ in range(n_sims):
        yy = rng.random(len(p)) < p
        tp = np.bincount(grp, weights=yy & pr, minlength=G)
        nt = np.bincount(grp, weights=yy, minlength=G) + rng.poisson(lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(npred > 0, tp / np.maximum(npred, 1), 0)
            R = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
            f = np.where(nt == 0, (npred == 0).astype(float), np.where(tp == 0, 0.0, (1 + BETA2) * P * R / np.maximum(BETA2 * P + R, 1e-12)))
        acc += f
    first = d.select("s1_id", "country").unique(subset=["s1_id"], keep="first", maintain_order=True)
    return first.with_columns(pl.Series("ef", acc / n_sims))
