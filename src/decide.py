"""Turn pair probabilities into per-entity match sets, optimised for macro F0.5.

1) one_s1_per_record: an S2/S3 record belongs to at most one S1 (hard fact in train labels), so each
   record is kept only for the S1 that scores it highest.
2a) threshold: keep pairs with p >= t (baseline).
2b) expected_f: per S1, sort candidates by p and pick the prefix size k (including k = 0, "no match") that
   maximises expected F0.5 (decision-theoretic approach, Ye et al., ICML 2012), using the ratio-of-expectations
   approximation  E[F_k] ~ (1+b^2) * sum_{i<=k} p_i / (k + b^2 * E[T]),  E[T] = sum_i p_i + mu,
   and  E[F_0] = P(no true match) ~ prod_i (1 - p_i) * exp(-mu).
   mu = expected true matches the blocker missed (tuned on validation).
"""
import polars as pl

BETA2 = 0.25


def one_s1_per_record(scored: pl.DataFrame) -> pl.DataFrame:
    return scored.filter(pl.col("p") == pl.col("p").max().over("m_id")).unique(subset=["m_id"], keep="first")


def by_threshold(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    return scored.filter(pl.col("p") >= t).select("s1_id", "m_id")


def by_expected_f(scored: pl.DataFrame, mu: float = 0.05, p_floor: float = 0.0) -> pl.DataFrame:
    d = (scored.filter(pl.col("p") > p_floor).sort(["s1_id", "p"], descending=[False, True])
         .with_columns(pl.col("p").clip(1e-6, 1 - 1e-6).alias("pc"))
         .with_columns(
             pl.col("pc").cum_sum().over("s1_id").alias("cum_p"),
             pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
             (pl.col("pc").sum().over("s1_id") + mu).alias("ET"),
             ((1 - pl.col("pc")).log().sum().over("s1_id") - mu).exp().alias("EF0"),
         )
         .with_columns(((1 + BETA2) * pl.col("cum_p") / (pl.col("k") + BETA2 * pl.col("ET"))).alias("EFk")))
    best = d.group_by("s1_id").agg(
        pl.col("EFk").max().alias("best"),
        pl.col("k").filter(pl.col("EFk") == pl.col("EFk").max()).min().alias("kbest"),
        pl.col("EF0").first().alias("EF0"))
    d = d.join(best, on="s1_id").filter((pl.col("best") > pl.col("EF0")) & (pl.col("k") <= pl.col("kbest")))
    return d.select("s1_id", "m_id")


def to_sets(pairs: pl.DataFrame):
    g = pairs.group_by("s1_id").agg(pl.col("m_id"))
    return {r[0]: set(r[1]) for r in g.iter_rows()}


# ---------------------------------------------------------------- entity-level "has any match" model (review #1: singletons are traps)
ENTITY_FEATS = ["p_max", "p_2nd", "p_sum", "n_p50", "n_p20", "n_cand", "cos_comb_max", "cos_name_max", "cos_addr_max",
                "n_same_name", "num_share_top"]


def entity_table(scored: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """One row per S1 with aggregates of its (post one-S1-per-record) candidate scores."""
    d = scored.join(feats.select("s1_id", "m_id", "f_cos_comb", "f_cos_name", "f_cos_addr", "f_n_same_name_s1", "f_num_share"),
                    on=["s1_id", "m_id"], how="left").sort(["s1_id", "p"], descending=[False, True])
    return d.group_by("s1_id").agg(
        pl.col("p").max().alias("p_max"),
        pl.col("p").get(1, null_on_oob=True).fill_null(0).alias("p_2nd"),
        pl.col("p").sum().alias("p_sum"),
        (pl.col("p") > 0.5).sum().alias("n_p50"),
        (pl.col("p") > 0.2).sum().alias("n_p20"),
        pl.len().alias("n_cand"),
        pl.col("f_cos_comb").max().alias("cos_comb_max"),
        pl.col("f_cos_name").max().alias("cos_name_max"),
        pl.col("f_cos_addr").max().alias("cos_addr_max"),
        pl.col("f_n_same_name_s1").first().alias("n_same_name"),
        pl.col("f_num_share").first().alias("num_share_top"),
    )


def crossfit_entity_prob(ent: pl.DataFrame, has_match: dict, seed: int = 11) -> pl.DataFrame:
    """P(S1 has >= 1 match) by 2-fold cross-fitting (train on one half, predict the other)."""
    import lightgbm as lgb
    ent = ent.with_columns(pl.col("s1_id").map_elements(lambda s: int(has_match.get(s, False)), return_dtype=pl.Int8).alias("y"))
    if "fold" not in ent.columns:          # default: random halves; callers may pass state-based folds
        ent = ent.with_columns((pl.col("s1_id").hash(seed) % 2).alias("fold"))
    out = []
    for f in (0, 1):
        tr, te = ent.filter(pl.col("fold") != f), ent.filter(pl.col("fold") == f)
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1,
                           num_threads=7, seed=seed), lgb.Dataset(tr.select(ENTITY_FEATS).to_numpy(), tr["y"].to_numpy()), 300)
        out.append(te.select("s1_id").with_columns(pl.Series("p_match", m.predict(te.select(ENTITY_FEATS).to_numpy()))))
    return pl.concat(out)


def by_expected_f_entity(scored: pl.DataFrame, p_match: pl.DataFrame, mu: float = 0.05) -> pl.DataFrame:
    """Expected-F selection with P(no match) from the entity model (corrected after review #2).

    Given the entity has >= 1 match (prob p_m), candidate i is a match with q_i = p_i / (1 - prod_j (1 - p_j)).
    E[F_k] = p_m * (1+b^2) * sum_{i<=k} q_i / (k + b^2 * E[T | match]),  E[T | match] = sum_i q_i + mu,
    E[F_0] = 1 - p_m.
    """
    d = (scored.sort(["s1_id", "p"], descending=[False, True])
         .with_columns(pl.col("p").clip(1e-6, 1 - 1e-6).alias("pc"))
         .with_columns((1 - ((1 - pl.col("pc")).log().sum().over("s1_id")).exp()).clip(1e-6, 1).alias("p_any"))
         .with_columns((pl.col("pc") / pl.col("p_any")).clip(0, 1).alias("q"))
         .with_columns(pl.col("q").cum_sum().over("s1_id").alias("cum_q"),
                       pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
                       (pl.col("q").sum().over("s1_id") + mu).alias("ETm"))
         .join(p_match, on="s1_id", how="left")
         .with_columns(pl.col("p_match").fill_null(0.5).alias("pm"))
         .with_columns((pl.col("pm") * (1 + BETA2) * pl.col("cum_q") / (pl.col("k") + BETA2 * pl.col("ETm"))).alias("EFk"),
                       (1 - pl.col("pm")).alias("EF0")))
    best = d.group_by("s1_id").agg(pl.col("EFk").max().alias("best"),
                                   pl.col("k").filter(pl.col("EFk") == pl.col("EFk").max()).min().alias("kbest"),
                                   pl.col("EF0").first().alias("EF0"))
    d = d.join(best, on="s1_id").filter((pl.col("best") > pl.col("EF0")) & (pl.col("k") <= pl.col("kbest")))
    return d.select("s1_id", "m_id")


# ---------------------------------------------------------------- prior shift (review #1: test has ~2x distractors)
def saerens_prior(p, train_prior: float, iters: int = 100, tol: float = 1e-7):
    """EM estimate of the positive-class prior on unlabelled data (Saerens, Latinne, Decaestecker, Neural Comp. 2002).

    p: calibrated P(match) from a model trained where the prior was train_prior. Returns (new_prior, odds_ratio r)
    so that adjusted p' = r p / (r p + 1 - p).
    """
    import numpy as np
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    q = train_prior
    for _ in range(iters):
        r = (q / train_prior) / ((1 - q) / (1 - train_prior))
        adj = r * p / (r * p + 1 - p)
        q_new = adj.mean()
        if abs(q_new - q) < tol:
            q = q_new
            break
        q = q_new
    r = (q / train_prior) / ((1 - q) / (1 - train_prior))
    return q, r
