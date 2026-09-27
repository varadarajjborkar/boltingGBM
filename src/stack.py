"""Stage-2 (stacked) features built from stage-1 pair probabilities.

Entity resolution decisions are relational: whether record m belongs to S1 i depends on how strongly m fits
OTHER S1s, and on how many other good candidates i already has. Stage 1 scores pairs in isolation; stage 2
sees those scores in context. Training uses out-of-fold stage-1 probabilities so stage 2 never learns from
leaked (in-sample) scores.
"""
import numpy as np
import polars as pl

ANCHOR_P = 0.5     # stage-1 probability above which a record counts as a confident member of an S1 ("anchor")
SIB_COLS = ["s3_sib_n", "s3_sib_name", "s3_sib_addr", "s3_sib_both", "s3_sib_wname", "s3_sib_same_src", "s3_sib_other_src"]
SIB_COUNTS = ["s3_sib_n", "s3_sib_same_src", "s3_sib_other_src"]


def stage2_features(scored: pl.DataFrame) -> pl.DataFrame:
    """scored: [s1_id, m_id, p1] (+ anything). Adds s2_* columns."""
    return scored.with_columns(
        # S1 side
        pl.col("p1").rank("dense", descending=True).over("s1_id").cast(pl.Int16).alias("s2_rank_s1"),
        (pl.col("p1").max().over("s1_id") - pl.col("p1")).alias("s2_gap_s1"),
        pl.col("p1").sum().over("s1_id").alias("s2_sum_s1"),
        (pl.col("p1") > 0.5).sum().over("s1_id").cast(pl.Int16).alias("s2_n50_s1"),
        (pl.col("p1") > 0.2).sum().over("s1_id").cast(pl.Int16).alias("s2_n20_s1"),
        # record side: competition from other S1s for the same S2/S3 record
        pl.col("p1").rank("dense", descending=True).over("m_id").cast(pl.Int16).alias("s2_rank_m"),
        (pl.col("p1").sum().over("m_id") - pl.col("p1")).alias("s2_other_sum_m"),
        pl.len().over("m_id").cast(pl.Int16).alias("s2_n_m"),
    ).with_columns(
        # best competing S1 probability for this record (excluding this pair)
        pl.when(pl.col("s2_rank_m") == 1)
          .then(pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("m_id"))
          .otherwise(pl.col("p1").max().over("m_id")).fill_null(0.0).alias("s2_best_other_m"),
    ).with_columns((pl.col("p1") - pl.col("s2_best_other_m")).alias("s2_margin_m"))


def sibling_features(p1tab: pl.DataFrame, texts: pl.DataFrame, p1_min: float) -> pl.DataFrame:
    """Collective ("sibling") evidence: does record m resemble the records already confidently matched to S1 s?

    S2/S3 duplicates of one business often share typos, abbreviations or a missing address with EACH OTHER while
    all looking weak against the clean S1 record. For every stage-2 pair (s, m) (p1 >= p1_min) we compare m with
    the anchors of s (p1(s, m') >= ANCHOR_P, m' != m) and keep: number of anchors, max name similarity, max address
    similarity, max of min(name, address) similarity, max p1(anchor) * name similarity (-1 when s has no anchor).
    p1tab: [s1_id, m_id, p1] over ALL candidate pairs; texts: [m_id, name_core, addr_text] of the record pool.
    Validation (v1 stage 1, VALID ny+ap): macro F0.5 0.9721 -> 0.9739."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    s2 = p1tab.filter(pl.col("p1") >= p1_min).select("s1_id", "m_id")
    anc = p1tab.filter(pl.col("p1") >= ANCHOR_P).select("s1_id", pl.col("m_id").alias("a_id"), pl.col("p1").alias("a_p1"))
    pr = s2.join(anc, on="s1_id").filter(pl.col("m_id") != pl.col("a_id"))
    tm = texts.rename({"name_core": "n_m", "addr_text": "ad_m"})
    ta = texts.rename({"m_id": "a_id", "name_core": "n_a", "addr_text": "ad_a"})
    pr = pr.join(tm, on="m_id", how="left").join(ta, on="a_id", how="left").with_columns(
        pl.col("n_m", "ad_m", "n_a", "ad_a").fill_null(""))
    ns = cpdist(pr["n_m"].to_list(), pr["n_a"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    ad = cpdist(pr["ad_m"].to_list(), pr["ad_a"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    pr = pr.select("s1_id", "m_id", "a_p1", (pl.col("m_id").str.slice(0, 2) == pl.col("a_id").str.slice(0, 2)).alias("same_src")).with_columns(pl.Series("ns", ns.astype(np.float32)),
                                                         pl.Series("ad", ad.astype(np.float32)))
    agg = pr.group_by("s1_id", "m_id").agg(
        pl.len().cast(pl.Int16).alias("s3_sib_n"),
        pl.col("ns").max().alias("s3_sib_name"),
        pl.col("ad").max().alias("s3_sib_addr"),
        pl.min_horizontal("ns", "ad").max().alias("s3_sib_both"),
        (pl.col("a_p1") * pl.col("ns")).max().alias("s3_sib_wname"),
        # anchors from the record's own source vs the other one (ids start with S2/S3): 85% of matched S1 have both
        pl.col("same_src").sum().cast(pl.Int16).alias("s3_sib_same_src"),
        (~pl.col("same_src")).sum().cast(pl.Int16).alias("s3_sib_other_src"))
    return s2.join(agg, on=["s1_id", "m_id"], how="left").with_columns(
        pl.col(SIB_COUNTS).fill_null(0), pl.col([c for c in SIB_COLS if c not in SIB_COUNTS]).fill_null(-1.0))


def pool_texts(pool_files) -> pl.DataFrame:
    """[m_id, name_core, addr_text] for the record pools (inputs of sibling_features)."""
    return pl.concat([pl.read_parquet(f, columns=["entity_id", "name_core", "addr_text"]) for f in pool_files]).rename({"entity_id": "m_id"})


def oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: dict, rounds: int, n_folds: int = 3, seed: int = 7):
    """Out-of-fold probabilities, folds split by S1 group (hash) so no S1 is in its own training fold."""
    import lightgbm as lgb
    fold = (np.abs(groups) % n_folds).astype(int)
    oof = np.zeros(len(y), dtype=np.float32)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds)
        oof[te] = m.predict(X[te], num_threads=params.get("num_threads", 7))
    return oof
