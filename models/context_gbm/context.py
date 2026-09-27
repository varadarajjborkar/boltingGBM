"""Arch2 context features from FROZEN base scores s0 (= v10 stage-1 p1, out-of-fold on the training states).

competition (doc 15): per S1 over all its candidates: rank, best, second best, margin to the best other candidate,
  counts above 0.9 / 0.5 / P1_MIN, sum; per record over all S1 that retrieved it: rank, best other S1, margin,
  number of S1, number of S1 with s0 >= 0.5, sum of the other S1 scores.
peer support (doc 4, 16, 17): trusted seeds of S1 s = its candidates k with s0 >= tau (frozen, never updated).
  For a stage-2 pair (s, j): K(j, k) = record-record similarity of j and every seed k != j (name token set ratio,
  address token set ratio; K = mean of both, or the name alone when an address is empty). Features:
  P_j = sum_k K * s0_k / (n + 1), max and mean K, max name and address similarity, max min(name, address),
  trusted count (and same-source / other-source counts), contradiction = a seed with name similarity >= 0.9 and a
  different house number, coherence = mean K over seed-seed pairs of s. One step, order independent (sums, maxima).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import P1_MIN, THREADS


def competition(p1: pl.DataFrame) -> pl.DataFrame:
    """p1: [s1_id, m_id, p1] over ALL candidate pairs. Returns the stage-2 pairs with cm_* columns."""
    s = p1.group_by("s1_id").agg(pl.col("p1").top_k(2).alias("t"), pl.len().alias("cm_ncand_s1"),
                                 (pl.col("p1") >= 0.9).sum().alias("cm_n90_s1"), (pl.col("p1") >= 0.5).sum().alias("cm_n50_s1"),
                                 (pl.col("p1") >= P1_MIN).sum().alias("cm_n05_s1"), pl.col("p1").sum().alias("cm_sum_s1"))
    s = s.with_columns(pl.col("t").list.get(0).alias("cm_best_s1"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0).alias("sec")).drop("t")
    m = p1.group_by("m_id").agg(pl.col("p1").top_k(2).alias("t"), pl.len().alias("cm_n_m"),
                                (pl.col("p1") >= 0.5).sum().alias("cm_n50_m"), pl.col("p1").sum().alias("msum"))
    m = m.with_columns(pl.col("t").list.get(0).alias("mbest"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0).alias("msec")).drop("t")
    d = p1.filter(pl.col("p1") >= P1_MIN).select("s1_id", "m_id", "p1")
    d = d.with_columns(pl.col("p1").rank("min", descending=True).over("s1_id").alias("cm_rank_s1"),
                       pl.col("p1").rank("min", descending=True).over("m_id").alias("cm_rank_m_top"))
    d = d.join(s, on="s1_id", how="left").join(m, on="m_id", how="left")
    other_s1 = pl.when(pl.col("p1") >= pl.col("cm_best_s1")).then(pl.col("sec")).otherwise(pl.col("cm_best_s1"))
    other_m = pl.when(pl.col("p1") >= pl.col("mbest")).then(pl.col("msec")).otherwise(pl.col("mbest"))
    d = d.with_columns(
        (pl.col("p1") - other_s1).alias("cm_margin_s1"),
        pl.col("sec").alias("cm_second_s1"),
        other_m.alias("cm_best_other_m"),
        (pl.col("p1") - other_m).alias("cm_margin_m"),
        (pl.col("msum") - pl.col("p1")).alias("cm_other_sum_m"),
    ).drop("sec", "mbest", "msec", "msum")
    # rank of the record among ALL candidates of the S1 is computed over stage-2 pairs only; ranks of pairs below
    # P1_MIN never matter (they are not scored). Record-side rank uses the same set; both are order independent.
    return d.rename({"cm_rank_m_top": "cm_rank_m"}).with_columns(pl.all().exclude("s1_id", "m_id").cast(pl.Float32))


def _house(nums: pl.Expr) -> pl.Expr:
    return nums.str.split(" ").list.eval(pl.element().cast(pl.Int64, strict=False)).list.max()


def peer(p1: pl.DataFrame, pool: pl.DataFrame, tau: float, pre: str) -> pl.DataFrame:
    """p1: [s1_id, m_id, p1] all pairs; pool: [entity_id, name_core, addr_text, addr_nums]. Returns stage-2 pairs + pre* cols."""
    s2 = p1.filter(pl.col("p1") >= P1_MIN).select("s1_id", "m_id")
    seeds = p1.filter(pl.col("p1") >= tau).select("s1_id", pl.col("m_id").alias("k_id"), pl.col("p1").alias("pk"))
    tx = pool.select(pl.col("entity_id").alias("id"), pl.col("name_core").fill_null(""), pl.col("addr_text").fill_null(""),
                     _house(pl.col("addr_nums").fill_null("")).alias("hn"))
    tr = s2.join(seeds, on="s1_id").filter(pl.col("m_id") != pl.col("k_id"))
    tr = (tr.join(tx.rename({"id": "m_id", "name_core": "nj", "addr_text": "aj", "hn": "hj"}), on="m_id", how="left")
            .join(tx.rename({"id": "k_id", "name_core": "nk", "addr_text": "ak", "hn": "hk"}), on="k_id", how="left")
            .with_columns(pl.col("nj", "nk", "aj", "ak").fill_null("")))
    ns = cpdist(tr["nj"].to_list(), tr["nk"].to_list(), scorer=fuzz.token_set_ratio, workers=THREADS).astype(np.float32) / 100
    ad = cpdist(tr["aj"].to_list(), tr["ak"].to_list(), scorer=fuzz.token_set_ratio, workers=THREADS).astype(np.float32) / 100
    tr = tr.select("s1_id", "m_id", "k_id", "pk", "aj", "ak", "hj", "hk").with_columns(pl.Series("ns", ns), pl.Series("as", ad))
    noaddr = (pl.col("aj") == "") | (pl.col("ak") == "")
    tr = tr.with_columns(
        pl.when(noaddr).then(pl.col("ns")).otherwise((pl.col("ns") + pl.col("as")) / 2).alias("K"),
        pl.when(noaddr).then(None).otherwise(pl.col("as")).alias("as"),
        ((pl.col("ns") >= 0.9) & pl.col("hj").is_not_null() & pl.col("hk").is_not_null() & (pl.col("hj") != pl.col("hk"))).alias("contra"),
        (pl.col("m_id").str.slice(0, 2) == pl.col("k_id").str.slice(0, 2)).alias("same_src"))
    agg = tr.group_by("s1_id", "m_id").agg(
        pl.len().alias(f"{pre}n"),
        ((pl.col("K") * pl.col("pk")).sum() / (pl.len() + 1)).alias(f"{pre}P"),
        pl.col("K").max().alias(f"{pre}kmax"),
        pl.col("K").mean().alias(f"{pre}kmean"),
        pl.col("ns").max().alias(f"{pre}name"),
        pl.col("as").max().fill_null(-1).alias(f"{pre}addr"),
        pl.min_horizontal("ns", pl.col("as").fill_null(pl.col("ns"))).max().alias(f"{pre}both"),
        pl.col("contra").sum().alias(f"{pre}contra"),
        pl.col("same_src").sum().alias(f"{pre}same_src"),
        (~pl.col("same_src")).sum().alias(f"{pre}other_src"))
    # coherence of the seed set of each S1 (seed-seed pairs, k < l)
    sp_ = seeds.join(seeds.rename({"k_id": "l_id", "pk": "pl"}), on="s1_id").filter(pl.col("k_id") < pl.col("l_id"))
    if sp_.height:
        sp_ = (sp_.join(tx.rename({"id": "k_id", "name_core": "nk", "addr_text": "ak", "hn": "hk"}), on="k_id", how="left")
                  .join(tx.rename({"id": "l_id", "name_core": "nl", "addr_text": "al", "hn": "hl"}), on="l_id", how="left")
                  .with_columns(pl.col("nk", "nl", "ak", "al").fill_null("")))
        cn = cpdist(sp_["nk"].to_list(), sp_["nl"].to_list(), scorer=fuzz.token_set_ratio, workers=THREADS).astype(np.float32) / 100
        ca = cpdist(sp_["ak"].to_list(), sp_["al"].to_list(), scorer=fuzz.token_set_ratio, workers=THREADS).astype(np.float32) / 100
        sp_ = sp_.select("s1_id", "ak", "al").with_columns(pl.Series("cn", cn), pl.Series("ca", ca))
        coh = sp_.group_by("s1_id").agg(
            pl.when((pl.col("ak") == "") | (pl.col("al") == "")).then(pl.col("cn")).otherwise((pl.col("cn") + pl.col("ca")) / 2)
              .mean().alias(f"{pre}coh"))
    else:
        coh = pl.DataFrame({"s1_id": [], f"{pre}coh": []}, schema={"s1_id": pl.String, f"{pre}coh": pl.Float32})
    out = s2.join(agg, on=["s1_id", "m_id"], how="left").join(coh, on="s1_id", how="left")
    cnt = [f"{pre}n", f"{pre}contra", f"{pre}same_src", f"{pre}other_src"]
    out = out.with_columns(pl.col(cnt).fill_null(0), pl.col(f"{pre}P").fill_null(0),
                           pl.col([f"{pre}kmax", f"{pre}kmean", f"{pre}name", f"{pre}addr", f"{pre}both", f"{pre}coh"]).fill_null(-1))
    return out.with_columns(pl.all().exclude("s1_id", "m_id").cast(pl.Float32))
