"""Phase 1: full-universe pair features, chunked, with chunk-level checkpoints.

Context that needs ALL candidates (record-side ranks, name frequencies, IDF) is computed over the whole country,
then heavy string features are built in chunks of S1 entities. Used for:
  - train split: only S1 whose role is in --roles (fit / valid), features saved for model training / validation
  - test split: every S1 (called from p1_score.py, which scores each chunk immediately)
"""
import gc

import polars as pl

from features import add_m_context, build_features, name_freq_tables, token_idf
from utils import WORK, log, rss_gb, write_parquet_atomic

ID_COLS = ["s1_id", "m_id"]


def load_country(split, country):
    d = WORK / "p1" / split / country
    s1 = pl.read_parquet(d / "s1.parquet").with_row_index("i").with_columns(pl.col("i").cast(pl.Int32))
    pool = pl.read_parquet(d / "pool.parquet").with_row_index("j").with_columns(pl.col("j").cast(pl.Int32))
    cands = pl.read_parquet(d / "cands.parquet")
    return s1, pool, cands


def country_context(s1, pool, cands):
    """Everything that must see the full country: record-side context, IDF, name frequencies."""
    c = add_m_context(cands.rename({"j": "m_id_int"}).with_columns(pl.col("m_id_int").alias("m_id"))).drop("m_id")
    c = c.rename({"m_id_int": "j"})
    idf = {s1["country"][0]: (token_idf(s1["name_core"].to_list() + pool["name_core"].to_list()),
                              token_idf(s1["addr_text"].to_list() + pool["addr_text"].to_list()))}
    freq = name_freq_tables(s1, pool)
    return c, idf, freq


def iter_feature_chunks(s1, pool, c, idf, freq, s1_mask=None, chunk_pairs=1_500_000, labels=None):
    """Yields (chunk_no, n_chunks, features_df) with columns i, j, s1_id, m_id, f_*, [y]."""
    if s1_mask is not None:
        keep = s1.filter(s1_mask).select("i")
        c = c.join(keep, on="i", how="semi")
    n_chunks = max(1, -(-c.height // chunk_pairs))
    ids1 = s1.select("i", pl.col("entity_id").alias("s1_id"))
    ids2 = pool.select("j", pl.col("entity_id").alias("m_id"))
    for b in range(n_chunks):
        sub = c.filter((pl.col("i") % n_chunks) == b).join(ids1, on="i").join(ids2, on="j")
        f = build_features(sub.drop("i", "j"), s1, pool, idf, freq_tables=freq)
        f = f.join(sub.select("i", "j", *ID_COLS), on=ID_COLS)
        if labels is not None:
            f = f.join(labels, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0).cast(pl.Int8))
        yield b, n_chunks, f
        del sub, f
        gc.collect()


def save_features(split, country, role_name, s1_mask, labels=None):
    """Train split: write features for the S1 subset to work/p1/<split>/<country>/feats_<role>/part_*.parquet."""
    out_dir = WORK / "p1" / split / country / f"feats_{role_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1, pool, cands = load_country(split, country)
    c, idf, freq = country_context(s1, pool, cands)
    for b, n, f in iter_feature_chunks(s1, pool, c, idf, freq, s1_mask=s1_mask, labels=labels):
        part = out_dir / f"part_{b:03d}.parquet"
        if part.exists():
            continue
        write_parquet_atomic(f, part)
        log(f"  {split}/{country}/{role_name} chunk {b + 1}/{n}: {f.height} pairs, RSS {rss_gb():.1f} GB")
