"""Candidate generation (blocking), always within one country.

Several top-k TF-IDF searches are unioned (Sparkly, PVLDB 2023: top-k tf/idf is a strong blocker):
  name   : char 3-grams of core name (+ f/k/a alternative names)
  phon   : char 2-3-grams of the phonetic key (catches transliterated / misspelled names)
  addr   : char 3-grams of the canonical address (catches trade-name matches at the same address)
  comb   : name and address together
  rev    : for each S2/S3 record, its closest S1s by comb (each record belongs to at most one S1)
The cosine of every method is kept for every candidate pair (they become features).
"""
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn


@dataclass
class BlockCfg:
    k_name: int = 15
    k_phon: int = 10
    k_addr: int = 15
    k_comb: int = 20
    k_rev: int = 3
    w_name: float = 0.6        # weight of name vs address in the combined vector
    threads: int = 7
    chunk: int = 50_000


def _texts(df: pl.DataFrame):
    name = (df["name_core"] + pl.Series([" "] * df.height) + df["name_alts"].str.replace_all(r"\|", " ")).str.strip_chars()
    return name.to_list(), df["name_phon"].to_list(), df["addr_text"].to_list()


def _vec(analyzer, ngram, a_texts, b_texts, max_df=1.0):
    # max_df prunes n-grams present in many records: they carry little IDF weight but make top-k search quadratic
    v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, max_df=max_df, dtype=np.float32, sublinear_tf=True)
    v.fit(a_texts + b_texts)
    return v.transform(a_texts), v.transform(b_texts)


def _topk(A, B, k, threads, chunk):
    """Row-chunked top-k of A @ B.T. Returns (row_idx, col_idx, score) arrays."""
    BT = B.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[s:s + chunk], BT, top_n=k, threshold=0.0, n_threads=threads).tocoo()
        rows.append(C.row + s); cols.append(C.col); vals.append(C.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def rowwise_cos(A, B, ia, ib, batch=250_000):
    """cosine between A[ia[i]] and B[ib[i]] for aligned index arrays (both L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), batch):
        out[s:s + batch] = np.asarray(A[ia[s:s + batch]].multiply(B[ib[s:s + batch]]).sum(axis=1)).ravel()
    return out


def block_country(s1: pl.DataFrame, pool: pl.DataFrame, cfg: BlockCfg):
    """Returns candidate pairs DataFrame [s1_id, m_id, cos_name, cos_phon, cos_addr, cos_comb, via_*]."""
    a_name, a_phon, a_addr = _texts(s1)
    b_name, b_phon, b_addr = _texts(pool)
    An, Bn = _vec("char_wb", (3, 3), a_name, b_name)
    Ap, Bp = _vec("char_wb", (2, 3), a_phon, b_phon)
    Aa, Ba = _vec("char_wb", (3, 3), a_addr, b_addr)
    w = cfg.w_name
    Ac = l2norm(sp.hstack([An * np.sqrt(w), Aa * np.sqrt(1 - w)]).tocsr())
    Bc = l2norm(sp.hstack([Bn * np.sqrt(w), Ba * np.sqrt(1 - w)]).tocsr())

    found = {}
    for tag, A, B, k in [("name", An, Bn, cfg.k_name), ("phon", Ap, Bp, cfg.k_phon),
                         ("addr", Aa, Ba, cfg.k_addr), ("comb", Ac, Bc, cfg.k_comb)]:
        r, c, _ = _topk(A, B, k, cfg.threads, cfg.chunk)
        found[tag] = (r, c)
    c, r, _ = _topk(Bc, Ac, cfg.k_rev, cfg.threads, cfg.chunk)      # reverse: pool -> S1
    found["rev"] = (r, c)

    keys = []
    for tag, (r, c) in found.items():
        keys.append(pl.DataFrame({"i": r.astype(np.int64), "j": c.astype(np.int64), "via": [tag] * len(r)}))
    pairs = (pl.concat(keys).group_by(["i", "j"]).agg(pl.col("via").unique().sort().str.join("+").alias("via")))
    ia, ib = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    pairs = pairs.with_columns(
        pl.Series("cos_name", rowwise_cos(An, Bn, ia, ib)),
        pl.Series("cos_phon", rowwise_cos(Ap, Bp, ia, ib)),
        pl.Series("cos_addr", rowwise_cos(Aa, Ba, ia, ib)),
        pl.Series("cos_comb", rowwise_cos(Ac, Bc, ia, ib)),
    )
    s1_ids = s1["entity_id"].to_numpy()
    m_ids = pool["entity_id"].to_numpy()
    return pairs.with_columns(pl.Series("s1_id", s1_ids[ia]), pl.Series("m_id", m_ids[ib])).drop("i", "j")


def block(s1: pl.DataFrame, pool: pl.DataFrame, cfg: BlockCfg = BlockCfg()):
    """Run per country (open set of labels: whatever countries appear in s1)."""
    out = []
    for country in s1["country"].unique().to_list():
        a = s1.filter(pl.col("country") == country)
        b = pool.filter(pl.col("country") == country)
        if a.height and b.height:
            out.append(block_country(a, b, cfg))
    return pl.concat(out)


# ============================================================================ v2: state-partitioned blocking
# (review #1: block within (country, state); Telangana/Andhra Pradesh merged; country-wide name-only path for
#  records without a usable state; extra keys for domain-style and acronym names)
from normalize import acronym_key, concat_key  # noqa: E402

STATE_MERGE = {"ts": "ap"}          # matched pairs disagree on state only for Telangana vs Andhra Pradesh


@dataclass
class BlockCfgV2:
    k_name: int = 15
    k_phon: int = 10
    k_addr: int = 15
    k_comb: int = 20
    k_cat: int = 5
    k_rev: int = 3
    k_stateless: int = 5          # stateless pool record -> top S1s of the whole country, per method
    w_name: float = 0.6
    min_state_coverage: float = 0.5   # below this, the country is one partition (e.g. France)
    max_df: float = 0.02              # drop n-grams in > 2% of records (10-30x faster top-k, measured)
    verbose: bool = True
    threads: int = 7
    chunk: int = 50_000


def _pairs_from(r, c, a_idx, b_idx, tag):
    return pl.DataFrame({"i": a_idx[r].astype(np.int64), "j": b_idx[c].astype(np.int64), "via": [tag] * len(r)})


def block_country_v2(s1: pl.DataFrame, pool: pl.DataFrame, cfg: BlockCfgV2):
    import time
    from utils import log
    T = [time.time()]

    def tick(msg):
        if cfg.verbose:
            log(f"    [{s1['country'][0]}] {msg}: {time.time() - T[0]:.1f}s")
        T[0] = time.time()

    a_name, a_phon, a_addr = _texts(s1)
    b_name, b_phon, b_addr = _texts(pool)
    a_cat = [concat_key(x) for x in s1["name_core"].to_list()]
    b_cat = [concat_key(x) for x in pool["name_core"].to_list()]
    tick("texts")
    md = cfg.max_df
    An, Bn = _vec("char_wb", (3, 3), a_name, b_name, md)
    Ap, Bp = _vec("char_wb", (3, 3), a_phon, b_phon, md)
    Aa, Ba = _vec("char_wb", (3, 3), a_addr, b_addr, md)
    At, Bt = _vec("char", (3, 3), a_cat, b_cat, md)
    tick("tf-idf vectors")
    w = cfg.w_name
    Ac = l2norm(sp.hstack([An * np.sqrt(w), Aa * np.sqrt(1 - w)]).tocsr()).astype(np.float32)
    Bc = l2norm(sp.hstack([Bn * np.sqrt(w), Ba * np.sqrt(1 - w)]).tocsr()).astype(np.float32)

    sa = np.array([STATE_MERGE.get(x, x) for x in s1["addr_state"].to_list()])
    sb = np.array([STATE_MERGE.get(x, x) for x in pool["addr_state"].to_list()])
    partitioned = (sa != "").mean() >= cfg.min_state_coverage
    if not partitioned:
        sa[:] = "*"; sb[:] = "*"
    parts = []
    for st in np.unique(sa):
        ai = np.where(sa == st)[0]
        bi = np.where(sb == st)[0] if st != "" else np.arange(len(sb))   # stateless S1: whole-country pool
        if len(ai) == 0 or len(bi) == 0:
            continue
        for tag, A, B, k in [("name", An, Bn, cfg.k_name), ("phon", Ap, Bp, cfg.k_phon), ("addr", Aa, Ba, cfg.k_addr),
                             ("comb", Ac, Bc, cfg.k_comb), ("cat", At, Bt, cfg.k_cat)]:
            r, c, _ = _topk(A[ai], B[bi], k, cfg.threads, cfg.chunk)
            parts.append(_pairs_from(r, c, ai, bi, tag))
        c, r, _ = _topk(Bc[bi], Ac[ai], cfg.k_rev, cfg.threads, cfg.chunk)
        parts.append(_pairs_from(r, c, ai, bi, "rev"))
    tick(f"top-k over {len(np.unique(sa))} partitions")
    # stateless pool records (empty / unparseable address): name-only search over the whole country's S1
    if partitioned:
        bi = np.where(sb == "")[0]
        ai = np.arange(len(sa))
        if len(bi):
            for tag, A, B in [("sl_name", An, Bn), ("sl_phon", Ap, Bp), ("sl_cat", At, Bt)]:
                c, r, _ = _topk(B[bi], A[ai], cfg.k_stateless, cfg.threads, cfg.chunk)
                parts.append(_pairs_from(r, c, ai, bi, tag))
    tick("stateless path")
    # acronym key: pool name equals the S1 name's initials, same partition, then best 3 by address
    acr = pl.DataFrame({"i": np.arange(len(sa)), "acr": [acronym_key(x) for x in s1["name_core"].to_list()], "st": sa})
    pc = pl.DataFrame({"j": np.arange(len(sb)), "acr": [x if 2 <= len(x) <= 5 and " " not in x else "" for x in pool["name_core"].to_list()], "st": sb})
    aj = acr.filter(pl.col("acr") != "").join(pc.filter(pl.col("acr") != ""), on=["acr", "st"])
    if aj.height:
        aj = aj.with_columns(pl.Series("s", rowwise_cos(Aa, Ba, aj["i"].to_numpy(), aj["j"].to_numpy())))
        aj = aj.filter(pl.col("s").rank("ordinal", descending=True).over("j") <= 3)
        parts.append(aj.select(pl.col("i").cast(pl.Int64), pl.col("j").cast(pl.Int64), pl.lit("acr").alias("via")))

    tick("acronym key")
    pairs = pl.concat(parts).group_by(["i", "j"]).agg(pl.col("via").unique().sort().str.join("+").alias("via"))
    tick(f"union -> {pairs.height} pairs")
    ia, ib = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    pairs = pairs.with_columns(
        pl.Series("cos_name", rowwise_cos(An, Bn, ia, ib)),
        pl.Series("cos_phon", rowwise_cos(Ap, Bp, ia, ib)),
        pl.Series("cos_addr", rowwise_cos(Aa, Ba, ia, ib)),
        pl.Series("cos_comb", rowwise_cos(Ac, Bc, ia, ib)),
        pl.Series("cos_cat", rowwise_cos(At, Bt, ia, ib)),
    )
    tick("pair cosines")
    s1_ids, m_ids = s1["entity_id"].to_numpy(), pool["entity_id"].to_numpy()
    return pairs.with_columns(pl.Series("s1_id", s1_ids[ia]), pl.Series("m_id", m_ids[ib])).drop("i", "j")


def block_v2(s1: pl.DataFrame, pool: pl.DataFrame, cfg: BlockCfgV2 = BlockCfgV2()):
    out = []
    for country in s1["country"].unique().to_list():
        a = s1.filter(pl.col("country") == country)
        b = pool.filter(pl.col("country") == country)
        if a.height and b.height:
            out.append(block_country_v2(a, b, cfg))
    return pl.concat(out)
