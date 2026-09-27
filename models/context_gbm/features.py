"""Arch2 pair features (architecture doc sections 6, 7, 9, 14), vectorised.

Name (name_core): exact, Jaro-Winkler, Damerau-Levenshtein, Indel ratio, token sort/set ratio, partial ratio,
  char 3-gram and 4-gram TF-IDF cosine, token TF-IDF cosine, Soft TF-IDF (Cohen et al. 2003: tokens with
  Jaro-Winkler >= THETA contribute w_a * w_b * sim), Monge-Elkan (both directions), containment short to long,
  rare-token agreement (max capped IDF of agreeing tokens, IDF-weighted coverage), extra-token rarity (max and sum
  of capped IDF of record tokens with no close S1 token, and the reverse), phonetic agreement (name_phon).
Address (addr_text): exact, state equality, Indel and token set ratio, char 3/4-gram TF-IDF cosine, token TF-IDF
  cosine, Soft TF-IDF and Monge-Elkan (digit tokens match only exactly), BM25 with the S1 address as query (raw and
  normalised), coverage ratios, numeric agreement on addr_nums (exact set, subset both ways, Jaccard, conflict,
  house number = largest number: equal, edit distance, log abs difference, same length, same leading digits).
  PIN/ZIP agreement is not built: 5/6-digit postal tokens are absent from the normalised addresses (checked on
  300k train rows per country: 0.0% India, US 5-digit tokens are house numbers).
IDF / DF statistics come from the split's S1 + pool of one country (Corpus). IDF = log((N+1)/(df+1)) + 1, capped
at IDF_CAP so rare-token features do not depend on corpus size.
"""
import os

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer

THETA = 0.9          # Soft TF-IDF / containment token-similarity threshold
IDF_CAP = 10.0       # capped IDF (df / N below ~4.5e-5 all count as equally rare)
NF = 2 ** 20         # hashed char n-gram space
K1, BM_B = 1.2, 0.75
WORKERS = int(os.environ.get("A2_THREADS", "4"))
CHUNK = int(os.environ.get("A2_CHUNK", "250000"))

VEC_SPECS = {"n3": ("name_core", 3), "n4": ("name_core", 4), "a3": ("addr_text", 3), "a4": ("addr_text", 4)}


def _vec(n):
    return HashingVectorizer(analyzer="char_wb", ngram_range=(n, n), n_features=NF, alternate_sign=False, norm=None,
                             binary=True, lowercase=False, dtype=np.float32)


def _df_chunk(texts, n):
    X = _vec(n).transform(texts)
    return np.bincount(X.indices, minlength=NF).astype(np.int64)


def hashed_df(texts, n, workers=WORKERS, chunk=200_000):
    from joblib import Parallel, delayed
    parts = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    res = Parallel(n_jobs=workers)(delayed(_df_chunk)(p, n) for p in parts)
    return np.sum(res, axis=0)


def token_df(texts: pl.Series) -> pl.DataFrame:
    d = (pl.DataFrame({"t": texts}).with_row_index("r")
           .with_columns(pl.col("t").str.split(" ")).explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
           .unique(["r", "t"]).group_by("t").len("df"))
    return d


class Corpus:
    """DF statistics of one split-country (S1 + pool): token DF per field, hashed char n-gram DF, BM25 avgdl."""

    def __init__(self, s1: pl.DataFrame, pool: pl.DataFrame, cache_dir=None):
        import pickle
        f = cache_dir / "corpus.pkl" if cache_dir else None
        if f is not None and f.exists():
            self.__dict__.update(pickle.loads(f.read_bytes()))
            return
        names = pl.concat([s1["name_core"], pool["name_core"]]).fill_null("")
        addrs = pl.concat([s1["addr_text"], pool["addr_text"]]).fill_null("")
        self.N = len(names)
        self.tok = {"name": token_df(names), "addr": token_df(addrs)}
        self.avgdl = float(addrs.str.split(" ").list.len().mean())
        self.cdf = {}
        for k, (col, n) in VEC_SPECS.items():
            self.cdf[k] = hashed_df((names if col == "name_core" else addrs).to_list(), n)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(self.__dict__))

    def idf_table(self, field):
        N = self.N
        return self.tok[field].with_columns(
            (np.log((N + 1) / (pl.col("df") + 1)) + 1).clip(upper_bound=IDF_CAP).cast(pl.Float32).alias("idf"),
            np.log(1 + (N - pl.col("df") + 0.5) / (pl.col("df") + 0.5)).clip(upper_bound=IDF_CAP).cast(pl.Float32).alias("bidf"),
        ).select("t", "idf", "bidf")

    def char_idf(self, k):
        return np.minimum(np.log((self.N + 1) / (self.cdf[k] + 1)) + 1, IDF_CAP).astype(np.float32)


# ---------------------------------------------------------------- char n-gram TF-IDF cosine
def char_cosine(a: list, b: list, n: int, idf: np.ndarray) -> np.ndarray:
    """Row-wise cosine of hashed char n-gram TF-IDF vectors (binary tf) for aligned string lists."""
    from sklearn.preprocessing import normalize
    ua = {}
    ia = np.fromiter((ua.setdefault(s, len(ua)) for s in a), np.int64, len(a))
    ib = np.fromiter((ua.setdefault(s, len(ua)) for s in b), np.int64, len(b))
    X = _vec(n).transform(list(ua.keys()))
    X = normalize(X @ sp.diags(idf), norm="l2", copy=False).tocsr()
    return np.asarray(X[ia].multiply(X[ib]).sum(axis=1)).ravel().astype(np.float32)


# ---------------------------------------------------------------- token engine (Soft TF-IDF, Monge-Elkan, BM25, ...)
def _explode(s: pl.Series, idf: pl.DataFrame) -> pl.DataFrame:
    d = (pl.DataFrame({"t": s}).with_row_index("r").with_columns(pl.col("t").fill_null("").str.split(" "))
           .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
           .group_by("r", "t").len("tf")
           .join(idf, on="t", how="left")
           .with_columns(pl.col("idf").fill_null(IDF_CAP), pl.col("bidf").fill_null(IDF_CAP)))
    return d.with_columns((pl.col("idf") / (pl.col("idf") ** 2).sum().over("r").sqrt()).alias("w"),
                          pl.col("tf").sum().over("r").alias("dl"))


def token_features(a: pl.Series, b: pl.Series, idf: pl.DataFrame, pre: str, numeric_exact=False, bm25_avgdl=None):
    """a = S1 field, b = record field (aligned). Returns a DataFrame with len(a) rows."""
    n = len(a)
    A, B = _explode(a, idf), _explode(b, idf)
    voc = pl.concat([A.select("t"), B.select("t")]).unique(maintain_order=True).with_row_index("v")
    A = A.join(voc, on="t", how="left")
    B = B.join(voc, on="t", how="left")
    X = A.select("r", pl.col("v").alias("va"), pl.col("w").alias("wa"), pl.col("idf").alias("ia"), pl.col("bidf").alias("ba")).join(
        B.select("r", pl.col("v").alias("vb"), pl.col("w").alias("wb"), pl.col("idf").alias("ib"), pl.col("tf").alias("tfb"),
                 pl.col("dl").alias("dlb")), on="r")
    U = X.select("va", "vb").unique()
    vs = voc["t"].to_numpy()
    ta, tb = vs[U["va"].to_numpy()], vs[U["vb"].to_numpy()]
    sim = cpdist(ta.tolist(), tb.tolist(), scorer=JaroWinkler.normalized_similarity, workers=WORKERS).astype(np.float32)
    if numeric_exact:
        dig = voc.select(pl.col("t").str.contains(r"\d")).to_series().to_numpy()
        numeric = dig[U["va"].to_numpy()] | dig[U["vb"].to_numpy()]
        sim = np.where(numeric, (U["va"].to_numpy() == U["vb"].to_numpy()).astype(np.float32), sim)
    X = X.join(U.with_columns(pl.Series("sim", sim)), on=["va", "vb"], how="left").with_columns(
        (pl.col("va") == pl.col("vb")).alias("ex"))

    def side(key, w, i, wo):
        g = X.group_by("r", key).agg(pl.col("sim").max().alias("s"), pl.col(wo).sort_by("sim").last().alias("wo"),
                                     pl.col(w).first().alias("w"), pl.col(i).first().alias("i"))
        close = pl.col("s") >= THETA
        return g.group_by("r").agg(
            (pl.col("w") * pl.col("wo") * pl.col("s")).filter(close).sum().alias("soft"),
            pl.col("s").mean().alias("me"),
            close.mean().alias("cont"),
            pl.col("i").filter(close).max().fill_null(0).alias("rare"),
            (pl.col("i").filter(close).sum() / pl.col("i").sum()).alias("wcov"),
            pl.col("i").filter(~close).max().fill_null(0).alias("xmax"),
            pl.col("i").filter(~close).sum().alias("xsum"),
            (~close).sum().alias("xn"),
            pl.len().alias("n"))

    sa = side("va", "wa", "ia", "wb")
    sb = side("vb", "wb", "ib", "wa")
    ex = X.filter(pl.col("ex"))
    agg = [(pl.col("wa") * pl.col("wb")).sum().alias("tokcos")]
    if bm25_avgdl is not None:
        tf = pl.col("tfb")
        agg.append((pl.col("ba") * tf * (K1 + 1) / (tf + K1 * (1 - BM_B + BM_B * pl.col("dlb") / bm25_avgdl))).sum().alias("bm25"))
    ea = ex.group_by("r").agg(agg)
    out = pl.DataFrame({"r": np.arange(n, dtype=np.uint32)})
    out = (out.join(sa.rename({c: f"{c}_ab" for c in sa.columns if c != "r"}), on="r", how="left")
              .join(sb.rename({c: f"{c}_ba" for c in sb.columns if c != "r"}), on="r", how="left")
              .join(ea, on="r", how="left"))
    na, nb = pl.col("n_ab").fill_null(0), pl.col("n_ba").fill_null(0)
    cols = [
        pl.col("soft_ab").fill_null(0).alias(f"{pre}soft_ab"),
        pl.col("soft_ba").fill_null(0).alias(f"{pre}soft_ba"),
        ((pl.col("soft_ab").fill_null(0) + pl.col("soft_ba").fill_null(0)) / 2).alias(f"{pre}soft"),
        pl.col("me_ab").fill_null(-1).alias(f"{pre}me_ab"),
        pl.col("me_ba").fill_null(-1).alias(f"{pre}me_ba"),
        pl.when(na <= nb).then(pl.col("cont_ab")).otherwise(pl.col("cont_ba")).fill_null(-1).alias(f"{pre}cont_s2l"),
        pl.col("cont_ab").fill_null(-1).alias(f"{pre}cont_ab"),
        pl.col("cont_ba").fill_null(-1).alias(f"{pre}cont_ba"),
        pl.max_horizontal(pl.col("rare_ab"), pl.col("rare_ba")).fill_null(0).alias(f"{pre}rare"),
        pl.col("wcov_ab").fill_null(-1).alias(f"{pre}wcov_ab"),
        pl.col("wcov_ba").fill_null(-1).alias(f"{pre}wcov_ba"),
        pl.col("xmax_ba").fill_null(0).alias(f"{pre}xrare_b"),
        pl.col("xsum_ba").fill_null(0).alias(f"{pre}xsum_b"),
        pl.col("xn_ba").fill_null(0).alias(f"{pre}xn_b"),
        pl.col("xmax_ab").fill_null(0).alias(f"{pre}xrare_a"),
        pl.col("xsum_ab").fill_null(0).alias(f"{pre}xsum_a"),
        pl.col("xn_ab").fill_null(0).alias(f"{pre}xn_a"),
        pl.col("tokcos").fill_null(0).alias(f"{pre}tokcos"),
        na.alias(f"{pre}ntok_a"), nb.alias(f"{pre}ntok_b"),
    ]
    if bm25_avgdl is not None:
        cols += [pl.col("bm25").fill_null(0).alias(f"{pre}bm25")]
    res = out.sort("r").select(cols)
    if bm25_avgdl is not None:
        # normaliser: BM25 of the query against itself-sized document (sum of the query tokens' BM25 IDF)
        qn = A.group_by("r").agg(pl.col("bidf").sum().alias("q")).join(pl.DataFrame({"r": np.arange(n, dtype=np.uint32)}), on="r", how="right")
        q = qn.sort("r")["q"].fill_null(0).to_numpy()
        res = res.with_columns((pl.col(f"{pre}bm25") / np.maximum(q, 1e-6)).alias(f"{pre}bm25n"))
    return res.with_columns(pl.all().cast(pl.Float32))


# ---------------------------------------------------------------- numbers
def number_features(na: pl.Series, nb: pl.Series) -> pl.DataFrame:
    d = pl.DataFrame({"a": na.fill_null(""), "b": nb.fill_null("")}).with_columns(
        pl.col("a").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("la"),
        pl.col("b").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("lb"))
    d = d.with_columns(
        pl.col("la").list.len().alias("n_a"), pl.col("lb").list.len().alias("n_b"),
        pl.col("la").list.set_intersection("lb").list.len().alias("inter"),
        pl.col("la").list.set_union("lb").list.len().alias("union"),
        pl.col("la").list.eval(pl.element().cast(pl.Int64, strict=False)).list.max().alias("ha"),
        pl.col("lb").list.eval(pl.element().cast(pl.Int64, strict=False)).list.max().alias("hb"))
    both = (pl.col("n_a") > 0) & (pl.col("n_b") > 0)
    hs_a, hs_b = pl.col("ha").cast(pl.String), pl.col("hb").cast(pl.String)
    d = d.with_columns(hs_a.fill_null("").alias("hsa"), hs_b.fill_null("").alias("hsb"))
    ed = cpdist(d["hsa"].to_list(), d["hsb"].to_list(), scorer=Levenshtein.distance, workers=WORKERS)
    hb = pl.col("ha").is_not_null() & pl.col("hb").is_not_null()
    out = d.select(
        pl.col("n_a").alias("num_n_a"), pl.col("n_b").alias("num_n_b"),
        pl.when(both).then(pl.col("inter") / pl.col("union")).otherwise(-1).alias("num_jacc"),
        (both & (pl.col("inter") == pl.col("n_a")) & (pl.col("inter") == pl.col("n_b"))).alias("num_eq"),
        (both & (pl.col("inter") == pl.col("n_a")) & (pl.col("n_b") > pl.col("n_a"))).alias("num_a_sub_b"),
        (both & (pl.col("inter") == pl.col("n_b")) & (pl.col("n_a") > pl.col("n_b"))).alias("num_b_sub_a"),
        (both & (pl.col("inter") == 0)).alias("num_conflict"),
        pl.when(hb).then((pl.col("ha") == pl.col("hb")).cast(pl.Int8)).otherwise(-1).alias("hn_eq"),
        pl.when(hb).then((pl.col("ha") - pl.col("hb")).abs().cast(pl.Float64).log1p()).otherwise(-1).alias("hn_logdiff"),
        pl.when(hb).then((pl.col("hsa").str.len_chars() == pl.col("hsb").str.len_chars()).cast(pl.Int8)).otherwise(-1).alias("hn_samelen"),
        pl.when(hb).then((pl.col("ha") // 10 == pl.col("hb") // 10).cast(pl.Int8)).otherwise(-1).alias("hn_same_prefix"),
        pl.when(hb).then((pl.col("hsa").str.slice(0, 1) == pl.col("hsb").str.slice(0, 1)).cast(pl.Int8)).otherwise(-1).alias("hn_first_digit_eq"),
    ).with_columns(pl.Series("hn_edit", ed.astype(np.float32)))
    out = out.with_columns(pl.when(pl.col("hn_eq") == -1).then(-1).otherwise(pl.col("hn_edit")).alias("hn_edit"))
    return out.with_columns(pl.all().cast(pl.Float32))


# ---------------------------------------------------------------- main entry
def pair_texts(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    a = s1.rename({c: f"{c}_a" for c in s1.columns if c != "entity_id"}).rename({"entity_id": "s1_id"})
    b = pool.rename({c: f"{c}_b" for c in pool.columns if c != "entity_id"}).rename({"entity_id": "m_id"})
    d = pairs.select("s1_id", "m_id").join(a, on="s1_id", how="left").join(b, on="m_id", how="left")
    return d.with_columns(pl.col(pl.String).exclude("s1_id", "m_id").fill_null(""))


def _cp(a, b, scorer, scale=1.0):
    return (cpdist(a, b, scorer=scorer, workers=WORKERS) * scale).astype(np.float32)


def chunk_features(d: pl.DataFrame, corpus: Corpus, idf_name: pl.DataFrame, idf_addr: pl.DataFrame, cidf: dict) -> pl.DataFrame:
    na, nb = d["name_core_a"].to_list(), d["name_core_b"].to_list()
    aa, ab = d["addr_text_a"].to_list(), d["addr_text_b"].to_list()
    pa, pb = d["name_phon_a"].to_list(), d["name_phon_b"].to_list()
    f = {
        "nm_exact": (d["name_core_a"] == d["name_core_b"]).cast(pl.Float32).to_numpy(),
        "nm_jw": _cp(na, nb, JaroWinkler.normalized_similarity),
        "nm_dl": _cp(na, nb, DamerauLevenshtein.normalized_similarity),
        "nm_ratio": _cp(na, nb, fuzz.ratio, 0.01),
        "nm_tsort": _cp(na, nb, fuzz.token_sort_ratio, 0.01),
        "nm_tset": _cp(na, nb, fuzz.token_set_ratio, 0.01),
        "nm_partial": _cp(na, nb, fuzz.partial_ratio, 0.01),
        "nm_c3": char_cosine(na, nb, 3, cidf["n3"]),
        "nm_c4": char_cosine(na, nb, 4, cidf["n4"]),
        "ph_eq": ((d["name_phon_a"] == d["name_phon_b"]) & (d["name_phon_a"] != "")).cast(pl.Float32).to_numpy(),
        "ph_jw": _cp(pa, pb, JaroWinkler.normalized_similarity),
        "ph_tset": _cp(pa, pb, fuzz.token_set_ratio, 0.01),
        "ad_exact": ((d["addr_text_a"] == d["addr_text_b"]) & (d["addr_text_a"] != "")).cast(pl.Float32).to_numpy(),
        "ad_state_eq": (d["addr_state_a"] == d["addr_state_b"]).cast(pl.Float32).to_numpy(),
        "ad_empty_b": (d["addr_text_b"] == "").cast(pl.Float32).to_numpy(),
        "ad_ratio": _cp(aa, ab, fuzz.ratio, 0.01),
        "ad_tset": _cp(aa, ab, fuzz.token_set_ratio, 0.01),
        "ad_c3": char_cosine(aa, ab, 3, cidf["a3"]),
        "ad_c4": char_cosine(aa, ab, 4, cidf["a4"]),
    }
    base = pl.DataFrame(f)
    tn = token_features(d["name_core_a"], d["name_core_b"], idf_name, "nm_")
    ta = token_features(d["addr_text_a"], d["addr_text_b"], idf_addr, "ad_", numeric_exact=True, bm25_avgdl=corpus.avgdl)
    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    ph = (pl.DataFrame({"a": d["name_phon_a"], "b": d["name_phon_b"]})
            .with_columns(tok("a").list.set_intersection(tok("b")).list.len().alias("i"),
                          tok("a").list.set_union(tok("b")).list.len().alias("u"))
            .select((pl.col("i") / pl.col("u").clip(lower_bound=1)).cast(pl.Float32).alias("ph_jacc")))
    nums = number_features(d["addr_nums_a"], d["addr_nums_b"])
    return pl.concat([d.select("s1_id", "m_id"), base, tn, ta, ph, nums], how="horizontal")


def build(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, corpus: Corpus, log=print) -> pl.DataFrame:
    idf_name, idf_addr = corpus.idf_table("name"), corpus.idf_table("addr")
    cidf = {k: corpus.char_idf(k) for k in VEC_SPECS}
    d = pair_texts(pairs, s1, pool)
    outs = []
    for s in range(0, d.height, CHUNK):
        outs.append(chunk_features(d.slice(s, CHUNK), corpus, idf_name, idf_addr, cidf))
        log(f"  feats {min(s + CHUNK, d.height)}/{d.height}")
    return pl.concat(outs)
