"""Phase 1, stage 1: full-scale candidate generation for one split (train or test), memory-lean.

Same logic as the Phase 0 validated blocking v2 (blocking.block_country_v2), restructured for 1.7M x 10M:
  - TF-IDF vocab/IDF fitted on a sample (<= 1.5M texts) per country and method
  - each state partition is vectorised, searched and freed before the next one
  - ids are int32 row indices into work/p1/<split>/<country>/{s1,pool}.parquet
Outputs per country: s1.parquet, pool.parquet, cands.parquet (i, j, via, cos_name, cos_phon, cos_addr, cos_comb, cos_cat)
Checkpoint: a country is skipped if its cands.parquet exists (use --force). No auto-resume.
Usage: caffeinate -i -s -m python p1_block.py --split test [--countries India,US,France]
"""
import argparse
import gc
import os as _os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm

from blocking import STATE_MERGE, BlockCfgV2, _topk, rowwise_cos
from normalize import acronym_key, concat_key
from utils import WORK, announce_pid, done, log, resource_guard, rss_gb, write_parquet_atomic

VIA_BITS = {t: 1 << k for k, t in enumerate(["name", "phon", "addr", "comb", "cat", "rev", "acr", "sl_name", "sl_phon",
                                             "sl_cat", "sl_addr", "sl_comb", "s1sl_name", "s1sl_phon", "s1sl_cat", "s1sl_comb",
                                             "x_name", "x_phon", "x_addr", "x_numtok", "x_sl"])}
# records without a usable state: top-k S1 of the whole country per search method. v1-v3 used 5; validation showed
# recall on such records was only 74-78% (vs 98-99.5% with a state), so v4 uses 20.
K_STATELESS = int(_os.environ.get("ER_K_STATELESS", 20))
STATELESS_EXACT = _os.environ.get("ER_STATELESS_EXACT", "1") == "1"   # v1-v3 models were trained without it

KEEP_S1 = ["entity_id", "country", "name_core", "name_legal", "name_alts", "name_phon", "name_is_domain", "name_has_alt",
           "name_non_latin", "addr_text", "addr_state", "addr_nums", "addr_empty", "addr_native_state"]


def texts(df):
    """Compact (Arrow) text columns; only the rows of one partition are turned into Python strings at a time."""
    name = (df["name_core"] + " " + df["name_alts"].str.replace_all(r"\|", " ")).str.strip_chars()
    cat = df["name_core"].str.replace_all(r"\b(and|the|of)\b", "").str.replace_all(" ", "")
    return {"name": name, "phon": df["name_phon"], "addr": df["addr_text"], "cat": cat}


SPEC = {"name": ("char_wb", (3, 3)), "phon": ("char_wb", (3, 3)), "addr": ("char_wb", (3, 3)), "cat": ("char", (3, 3))}


def fit_vectorizers(ta, tb, max_df, sample=1_500_000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m, (an, ng) in SPEC.items():
        both = pl.concat([ta[m], tb[m]])
        idx = rng.choice(len(both), min(sample, len(both)), replace=False)
        corpus = both.gather(idx).to_list()
        v = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=2, max_df=max_df, dtype=np.float32, sublinear_tf=True)
        out[m] = v.fit(corpus)
    return out


def vecs_for(V, t, rows, w):
    sub = {m: t[m].gather(rows).to_list() for m in SPEC}
    X = {m: V[m].transform(sub[m]) for m in SPEC}
    X["comb"] = l2norm(sp.hstack([X["name"] * np.sqrt(w), X["addr"] * np.sqrt(1 - w)]).tocsr()).astype(np.float32)
    return X


# max records an n-gram may appear in (within a partition) to be used for SEARCH; names are short (cheap), addresses long
SEARCH_CAPS = {"name": int(_os.environ.get("ER_CAP_NAME", 6000)), "phon": int(_os.environ.get("ER_CAP_NAME", 6000)),
               "cat": int(_os.environ.get("ER_CAP_NAME", 6000)), "addr": int(_os.environ.get("ER_CAP_ADDR", 6000))}


def prune_for_search(XA, XB, frac=0.02, min_rows=5000):
    """Search-only copies without n-grams that are common INSIDE this partition (e.g. the local city's letters).

    Country-level max_df misses them; left in, they make top-k search inside big states nearly quadratic.
    Similarity features keep using the unpruned vectors.
    """
    PA, PB = {}, {}
    for m in ["name", "phon", "addr", "cat"]:
        A, B = XA[m], XB[m]
        if B.shape[0] < min_rows:
            PA[m], PB[m] = A, B
            continue
        df = np.bincount(B.indices, minlength=B.shape[1]) + np.bincount(A.indices, minlength=A.shape[1])
        cap = min(frac * (A.shape[0] + B.shape[0]), SEARCH_CAPS[m])   # absolute cap keeps cost ~linear in size
        keep = sp.diags((df <= cap).astype(np.float32))
        PA[m], PB[m] = l2norm(A @ keep).astype(np.float32), l2norm(B @ keep).astype(np.float32)
    PA["comb"] = XA["comb"] if PA["name"] is XA["name"] else l2norm(sp.hstack([PA["name"] * np.sqrt(0.6), PA["addr"] * np.sqrt(0.4)]).tocsr()).astype(np.float32)
    PB["comb"] = XB["comb"] if PB["name"] is XB["name"] else l2norm(sp.hstack([PB["name"] * np.sqrt(0.6), PB["addr"] * np.sqrt(0.4)]).tocsr()).astype(np.float32)
    return PA, PB


EXACT_KEYS = [("name_core", "x_name"), ("name_phon", "x_phon"), ("addr_text", "x_addr")]


def exact_key_pairs(s1, pool, sa, sb, max_s1=30, max_pool=150):
    """Cheap hash joins on exact normalised keys within a state (recall safety for names made only of common
    n-grams, which the capped search can miss). Oversized key groups (chains, empty strings) are skipped."""
    out = []
    A = pl.DataFrame({"i": np.arange(len(sa), dtype=np.int32), "st": sa})
    B = pl.DataFrame({"j": np.arange(len(sb), dtype=np.int32), "st": sb})
    for col, tag in EXACT_KEYS:
        a = A.with_columns(s1[col].alias("k")).filter((pl.col("k").str.len_chars() >= 4) & (pl.col("st") != ""))
        b = B.with_columns(pool[col].alias("k")).filter((pl.col("k").str.len_chars() >= 4) & (pl.col("st") != ""))
        ga = a.group_by(["st", "k"]).len().filter(pl.col("len") <= max_s1).drop("len")
        gb = b.group_by(["st", "k"]).len().filter(pl.col("len") <= max_pool).drop("len")
        keys = ga.join(gb, on=["st", "k"])
        pr = a.join(keys, on=["st", "k"]).join(b, on=["st", "k"]).select("i", "j")
        out.append(pr.with_columns(pl.lit(VIA_BITS[tag], pl.Int32).alias("via")))
    return pl.concat(out).group_by(["i", "j"]).agg(pl.col("via").unique().sum())


def stateless_exact_pairs(s1, pool, sb, max_s1=30):
    """Records without a usable state: exact normalised name / sound-alike name matched against the S1 of the WHOLE
    country, when at most max_s1 businesses share that key (9-17% of the stateless misses on validation had an exact
    S1 name shared by <= 30 businesses; very common names are skipped, they cannot be resolved by name alone)."""
    out = []
    A = pl.DataFrame({"i": np.arange(s1.height, dtype=np.int32)})
    B = pl.DataFrame({"j": np.arange(len(sb), dtype=np.int32), "st": sb})
    for col in ["name_core", "name_phon"]:
        a = A.with_columns(s1[col].alias("k")).filter(pl.col("k").str.len_chars() >= 4)
        b = B.with_columns(pool[col].alias("k")).filter((pl.col("k").str.len_chars() >= 4) & (pl.col("st") == ""))
        ga = a.group_by("k").len().filter(pl.col("len") <= max_s1).drop("len")
        out.append(a.join(ga, on="k").join(b.select("j", "k"), on="k").select("i", "j"))
    return pl.concat(out).unique().with_columns(pl.lit(VIA_BITS["x_sl"], pl.Int32).alias("via"))


def infer_missing_states(s1, pool, min_n=20, purity=0.97):
    """Give a state/region to records whose address names a city but no state.

    35% of French test records look like "70 rue chanoine poupard nantes": the city is there, the region is not.
    In training data a missing state always came with an EMPTY address, so the model never saw this pattern, and
    such records were only searched through the weak country-wide path. The clean S1 reference of the same split
    tells us which region each address word belongs to ("nantes" -> pdl, "lille" -> hdf): a word is used when it
    appears in >= min_n S1 addresses and >= purity of them share one state. A record gets the state its mapped words
    vote for (>= 80% agreement). No labels and no external data are used. Returns (s1, pool, n_s1, n_pool)."""
    words = (s1.filter(pl.col("addr_state") != "")
               .select(pl.col("addr_state").alias("st"), pl.col("addr_text").str.split(" ").list.unique().alias("w"))
               .explode("w").filter((pl.col("w").str.len_chars() >= 3) & ~pl.col("w").str.contains(r"^\d+$")))
    g = words.group_by("w", "st").len()
    wmap = (g.group_by("w").agg(pl.col("len").sum().alias("n"), pl.col("len").max().alias("m"),
                                pl.col("st").sort_by("len").last().alias("st"))
             .filter((pl.col("n") >= min_n) & (pl.col("m") / pl.col("n") >= purity)).select("w", "st"))

    def fill(df):
        need = df.with_row_index("r").filter((pl.col("addr_state") == "") & (pl.col("addr_text") != ""))
        v = (need.select("r", pl.col("addr_text").str.split(" ").list.unique().alias("w")).explode("w")
                 .join(wmap, on="w").group_by("r", "st").len())
        best = (v.sort("len", descending=True).group_by("r")
                 .agg(pl.col("st").first().alias("inf"), (pl.col("len").first() / pl.col("len").sum()).alias("share"))
                 .filter(pl.col("share") >= 0.8).select("r", "inf"))
        out = (df.with_row_index("r").join(best, on="r", how="left")
                 .with_columns(pl.coalesce(pl.col("inf"), pl.col("addr_state")).alias("addr_state"),
                               # the region code is appended to the address too, as in records that carry it
                               pl.when(pl.col("inf").is_not_null()).then(pl.col("addr_text") + " " + pl.col("inf"))
                               .otherwise(pl.col("addr_text")).alias("addr_text")).drop("r", "inf"))
        return out, best.height

    s1, n1 = fill(s1)
    pool, n2 = fill(pool)
    return s1, pool, n1, n2


def numtok_pairs(s1, pool, sa, sb, max_s1=10, max_pool=40):
    """Key = (state, sound-alike name word, house/plot number). Recovers common-name pairs with partial addresses
    that the capped TF-IDF search misses ("5un enterprises"+110, "vn prodkts"+208)."""
    def keys(df, st, idx_name):
        return (pl.DataFrame({idx_name: np.arange(len(st), dtype=np.int32), "st": st,
                              "tok": df["name_phon"].str.split(" "), "num": df["addr_nums"].str.split(" ")})
                .filter(pl.col("st") != "").explode("tok").explode("num")
                .filter((pl.col("tok").str.len_chars() >= 3) & (pl.col("num").str.len_chars() >= 1)).unique())
    a, b = keys(s1, sa, "i"), keys(pool, sb, "j")
    ka = a.group_by(["st", "tok", "num"]).len().filter(pl.col("len") <= max_s1).drop("len")
    kb = b.group_by(["st", "tok", "num"]).len().filter(pl.col("len") <= max_pool).drop("len")
    k = ka.join(kb, on=["st", "tok", "num"])
    pr = a.join(k, on=["st", "tok", "num"]).join(b, on=["st", "tok", "num"]).select("i", "j").unique()
    return pr.with_columns(pl.lit(VIA_BITS["x_numtok"], pl.Int32).alias("via"))


def cos_all(XA, XB, li, lj):
    return {f"cos_{m}": rowwise_cos(XA[m], XB[m], li, lj) for m in ["name", "phon", "addr", "comb", "cat"]}


def pairs_df(i, j, via, cos):
    return pl.DataFrame({"i": i.astype(np.int32), "j": j.astype(np.int32), "via": pl.Series(via).cast(pl.Int32), **cos})


def block_country(s1, pool, cfg: BlockCfgV2):
    T = time.time()
    ta, tb = texts(s1), texts(pool)
    V = fit_vectorizers(ta, tb, cfg.max_df)
    log(f"  vectorizers fitted ({time.time() - T:.0f}s)")
    sa = np.array([STATE_MERGE.get(x, x) for x in s1["addr_state"].to_list()])
    sb = np.array([STATE_MERGE.get(x, x) for x in pool["addr_state"].to_list()])
    partitioned = (sa != "").mean() >= cfg.min_state_coverage
    if not partitioned:
        sa[:] = "*"; sb[:] = "*"
    parts = []
    states = [s for s in np.unique(sa) if s != ""]
    for n_st, st in enumerate(states):
        ai, bi = np.where(sa == st)[0], np.where(sb == st)[0]
        if len(ai) == 0 or len(bi) == 0:
            continue
        tp = time.time()
        XA, XB = vecs_for(V, ta, ai, cfg.w_name), vecs_for(V, tb, bi, cfg.w_name)
        PA, PB = prune_for_search(XA, XB, cfg.max_df)
        found = []
        for m, k in [("name", cfg.k_name), ("phon", cfg.k_phon), ("addr", cfg.k_addr), ("comb", cfg.k_comb), ("cat", cfg.k_cat)]:
            r, c, _ = _topk(PA[m], PB[m], k, cfg.threads, cfg.chunk)
            found.append(pl.DataFrame({"li": r.astype(np.int32), "lj": c.astype(np.int32), "via": np.full(len(r), VIA_BITS[m], np.int32)}))
        c, r, _ = _topk(PB["comb"], PA["comb"], cfg.k_rev, cfg.threads, cfg.chunk)
        found.append(pl.DataFrame({"li": r.astype(np.int32), "lj": c.astype(np.int32), "via": np.full(len(r), VIA_BITS["rev"], np.int32)}))
        u = pl.concat(found).group_by(["li", "lj"]).agg(pl.col("via").unique().sum())
        li, lj = u["li"].to_numpy(), u["lj"].to_numpy()
        parts.append(pairs_df(ai[li], bi[lj], u["via"], cos_all(XA, XB, li, lj)))
        del XA, XB, PA, PB, found, u
        gc.collect()
        if len(ai) > 2000:
            log(f"  partition {n_st + 1}/{len(states)} '{st}': {len(ai)} S1 x {len(bi)} pool in {time.time() - tp:.0f}s, RSS {rss_gb():.1f} GB")
    # records without a usable state: name-only search over the whole country's S1 (reverse direction)
    bi = np.where(sb == "")[0] if partitioned else np.array([], dtype=int)
    if len(bi):
        ai = np.arange(len(sa))
        XA, XB = vecs_for(V, ta, ai, cfg.w_name), vecs_for(V, tb, bi, cfg.w_name)
        found = []
        for m, tag in [("name", "sl_name"), ("phon", "sl_phon"), ("cat", "sl_cat"), ("addr", "sl_addr"), ("comb", "sl_comb")]:
            c, r, _ = _topk(XB[m], XA[m], cfg.k_stateless, cfg.threads, cfg.chunk)
            found.append(pl.DataFrame({"li": r.astype(np.int32), "lj": c.astype(np.int32), "via": np.full(len(r), VIA_BITS[tag], np.int32)}))
        u = pl.concat(found).group_by(["li", "lj"]).agg(pl.col("via").unique().sum())
        li, lj = u["li"].to_numpy(), u["lj"].to_numpy()
        parts.append(pairs_df(ai[li], bi[lj], u["via"], cos_all(XA, XB, li, lj)))
        del XA, XB
        log(f"  stateless path: {len(bi)} records, RSS {rss_gb():.1f} GB")
    # S1 records without a state (rare, ~0.1-0.7%): forward search over the WHOLE country pool, in pool chunks,
    # merging top-k across chunks (review #2: otherwise their true matches are silently lost)
    ai = np.where(sa == "")[0] if partitioned else np.array([], dtype=int)
    if len(ai):
        XA = vecs_for(V, ta, ai, cfg.w_name)
        hits = []
        for start in range(0, len(sb), 1_000_000):
            bi = np.arange(start, min(start + 1_000_000, len(sb)))
            XB = vecs_for(V, tb, bi, cfg.w_name)
            for m, k in [("name", cfg.k_name), ("phon", cfg.k_phon), ("cat", cfg.k_cat), ("comb", cfg.k_comb)]:
                r, c, v = _topk(XA[m], XB[m], k, cfg.threads, cfg.chunk)
                hits.append(pl.DataFrame({"li": r.astype(np.int32), "j": bi[c].astype(np.int32), "v": v,
                                          "via": np.full(len(r), VIA_BITS[f"s1sl_{m}"], np.int32)}))
            del XB
        h = pl.concat(hits)
        h = h.filter(pl.col("v").rank("ordinal", descending=True).over(["li", "via"]) <= cfg.k_name)
        u = h.group_by(["li", "j"]).agg(pl.col("via").unique().sum())
        li = u["li"].to_numpy(); gj = u["j"].to_numpy()
        uj, inv = np.unique(gj, return_inverse=True)
        XB = vecs_for(V, tb, uj, cfg.w_name)
        parts.append(pairs_df(ai[li], gj, u["via"], cos_all(XA, XB, li, inv)))
        del XA, XB, h, u
        log(f"  stateless S1 path: {len(ai)} S1 searched over {len(sb)} records")
    # acronym key within partition, best 3 S1 per pool record by address cosine
    acr = pl.DataFrame({"i": np.arange(len(sa), dtype=np.int32), "acr": [acronym_key(x) for x in s1["name_core"].to_list()], "st": sa})
    pc = pl.DataFrame({"j": np.arange(len(sb), dtype=np.int32),
                       "acr": [x if 2 <= len(x) <= 5 and " " not in x else "" for x in pool["name_core"].to_list()], "st": sb})
    aj = acr.filter(pl.col("acr") != "").join(pc.filter(pl.col("acr") != ""), on=["acr", "st"])
    if aj.height:
        ai, bi = aj["i"].to_numpy(), aj["j"].to_numpy()
        XA, XB = vecs_for(V, ta, ai, cfg.w_name), vecs_for(V, tb, bi, cfg.w_name)
        k = np.arange(len(ai))
        cs = cos_all(XA, XB, k, k)
        a = pairs_df(ai, bi, np.full(len(ai), VIA_BITS["acr"], np.int32), cs).filter(pl.col("cos_addr").rank("ordinal", descending=True).over("j") <= 3)
        parts.append(a)
    ex = (pl.concat([exact_key_pairs(s1, pool, sa, sb), numtok_pairs(s1, pool, sa, sb)]
                    + ([stateless_exact_pairs(s1, pool, sb)] if STATELESS_EXACT else []))
            .group_by(["i", "j"]).agg(pl.col("via").unique().sum())) if partitioned else None
    if ex is not None and ex.height:
        ii, jj = ex["i"].to_numpy(), ex["j"].to_numpy()
        ui, inv_i = np.unique(ii, return_inverse=True); uj, inv_j = np.unique(jj, return_inverse=True)
        XA, XB = vecs_for(V, ta, ui, cfg.w_name), vecs_for(V, tb, uj, cfg.w_name)
        parts.append(pairs_df(ii, jj, ex["via"].to_numpy(), cos_all(XA, XB, inv_i, inv_j)))
        del XA, XB
        log(f"  exact-key pairs: {ex.height}")
    out = pl.concat(parts)
    out = out.group_by(["i", "j"]).agg(pl.col("via").unique().sum(),   # each part has distinct bits per pair
                                       *[pl.col(c).max() for c in ["cos_name", "cos_phon", "cos_addr", "cos_comb", "cos_cat"]])
    log(f"  union: {out.height} pairs ({out.height / len(sa):.1f} per S1) in {time.time() - T:.0f}s, RSS {rss_gb():.1f} GB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--countries", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    announce_pid(f"p1_block[{a.split}]")
    scan = lambda f: pl.scan_parquet(WORK / "norm" / f"{a.split}_{f}.parquet").select(KEEP_S1)
    countries = (a.countries.split(",") if a.countries else
                 scan("source1").select("country").unique().collect()["country"].sort().to_list())
    for c in countries:
        d = WORK / "p1" / a.split / c
        if done(d / "cands.parquet", a.force):
            log(f"skip {c} (exists)"); continue
        resource_guard(f"block {a.split} {c}")
        # only this country is ever loaded (filter is pushed into the parquet scan)
        s1 = scan("source1").filter(pl.col("country") == c).collect()
        pool = pl.concat([scan("source2").filter(pl.col("country") == c).with_columns(pl.lit("S2").alias("src")),
                          scan("source3").filter(pl.col("country") == c).with_columns(pl.lit("S3").alias("src"))]).collect()
        s1, pool, n1, n2 = infer_missing_states(s1, pool)
        write_parquet_atomic(s1, d / "s1.parquet")
        write_parquet_atomic(pool, d / "pool.parquet")
        log(f"{a.split}/{c}: {s1.height} S1, {pool.height} pool; state inferred from address words for {n1} S1, {n2} pool")
        cands = block_country(s1, pool, BlockCfgV2(verbose=False, k_stateless=K_STATELESS))
        write_parquet_atomic(cands, d / "cands.parquet")
        del s1, pool, cands
        gc.collect()


if __name__ == "__main__":
    main()
