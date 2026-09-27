"""Pair features for candidate (S1, S2/S3) pairs. No country feature (country is an open set; France unseen)."""
import math
import os
import re
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

HNUM = os.environ.get("ER_HNUM", "1") == "1"      # house-number relation features (v7)
LU = os.environ.get("ER_LU", "1") == "1"          # legal-aware name uniqueness (v8)
S1_COLS = ["entity_id", "country", "name_core", "name_legal", "name_phon", "addr_text", "addr_state", "addr_nums"]
M_COLS = ["entity_id", "src", "name_core", "name_legal", "name_alts", "name_phon", "name_is_domain", "name_has_alt",
          "name_non_latin", "addr_text", "addr_state", "addr_nums", "addr_empty", "addr_native_state"]


def token_idf(texts):
    df = Counter()
    for t in texts:
        df.update(set(t.split()))
    n = len(texts)
    return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}


def _idf_overlap(a_list, b_list, idf, default):
    wj, mx, ns = np.zeros(len(a_list), np.float32), np.zeros(len(a_list), np.float32), np.zeros(len(a_list), np.int16)
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        A, B = set(a.split()), set(b.split())
        inter, union = A & B, A | B
        if union:
            wi = sum(idf.get(w, default) for w in inter)
            wu = sum(idf.get(w, default) for w in union)
            wj[k] = wi / wu
            mx[k] = max((idf.get(w, default) for w in inter), default=0.0)
            ns[k] = len(inter)
    return wj, mx, ns


def _set_feats(a_list, b_list):
    """number-set features: jaccard, any shared, conflict (both non-empty & disjoint), missing."""
    jac, share, conf, miss = (np.zeros(len(a_list), np.float32) for _ in range(4))
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        A, B = set(a.split()), set(b.split())
        if not A or not B:
            miss[k] = 1
            continue
        i = len(A & B)
        jac[k] = i / len(A | B)
        share[k] = i > 0
        conf[k] = i == 0
    return jac, share, conf, miss


def _num_feats(a_list, b_list):
    """longest number equal (house numbers are usually the longest digit run) and min digit edit distance."""
    leq = np.zeros(len(a_list), np.int8); med = np.full(len(a_list), -1, np.int8)
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        A, B = a.split(), b.split()
        if not A or not B:
            continue
        la = max(A, key=lambda x: (len(x), x)); lb = max(B, key=lambda x: (len(x), x))
        leq[k] = la == lb
        med[k] = min(Levenshtein.distance(x, y) for x in A[:6] for y in B[:6])
    return leq, med


_HREL_ORDER = ["drop", "transpose", "sub", "near10", "near100", "far"]     # most typo-like explanation wins
# codes sorted by VALID match rate: typos (one digit changed at a non-last position, a digit dropped) are the same place,
# numeric neighbours (last digit changed, |diff| <= 100) are the next building
HREL_CODE = {"equal": 0, "missing": 1, "subset": 2, "sub_mid": 3, "far": 4, "drop": 5, "sub_last": 6, "near100": 7,
             "transpose": 8, "near10": 9}


def _is_drop(a, b):
    L, S = (a, b) if len(a) > len(b) else (b, a)
    return len(L) - len(S) == 1 and any(L[:k] + L[k + 1:] == S for k in range(len(L)))


def _hnum_rel(a_list, b_list):
    """house-number relation of the closest differing number pair: code, log1p(|diff|), changed-digit position from the
    end (one-digit substitutions), digit count."""
    n = len(a_list)
    rel = np.full(n, HREL_CODE["missing"], np.int8); dlog = np.full(n, -1.0, np.float32)
    pos = np.full(n, -1, np.int8); nlen = np.zeros(n, np.int8)
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        A = {x for x in a.split() if x.isdigit() and len(x) <= 7}; B = {x for x in b.split() if x.isdigit() and len(x) <= 7}
        if not A or not B:
            continue
        if A == B:
            rel[k], dlog[k] = HREL_CODE["equal"], 0.0
            continue
        ra, rb = A - B, B - A
        if not ra or not rb:
            rel[k] = HREL_CODE["subset"]
            continue
        best = None
        for x in ra:
            for y in rb:
                dn = abs(int(x) - int(y)); p = -1
                if _is_drop(x, y):
                    c = "drop"
                elif len(x) == len(y) and sorted(x) == sorted(y) and sum(u != v for u, v in zip(x, y)) == 2:
                    c = "transpose"
                elif len(x) == len(y) and sum(u != v for u, v in zip(x, y)) == 1:
                    c = "sub"; p = next(len(x) - 1 - i for i in range(len(x)) if x[i] != y[i])
                elif dn <= 10:
                    c = "near10"
                elif dn <= 100:
                    c = "near100"
                else:
                    c = "far"
                key = (_HREL_ORDER.index(c), dn, x, y)      # full key: ties never depend on set order
                if best is None or key < best[0]:
                    best = (key, c, dn, p, max(len(x), len(y)))
        _, c, dn, p, ln = best
        rel[k] = HREL_CODE[("sub_last" if p == 0 else "sub_mid") if c == "sub" else c]
        dlog[k], pos[k], nlen[k] = np.log1p(dn), p, ln
    return {"f_hrel": rel, "f_hdiff_log": dlog, "f_hsub_pos": pos, "f_hnum_len": nlen}


HN_SHIFTS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}    # record number = S1 number + k: look-alike distractors (VALID 0.4-16% true)


def _hnum_dir(a_list, b_list):
    """signed house-number evidence: shift flag (record = S1 + a distractor offset), signed
    difference of the closest differing pair (record - S1, clipped to +-100, -999 undefined), typo flag (one digit
    dropped, or one digit changed that is not the last) with its position from the end and digit count, and the number
    of differing numbers."""
    n = len(a_list)
    sh = np.full(n, -1, np.int8); sd = np.full(n, -999, np.int16); ty = np.full(n, -1, np.int8)
    ps = np.full(n, -1, np.int8); ln = np.full(n, -1, np.int8); nd = np.zeros(n, np.int8)
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        A = {x for x in a.split() if x.isdigit() and len(x) <= 7}; B = {x for x in b.split() if x.isdigit() and len(x) <= 7}
        if not A or not B:
            continue
        ra, rb = A - B, B - A
        nd[k] = min(max(len(ra), len(rb)), 127)
        if not ra or not rb:
            sh[k], ty[k] = 0, 0
            continue
        sh[k] = any((int(y) - int(x)) in HN_SHIFTS for x in ra for y in rb)
        t, p, L = 0, -1, -1
        for x in ra:
            for y in rb:
                if _is_drop(x, y):
                    Lg, Sm = (x, y) if len(x) > len(y) else (y, x)
                    j = [j for j in range(len(Lg)) if Lg[:j] + Lg[j + 1:] == Sm][-1]
                    t, p, L = 1, max(p, len(Lg) - 1 - j), max(L, len(Lg))
                elif len(x) == len(y):
                    dif = [i for i, (u, v) in enumerate(zip(x, y)) if u != v]
                    if len(dif) == 1 and dif[0] < len(x) - 1:
                        t, p, L = 1, max(p, len(x) - 1 - dif[0]), max(L, len(x))
        ty[k], ps[k], ln[k] = t, p, L
        sd[k] = max(-100, min(100, min((abs(int(y) - int(x)), int(y) - int(x)) for x in ra for y in rb)[1]))
    return {"f_hn_shift": sh, "f_hn_sdiff": sd, "f_hn_typo": ty, "f_hn_pos": ps, "f_hn_len": ln, "f_hn_nd": nd}


_NUM_RE = re.compile(r"\d+")


def _hnum_ms(a_texts, b_texts):
    """shift flag on number MULTISETS from the address text: a +k look-alike can hide
    behind a repeated number ('58 14 14' -> '58 14 19'), which the set-based f_hn_shift cannot see. -1 = no numbers."""
    from collections import Counter as _C
    out = np.full(len(a_texts), -1, np.int8)
    for k, (a, b) in enumerate(zip(a_texts, b_texts)):
        A = _C(int(x) for x in _NUM_RE.findall(a or "") if len(x) <= 7); B = _C(int(x) for x in _NUM_RE.findall(b or "") if len(x) <= 7)
        if not A or not B:
            continue
        ma, mb = A - B, B - A
        out[k] = any((y - x) in HN_SHIFTS for x in ma for y in mb)
    return {"f_hn_shift_ms": out}


def _alt_best(s1_core, alts):
    out = np.full(len(s1_core), -1, np.float32)
    for k, (a, al) in enumerate(zip(s1_core, alts)):
        if al:
            out[k] = max(fuzz.token_set_ratio(a, x) for x in al.split("|"))
    return out


def name_freq_tables(s1: pl.DataFrame, pool: pl.DataFrame) -> dict:
    """How common is a name? (frequency encoding; unsupervised, so it is also computed on test).

    A rare name with a typo'd house number is likely the same business; a common name with a different
    number is likely a look-alike (error analysis of Phase 0 v2). State-level counts are exact in the regional
    samples; country-level counts matter for records without a usable address.
    """
    states = [x for x in s1["addr_state"].unique().to_list() if x]
    return {
        "s1_state": s1.group_by(["country", "addr_state", "name_core"]).len().rename({"len": "n"}),
        "s1_cty": s1.group_by(["country", "name_core"]).len().rename({"len": "n"}),
        "pool_state": pool.group_by(["country", "addr_state", "name_core"]).len().rename({"len": "n"}),
        "s1_phon_cty": s1.group_by(["country", "name_phon"]).len().rename({"len": "n"}),
        # look-alike density: how many businesses share this exact address (word order ignored). 11.8% of French
        # S1 share an address; a shared address with a different name is the classic look-alike distractor.
        "s1_addr_cty": s1.filter(pl.col("addr_text") != "").with_columns(_addr_key("addr_text", states).alias("addr_key"))
                         .group_by(["country", "addr_key"]).len().rename({"len": "n"}),
        "pool_addr_cty": pool.filter(pl.col("addr_text") != "").with_columns(_addr_key("addr_text", states).alias("addr_key"))
                             .group_by(["country", "addr_key"]).len().rename({"len": "n"}),
        "states": states,
        **(lu_tables(s1) if LU else {}),
    }


def _addr_key(col: str, states: list) -> pl.Expr:
    """Order-free address key without state codes and leading zeros ("lille 238 rue x hdf" == "238 rue x lille",
    "05131" == "5131"): noisy records shuffle parts, drop the region, or keep zero padding."""
    return (pl.col(col).str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(states)).str.strip_chars_start("0"))
              .list.eval(pl.element().filter(pl.element() != "")).list.sort().list.join(" "))


_LEGAL_FAMILY = {"sasu": "sas", "eurl": "sarl", "pvt": "ltd", "public": "", "co": ""}


def _legal_fam(col: str) -> pl.Expr:
    return (pl.col(col).str.split(" ").list.eval(pl.element().replace(_LEGAL_FAMILY))
              .list.eval(pl.element().filter(pl.element() != "")).list.unique())


def _legal_key(col: str) -> pl.Expr:
    return _legal_fam(col).list.sort().list.join(" ")


def lu_tables(s1: pl.DataFrame) -> dict:
    """S1 counts per (name, legal form) and per (name, legal family), country-wide (recall analysis 25 Sep 18:05)."""
    s = s1.select("country", "name_core", pl.col("name_legal").fill_null("")).with_columns(_legal_key("name_legal").alias("lf"))
    return {"s1_nl": s.group_by(["country", "name_core", "name_legal"]).len().rename({"len": "n"}),
            "s1_nlf": s.group_by(["country", "name_core", "lf"]).len().rename({"len": "n"})}


def add_lu(d: pl.DataFrame, t: dict) -> pl.DataFrame:
    """Two S1 may share a name but not a legal form: how many S1 carry the record's exact (name, legal) key, does this
    S1 carry it, and how the two legal forms relate (0 both empty, 1 equal, 2 same family, 3 one empty, 4 subset,
    5 overlap, 6 disjoint)."""
    d = d.with_columns(pl.col("a_name_legal", "b_name_legal").fill_null("").name.prefix("_"))
    d = d.with_columns(_legal_key("_a_name_legal").alias("_alf"), _legal_key("_b_name_legal").alias("_blf"))
    d = (d.join(t["s1_nl"].rename({"n": "f_lu_k_nl"}), left_on=["a_country", "b_name_core", "_b_name_legal"],
                right_on=["country", "name_core", "name_legal"], how="left")
          .join(t["s1_nlf"].rename({"n": "f_lu_k_nlf"}), left_on=["a_country", "b_name_core", "_blf"],
                right_on=["country", "name_core", "lf"], how="left"))
    ta, tb = pl.col("_a_name_legal").str.split(" "), pl.col("_b_name_legal").str.split(" ")
    d = d.with_columns(ta.list.set_intersection(tb).list.len().alias("_i"), ta.list.len().alias("_na"), tb.list.len().alias("_nb"))
    same = pl.col("a_name_core") == pl.col("b_name_core")
    return d.with_columns(
        pl.col("f_lu_k_nl", "f_lu_k_nlf").fill_null(0).clip(0, 10).cast(pl.Int16),
        (same & (pl.col("_a_name_legal") == pl.col("_b_name_legal"))).cast(pl.Int8).alias("f_lu_nl_eq"),
        (same & (pl.col("_alf") == pl.col("_blf"))).cast(pl.Int8).alias("f_lu_nlf_eq"),
        pl.when((pl.col("_a_name_legal") == "") & (pl.col("_b_name_legal") == "")).then(0)
          .when(pl.col("_a_name_legal") == pl.col("_b_name_legal")).then(1)
          .when(pl.col("_alf") == pl.col("_blf")).then(2)
          .when((pl.col("_a_name_legal") == "") | (pl.col("_b_name_legal") == "")).then(3)
          .when((pl.col("_i") == pl.col("_na")) | (pl.col("_i") == pl.col("_nb"))).then(4)
          .when(pl.col("_i") > 0).then(5).otherwise(6).cast(pl.Int8).alias("f_lu_legal_rel"),
    ).drop("_a_name_legal", "_b_name_legal", "_alf", "_blf", "_i", "_na", "_nb")


def add_name_freq(d: pl.DataFrame, t: dict) -> pl.DataFrame:
    j = lambda df, tab, left, name: df.join(tab.rename({"n": name}), left_on=left, right_on=tab.columns[:-1], how="left")
    d = j(d, t["s1_state"], ["a_country", "a_addr_state", "a_name_core"], "f_freq_s1name_state")
    d = j(d, t["s1_cty"], ["a_country", "a_name_core"], "f_freq_s1name_cty")
    d = j(d, t["s1_cty"], ["a_country", "b_name_core"], "f_freq_bname_in_s1_cty")
    d = j(d, t["pool_state"], ["a_country", "a_addr_state", "a_name_core"], "f_freq_s1name_in_pool_state")
    d = j(d, t["s1_phon_cty"], ["a_country", "b_name_phon"], "f_freq_bphon_in_s1_cty")
    cols = ["f_freq_s1name_state", "f_freq_s1name_cty", "f_freq_bname_in_s1_cty", "f_freq_s1name_in_pool_state",
            "f_freq_bphon_in_s1_cty"]
    if "s1_addr_cty" in t:
        st = t["states"]
        d = d.with_columns(_addr_key("a_addr_text", st).alias("_ak"), _addr_key("b_addr_text", st).alias("_bk"))
        d = j(d, t["s1_addr_cty"], ["a_country", "_ak"], "f_freq_s1addr_cty")          # S1 sharing the S1 address
        d = j(d, t["s1_addr_cty"], ["a_country", "_bk"], "f_freq_baddr_in_s1_cty")      # S1 sharing the record address
        d = j(d, t["pool_addr_cty"], ["a_country", "_bk"], "_n_pool_addr")              # records sharing it
        d = d.with_columns(pl.col("f_freq_s1addr_cty", "f_freq_baddr_in_s1_cty").fill_null(0).clip(0, 10),  # test tails
                           # reach 104 (France), train ~10: absolute counts capped; pool side as records per business
                           (pl.col("_n_pool_addr").fill_null(0) / pl.col("f_freq_baddr_in_s1_cty").fill_null(0).clip(1, None))
                           .clip(0, 20).cast(pl.Float32).alias("f_freq_baddr_pool_per_s1")).drop("_ak", "_bk", "_n_pool_addr")
        cols += ["f_freq_s1addr_cty", "f_freq_baddr_in_s1_cty"]
    return d.with_columns([pl.col(c).fill_null(0).cast(pl.Int32) for c in cols])


def build_features(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, idf_by_country=None,
                   freq_tables=None) -> pl.DataFrame:
    """pairs: [s1_id, m_id, cos_*, via]. Returns pairs + feature columns (prefix f_)."""
    a = s1.select(S1_COLS).rename({c: f"a_{c}" for c in S1_COLS if c != "entity_id"}).rename({"entity_id": "s1_id"})
    b = pool.select(M_COLS).rename({c: f"b_{c}" for c in M_COLS if c != "entity_id"}).rename({"entity_id": "m_id"})
    d = pairs.join(a, on="s1_id", how="left").join(b, on="m_id", how="left")
    an, bn = d["a_name_core"].to_list(), d["b_name_core"].to_list()
    ap, bp = d["a_name_phon"].to_list(), d["b_name_phon"].to_list()
    aa, ba = d["a_addr_text"].to_list(), d["b_addr_text"].to_list()
    W = -1
    f = {
        "f_name_ratio": cpdist(an, bn, scorer=fuzz.ratio, workers=W),
        "f_name_partial": cpdist(an, bn, scorer=fuzz.partial_ratio, workers=W),
        "f_name_tsort": cpdist(an, bn, scorer=fuzz.token_sort_ratio, workers=W),
        "f_name_tset": cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=W),
        "f_name_jw": cpdist(an, bn, scorer=JaroWinkler.normalized_similarity, workers=W),
        "f_name_lev": cpdist(an, bn, scorer=Levenshtein.normalized_similarity, workers=W),
        "f_phon_ratio": cpdist(ap, bp, scorer=fuzz.ratio, workers=W),
        "f_phon_tset": cpdist(ap, bp, scorer=fuzz.token_set_ratio, workers=W),
        "f_addr_ratio": cpdist(aa, ba, scorer=fuzz.ratio, workers=W),
        "f_addr_tset": cpdist(aa, ba, scorer=fuzz.token_set_ratio, workers=W),
        "f_addr_partial": cpdist(aa, ba, scorer=fuzz.partial_ratio, workers=W),
        "f_addr_tsort": cpdist(aa, ba, scorer=fuzz.token_sort_ratio, workers=W),
    }
    # IDF-weighted token overlap (rare shared words are strong evidence), per country
    countries = d["a_country"].to_list()
    wj = np.zeros(d.height, np.float32); mx = np.zeros(d.height, np.float32); ns = np.zeros(d.height, np.int16)
    awj = np.zeros(d.height, np.float32)
    for c in set(countries):
        idx = np.where(np.array(countries) == c)[0]
        idf_n, idf_a = idf_by_country[c]
        dn, da = max(idf_n.values()), max(idf_a.values())
        r = _idf_overlap([an[i] for i in idx], [bn[i] for i in idx], idf_n, dn)
        wj[idx], mx[idx], ns[idx] = r
        awj[idx] = _idf_overlap([aa[i] for i in idx], [ba[i] for i in idx], idf_a, da)[0]
    f["f_name_idf_jacc"], f["f_name_idf_max_shared"], f["f_name_n_shared"] = wj, mx, ns
    f["f_addr_idf_jacc"] = awj
    nj, nsh, ncf, nms = _set_feats(d["a_addr_nums"].to_list(), d["b_addr_nums"].to_list())
    f["f_num_jacc"], f["f_num_share"], f["f_num_conflict"], f["f_num_missing"] = nj, nsh, ncf, nms
    f["f_alt_best"] = _alt_best(an, d["b_name_alts"].to_list())
    f["f_num_longest_eq"], f["f_num_min_edit"] = _num_feats(d["a_addr_nums"].to_list(), d["b_addr_nums"].to_list())
    if HNUM:
        f.update(_hnum_rel(d["a_addr_nums"].to_list(), d["b_addr_nums"].to_list()))
        f.update(_hnum_dir(d["a_addr_nums"].to_list(), d["b_addr_nums"].to_list()))
        f.update(_hnum_ms(aa, ba))
    from normalize import acronym_key, concat_key
    ac = [concat_key(x) for x in an]; bc = [concat_key(x) for x in bn]
    f["f_cat_ratio"] = cpdist(ac, bc, scorer=fuzz.ratio, workers=W)
    f["f_acr_match"] = np.array([bool(x) and bool(y) and ((acronym_key(x) == y) or (acronym_key(y) == x and len(x) <= 5))
                                 for x, y in zip(an, bn)], np.int8)
    out = d.with_columns([pl.Series(k, v) for k, v in f.items()])
    # Telangana and Andhra Pradesh are one state for matching purposes (review #1/#2)
    out = out.with_columns(pl.col("a_addr_state").replace("ts", "ap"), pl.col("b_addr_state").replace("ts", "ap"))
    out = out.with_columns(
        (pl.col("a_addr_state") == pl.col("b_addr_state")).and_(pl.col("a_addr_state") != "").cast(pl.Int8).alias("f_state_eq"),
        ((pl.col("a_addr_state") != pl.col("b_addr_state")) & (pl.col("a_addr_state") != "") & (pl.col("b_addr_state") != "")).cast(pl.Int8).alias("f_state_conflict"),
        ((pl.col("a_name_legal") == pl.col("b_name_legal")) & (pl.col("a_name_legal") != "")).cast(pl.Int8).alias("f_legal_eq"),
        ((pl.col("a_name_legal") == "") | (pl.col("b_name_legal") == "")).cast(pl.Int8).alias("f_legal_missing"),
        (pl.col("a_name_core").str.split(" ").list.first() == pl.col("b_name_core").str.split(" ").list.first()).cast(pl.Int8).alias("f_first_tok_eq"),
        pl.col("a_name_core").str.len_chars().alias("f_len_a"),
        pl.col("b_name_core").str.len_chars().alias("f_len_b"),
        pl.col("b_addr_text").str.len_chars().alias("f_addr_len_b"),
        (pl.col("b_src") == "S3").cast(pl.Int8).alias("f_is_s3"),
        pl.col("b_name_is_domain").cast(pl.Int8).alias("f_is_domain"),
        pl.col("b_name_has_alt").cast(pl.Int8).alias("f_has_alt"),
        pl.col("b_name_non_latin").cast(pl.Int8).alias("f_non_latin"),
        pl.col("b_addr_empty").cast(pl.Int8).alias("f_addr_empty"),
        pl.col("b_addr_native_state").cast(pl.Int8).alias("f_native_state"),
        *[_via_flag(d, t).alias(f"f_via_{t}")
          for t in ["name", "phon", "addr", "comb", "rev", "cat", "acr", "sl_name", "sl_phon", "sl_cat", "sl_addr", "sl_comb",
                    "x_name", "x_phon", "x_addr", "x_numtok", "x_sl"]],
        *[pl.col(c).alias(f"f_{c}") for c in d.columns if c.startswith("cos_")],
        # address present but no state/region found (empty addresses are f_addr_empty); France-safe semantics
        ((pl.col("b_addr_state") == "") & ~pl.col("b_addr_empty")).cast(pl.Int8).alias("f_b_stateless"),
    )
    # legal forms compared as FAMILY SETS: "sas" == "sasu", "sarl" == "eurl", "pvt ltd" == "ltd"; the generic "co"
    # (Company/Cie/Ets/Societe) is ignored (the old whole-string conflict flagged "co sarl" vs "sarl"; removed).
    out = (out.with_columns(_legal_fam("a_name_legal").alias("_la"), _legal_fam("b_name_legal").alias("_lb"))
              .with_columns(pl.col("_la").list.set_intersection("_lb").list.len().alias("_li"),
                            pl.col("_la").list.set_union("_lb").list.len().alias("_lu"))
              .with_columns(((pl.col("_la").list.len() > 0) & (pl.col("_lb").list.len() > 0) & (pl.col("_li") == 0))
                            .cast(pl.Int8).alias("f_legal_fam_conflict"),
                            pl.when(pl.col("_lu") > 0).then(pl.col("_li") / pl.col("_lu")).otherwise(-1.0)   # -1: both empty
                            .cast(pl.Float32).alias("f_legal_fam_jacc"))
              .drop("_la", "_lb", "_li", "_lu"))
    out = add_context(out)
    if freq_tables is not None:
        out = add_name_freq(out, freq_tables)
        if "s1_nl" in freq_tables:
            out = add_lu(out, freq_tables)
    fcols = [c for c in out.columns if c.startswith("f_")]
    return out.select(["s1_id", "m_id", "a_country"] + fcols).with_columns(
        [pl.col(c).cast(pl.Float32) for c in fcols if out.schema[c] in (pl.Float64,)])


_VIA_ORDER = ["name", "phon", "addr", "comb", "cat", "rev", "acr", "sl_name", "sl_phon", "sl_cat", "sl_addr", "sl_comb",
              "s1sl_name", "s1sl_phon", "s1sl_cat", "s1sl_comb", "x_name", "x_phon", "x_addr", "x_numtok", "x_sl"]


def _via_flag(d: pl.DataFrame, tag: str) -> pl.Expr:
    """Search-path flag from either the Phase 0 text form ("name+rev") or the Phase 1 bit-mask (int)."""
    if d.schema["via"] == pl.Utf8:
        if tag not in ("name", "phon", "addr", "comb", "rev", "cat", "acr", "sl_name", "sl_phon", "sl_cat"):
            return pl.lit(0, pl.Int8)
        return pl.col("via").str.contains(rf"(^|\+){tag}($|\+)").cast(pl.Int8)
    return ((pl.col("via") & (1 << _VIA_ORDER.index(tag))) != 0).cast(pl.Int8)


def add_m_context(pairs: pl.DataFrame) -> pl.DataFrame:
    """Record-level context; needs ALL pairs of a record, so it runs once on the thin pairs table."""
    return pairs.with_columns(
        pl.col("cos_comb").rank("dense", descending=True).over("m_id").cast(pl.Int32).alias("f_rank_comb_m"),
        (pl.col("cos_comb").max().over("m_id") - pl.col("cos_comb")).cast(pl.Float32).alias("f_gap_comb_m"),
        pl.len().over("m_id").cast(pl.Int32).alias("f_ncand_m"),
    )


def add_context(d: pl.DataFrame) -> pl.DataFrame:
    """Where does this pair stand among the S1's candidates (all pairs of an S1 are in the same chunk)."""
    return d.with_columns(
        pl.col("cos_comb").rank("dense", descending=True).over("s1_id").cast(pl.Int32).alias("f_rank_comb_s1"),
        pl.col("cos_name").rank("dense", descending=True).over("s1_id").cast(pl.Int32).alias("f_rank_name_s1"),
        pl.col("cos_addr").rank("dense", descending=True).over("s1_id").cast(pl.Int32).alias("f_rank_addr_s1"),
        (pl.col("cos_comb").max().over("s1_id") - pl.col("cos_comb")).alias("f_gap_comb_s1"),
        pl.len().over("s1_id").cast(pl.Int32).alias("f_ncand_s1"),
        (pl.col("cos_comb") > 0.6).sum().over("s1_id").cast(pl.Int32).alias("f_n_strong_s1"),
    ).with_columns(
        # tied look-alikes: candidates of this S1 with (nearly) the same name
        (pl.col("f_name_tset") >= 90).alias("_same"),
    ).with_columns(
        pl.col("_same").sum().over("s1_id").cast(pl.Int32).alias("f_n_same_name_s1"),
        pl.when(pl.col("_same")).then(pl.col("cos_addr")).otherwise(None).rank("dense", descending=True).over("s1_id")
          .fill_null(99).cast(pl.Int32).alias("f_addr_rank_same_name"),
        (pl.when(pl.col("_same")).then(pl.col("cos_addr")).otherwise(None).max().over("s1_id") - pl.col("cos_addr"))
          .fill_null(1.0).alias("f_addr_gap_same_name"),
        (pl.col("_same") & (pl.col("f_num_share") > 0)).sum().over("s1_id").cast(pl.Int32).alias("f_n_same_name_num_s1"),
    ).drop("_same")
