"""Append late feature groups to already built feature parts, in place (saves a full feature rebuild).

Groups: hn = house-number features (features._hnum_rel, _hnum_dir; inputs: addr_nums of both sides),
        lu = legal-aware name uniqueness (features.add_lu; inputs: names, legal forms, the split's S1 table),
        hms = shift flag on number multisets (features._hnum_ms; inputs: address texts),
        tr = S1 name vs the record name translated word by word (translit.py; native-script India records only,
             plain record name otherwise): f_tr_ratio, f_tr_tset, f_tr_eq. Train tables are cross-fitted.
Idempotent: a group is skipped when its columns exist, unless --force. ER_DECOY_POOL=1 (train): the decoy records of
exp_decoy.py (decoys.parquet) join like pool records, for the feats_fitD / feats_validD folders.
Usage: python add_hnum.py train feats_fit_ent40,feats_valid,feats_fit [--groups hn,lu] [--workers 6] [--force]
       python add_hnum.py test feats_all --groups lu
"""
import argparse
import os
from multiprocessing import Pool

import polars as pl

from features import _hnum_dir, _hnum_ms, _hnum_rel, add_lu, lu_tables
from utils import WORK, log, write_parquet_atomic

GROUPS = {"hn": ["f_hrel", "f_hdiff_log", "f_hsub_pos", "f_hnum_len", "f_hn_shift", "f_hn_sdiff", "f_hn_typo", "f_hn_pos",
                 "f_hn_len", "f_hn_nd"],
          "lu": ["f_lu_k_nl", "f_lu_k_nlf", "f_lu_nl_eq", "f_lu_nlf_eq", "f_lu_legal_rel"],
          "hms": ["f_hn_shift_ms"],
          "tr": ["f_tr_ratio", "f_tr_tset", "f_tr_eq"]}
_MAPS = {}
_TR = {}


def _maps(split, c):
    if (split, c) not in _MAPS:
        d = WORK / "p1" / split / c
        s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id", "country", "addr_nums", "addr_text", "name_core", "name_legal", "addr_state"])
        pool = pl.read_parquet(d / "pool.parquet", columns=["entity_id", "addr_nums", "addr_text", "name_core", "name_legal", "name_non_latin"])
        if split == "train" and os.environ.get("ER_DECOY_POOL") == "1" and (d / "decoys.parquet").exists():   # exp_decoy folders
            from exp_decoy import normalize_rows
            pool = pl.concat([pool, normalize_rows(pl.read_parquet(d / "decoys.parquet"), pool.schema).select(pool.columns)])
        _MAPS[(split, c)] = (s1.rename({"entity_id": "s1_id", "country": "a_country", "addr_nums": "a", "addr_text": "a_addr_text",
                                        "name_core": "a_name_core", "name_legal": "a_name_legal", "addr_state": "a_state"}),
                             pool.rename({"entity_id": "m_id", "addr_nums": "b", "addr_text": "b_addr_text", "name_core": "b_name_core",
                                          "name_legal": "b_name_legal", "name_non_latin": "b_nl"}),
                             lu_tables(s1))
    return _MAPS[(split, c)]


def _tr_cols(split, c, ab):
    """f_tr_*: S1 name_core vs the translated record name (India native-script records), plain name otherwise."""
    import numpy as np
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    import translit
    from p1_block import STATE_MERGE
    b = ab.select("_r", "m_id", pl.col("b_name_core").fill_null(""), pl.col("b_nl").fill_null(False))
    if c == "India":
        if split not in _TR:
            vs = [] if split == "test" else translit.valid_states_india()
            _TR[split] = (translit.tables(split, vs), vs)
        tabs, vs = _TR[split]
        key = translit.table_key(split, pl.col("s1_id"), pl.col("a_state").fill_null("").replace(STATE_MERGE), vs)
        b = b.with_columns(ab.with_columns(key.alias("k"))["k"])
        u = b.filter(pl.col("b_nl")).select("m_id", "k", "b_name_core").unique(["m_id", "k"])
        u = u.with_columns(pl.Series("bt", [translit.translate(n, tabs[k]) for n, k in zip(u["b_name_core"].to_list(), u["k"].to_list())],
                                     dtype=pl.Utf8)).select("m_id", "k", "bt")
        b = b.join(u, on=["m_id", "k"], how="left", maintain_order="left").with_columns(pl.coalesce("bt", "b_name_core").alias("bt"))
    else:
        b = b.with_columns(pl.col("b_name_core").alias("bt"))
    a_n, b_t = ab["a_name_core"].fill_null("").to_list(), b["bt"].to_list()
    return [pl.Series("f_tr_ratio", (cpdist(a_n, b_t, scorer=fuzz.ratio, workers=1) / 100).astype(np.float32)),
            pl.Series("f_tr_tset", (cpdist(a_n, b_t, scorer=fuzz.token_set_ratio, workers=1) / 100).astype(np.float32)),
            pl.Series("f_tr_eq", np.array([x == y and x != "" for x, y in zip(a_n, b_t)], np.int8))]


def one(job):
    split, c, p, groups, force = job
    df = pl.read_parquet(p)
    todo = [g for g in groups if force or GROUPS[g][0] not in df.columns]
    if not todo:
        return 0
    df = df.drop([x for g in todo for x in GROUPS[g] if x in df.columns])
    s1, pool, lut = _maps(split, c)
    ab = (df.select("s1_id", "m_id").with_row_index("_r").join(s1, on="s1_id", how="left", maintain_order="left")
            .join(pool, on="m_id", how="left", maintain_order="left").with_columns(pl.col("a", "b").fill_null("")))
    assert ab.height == df.height and ab["m_id"].equals(df["m_id"]) and ab["a_country"].null_count() == 0
    new = []
    if "hn" in todo:
        for fn in (_hnum_rel, _hnum_dir):
            new += [pl.Series(k, v) for k, v in fn(ab["a"].to_list(), ab["b"].to_list()).items()]
    if "hms" in todo:
        new += [pl.Series(k, v) for k, v in _hnum_ms(ab["a_addr_text"].to_list(), ab["b_addr_text"].to_list()).items()]
    if "tr" in todo:
        new += _tr_cols(split, c, ab)
    if "lu" in todo:
        lu = add_lu(ab, lut).sort("_r")
        assert lu.height == df.height and lu["m_id"].equals(df["m_id"])
        new += [lu[x] for x in GROUPS["lu"]]
    write_parquet_atomic(df.with_columns(new), p)
    return df.height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("dirs")
    ap.add_argument("--groups", default="hn")
    ap.add_argument("--countries", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="recompute the groups even when present")
    a = ap.parse_args()
    groups = a.groups.split(",")
    base = WORK / "p1" / a.split
    cs = a.countries.split(",") if a.countries else sorted(p.name for p in base.iterdir() if p.is_dir())
    jobs = [(a.split, c, p, groups, a.force) for c in cs for dn in a.dirs.split(",") for p in sorted((base / c / dn).glob("part_*.parquet"))]
    log(f"add groups {groups}: {len(jobs)} parts in {a.split} {cs} {a.dirs}")
    n = 0
    with Pool(a.workers, maxtasksperchild=20) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs)):
            n += r
            if k % 10 == 0:
                log(f"  {k + 1}/{len(jobs)} parts, {n} pairs")
    log(f"add groups done: {n} pairs updated")


if __name__ == "__main__":
    main()
