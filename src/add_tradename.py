"""v11: trade-name address search path (recall analyst, 26 Sep 11:30; docs/advisor/RECALL_ANALYSIS.md, r24).

Trade-name records = pool records with an address whose name_core shares no token with any S1 name of the same merged
state. For each such record, the S1 of the same state that share a house number (addr_nums token; numbers carried by more
than --max-per-num S1 of the state are skipped) are ranked by rapidfuzz token_set_ratio on addr_text; the top --k become
candidates. Pairs already in cands / cands_fz / cands_tr are dropped; the rest get the TF-IDF cosines of p1_block, their
own search bit (all f_via_* flags 0) and features written as part_7NN.parquet per role, same schema as the existing parts.
Run BEFORE `add_hnum.py ... --groups tr` when the tr columns are used.
Usage: python add_tradename.py --split train|test [--countries US,India] [--k 5] [--states ny] [--only-roles fit_ent80]
"""
import argparse

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from add_fuzzy import roles_for
from add_translit import TR_BIT
from p1_block import K_STATELESS, STATE_MERGE, BlockCfgV2, cos_all, fit_vectorizers, pairs_df, texts, vecs_for
from p1_features import country_context, iter_feature_chunks, load_country
from utils import WORK, announce_pid, log, write_parquet_atomic

TN_BIT = TR_BIT << 1          # x_tn: next free bit after the translit path


def trade_pairs(s1, pool, k, max_per_num, only_states=None):
    st = pl.col("addr_state").fill_null("").replace(STATE_MERGE)
    a = s1.select("i", st.alias("st"), pl.col("name_core").fill_null(""), pl.col("addr_text").fill_null(""), pl.col("addr_nums").fill_null(""))
    b = (pool.filter(~pl.col("addr_empty").fill_null(True))
             .select("j", st.alias("st"), pl.col("name_core").fill_null(""), pl.col("addr_text").fill_null(""), pl.col("addr_nums").fill_null("")))
    a, b = a.filter(pl.col("st") != ""), b.filter(pl.col("st") != "")
    if only_states:
        a, b = a.filter(pl.col("st").is_in(only_states)), b.filter(pl.col("st").is_in(only_states))
    vocab = a.select("st", pl.col("name_core").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "").unique()
    bt = b.select("j", "st", pl.col("name_core").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
    shared = bt.join(vocab, on=["st", "t"], how="semi").select("j").unique()
    tb = b.join(shared, on="j", how="anti")
    log(f"  trade-name records with an address: {tb.height} of {b.height}")
    num = lambda df, key: (df.select(key, "st", pl.col("addr_nums").str.split(" ").alias("n")).explode("n")
                             .filter(pl.col("n").str.contains(r"^\d{1,6}$")).unique())
    an, bn = num(a, "i"), num(tb, "j")
    common = an.group_by("st", "n").len().filter(pl.col("len") > max_per_num).select("st", "n")
    an = an.join(common, on=["st", "n"], how="anti")
    P = bn.join(an, on=["st", "n"]).select("i", "j").unique()
    log(f"  number-sharing pairs: {P.height}")
    if P.height == 0:
        return pl.DataFrame(schema={"i": pl.Int32, "j": pl.Int32})
    P = P.join(a.select("i", pl.col("addr_text").alias("aa")), on="i").join(tb.select("j", pl.col("addr_text").alias("ba")), on="j")
    P = P.with_columns(pl.Series("s", cpdist(P["aa"].to_list(), P["ba"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    P = P.with_columns(pl.col("s").rank("ordinal", descending=True).over("j").alias("r")).filter(pl.col("r") <= k)
    return P.select(pl.col("i").cast(pl.Int32), pl.col("j").cast(pl.Int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--countries", default="US,India")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-per-num", type=int, default=3000)
    ap.add_argument("--states", default="", help="smoke test: limit to these merged states (writes under smoke/)")
    ap.add_argument("--only-roles", default="", help="train: build parts only for these feature folders")
    a = ap.parse_args()
    announce_pid(f"add_tradename[{a.split}]")
    only_roles = set(a.only_roles.split(",")) if a.only_roles else None
    only_states = a.states.split(",") if a.states else None
    for c in a.countries.split(","):
        d = WORK / "p1" / a.split / c
        if (d / "cands_tn.parquet").exists() and not only_roles and not only_states:
            log(f"skip {c} (cands_tn exists)"); continue
        s1, pool, cands = load_country(a.split, c)
        if only_states:
            s1 = s1.filter(pl.col("addr_state").fill_null("").replace(STATE_MERGE).is_in(only_states))
            cands = cands.join(s1.select("i"), on="i", how="semi")
        P = trade_pairs(s1, pool, a.k, a.max_per_num, only_states)
        old = [cands.select("i", "j")] + [pl.read_parquet(d / f, columns=["i", "j"]) for f in ("cands_fz.parquet", "cands_tr.parquet") if (d / f).exists()]
        new = P.join(pl.concat(old), on=["i", "j"], how="anti")
        log(f"{a.split}/{c}: trade-name path {P.height} pairs, {new.height} new ({new.height / max(1, s1.height):.3f} per S1)")
        if new.height == 0:
            continue
        cfg = BlockCfgV2(verbose=False, k_stateless=K_STATELESS)
        ta, tb = texts(s1), texts(pool)
        V = fit_vectorizers(ta, tb, cfg.max_df)
        pos = np.full(int(s1["i"].max()) + 1, -1, np.int64)
        pos[s1["i"].to_numpy()] = np.arange(s1.height)
        ii, jj = new["i"].to_numpy(), new["j"].to_numpy()
        ui, inv_i = np.unique(ii, return_inverse=True)
        uj, inv_j = np.unique(jj, return_inverse=True)
        XA, XB = vecs_for(V, ta, pos[ui], cfg.w_name), vecs_for(V, tb, uj, cfg.w_name)
        new = pairs_df(ii, jj, np.full(len(ii), TN_BIT, np.int32), cos_all(XA, XB, inv_i, inv_j)).select(cands.columns)
        del XA, XB, V
        ctx, idf, freq = country_context(s1, pool, pl.concat([cands, new]))
        ctx = ctx.join(new.select("i", "j"), on=["i", "j"], how="semi")
        roles, lab = roles_for(a.split, c)
        for dname, mask in roles:
            if only_roles and dname not in only_roles:
                continue
            if mask is not None and s1.filter(mask).height == 0:
                continue
            ref = sorted((d / f"feats_{dname}").glob("part_*.parquet"))
            schema = pl.read_parquet_schema(ref[0]) if ref else None
            fdir = d / f"feats_{dname}" if not only_states else d / "smoke" / f"feats_{dname}"
            fdir.mkdir(parents=True, exist_ok=True)
            n = 0
            for b, nb, f in iter_feature_chunks(s1, pool, ctx, idf, freq, s1_mask=mask, labels=lab):
                if schema is not None:
                    missing = set(schema) - set(f.columns)
                    assert missing <= {"f_tr_ratio", "f_tr_tset", "f_tr_eq"} and set(f.columns) <= set(schema), f"schema differs: {set(f.columns) ^ set(schema)}"
                    f = f.select([pl.col(x).cast(t) for x, t in schema.items() if x in f.columns])
                write_parquet_atomic(f, fdir / f"part_{700 + b}.parquet")
                n += f.height
            log(f"  {a.split}/{c}/{dname}: {n} new pairs with features ({nb} parts)")
        if not only_roles and not only_states:
            write_parquet_atomic(new, d / "cands_tn.parquet")


if __name__ == "__main__":
    main()
