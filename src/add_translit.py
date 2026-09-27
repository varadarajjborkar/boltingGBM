"""v10: translated-name search path for native-script India records.

Each native-script pool record's name is translated word by word (translit.py; train uses the table learned without
the VALID states, test the table from all train states). Pairs = record x S1 of the same merged state whose name_core
equals the translated name, when at most --cap S1 of that state share the name. Pairs already in cands.parquet or
cands_fz.parquet are dropped; the rest get the TF-IDF cosines of p1_block, a search bit outside the f_via_* list
(all f_via_* flags 0, as for the fuzzy path) and features written as part_8NN.parquet per role, same schema as the
existing parts. Run BEFORE `add_hnum.py ... --groups tr` (the reference parts must not have the tr columns yet).
Usage: python add_translit.py --split train|test [--cap 30] [--states od] [--only-roles fit_ent80]
Train split needs ER_MODEL_DIR (roles.parquet), ER_FIT_MODE, ER_FIT_PCT and ER_S2_STATES as for p1_train.py.
"""
import argparse

import numpy as np
import polars as pl

import translit
from add_fuzzy import FZ_BIT, roles_for
from p1_block import K_STATELESS, STATE_MERGE, BlockCfgV2, cos_all, fit_vectorizers, pairs_df, texts, vecs_for
from p1_features import country_context, iter_feature_chunks, load_country
from utils import WORK, announce_pid, log, write_parquet_atomic

TR_BIT = FZ_BIT << 1          # x_tr: next free bit after the fuzzy path


def translit_pairs(s1, pool, table, cap):
    st = lambda: pl.col("addr_state").fill_null("").replace(STATE_MERGE)
    b = (pool.filter(pl.col("name_non_latin").fill_null(False)).select("j", "name_core", st().alias("st"))
             .filter(pl.col("st") != ""))
    b = b.with_columns(pl.Series("bt", [translit.translate(n, table) for n in b["name_core"].to_list()], dtype=pl.Utf8))
    a = s1.select("i", pl.col("name_core").fill_null(""), st().alias("st")).filter((pl.col("st") != "") & (pl.col("name_core") != ""))
    a = a.join(a.group_by("st", "name_core").len().filter(pl.col("len") <= cap).drop("len"), on=["st", "name_core"], how="semi")
    return (b.join(a.rename({"name_core": "bt"}), on=["st", "bt"]).select(pl.col("i").cast(pl.Int32), pl.col("j").cast(pl.Int32))
             .unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--states", default="", help="smoke test: limit to these merged states")
    ap.add_argument("--only-roles", default="", help="train: build parts only for these feature folders")
    a = ap.parse_args()
    announce_pid(f"add_translit[{a.split}]")
    c = "India"
    d = WORK / "p1" / a.split / c
    only_roles = set(a.only_roles.split(",")) if a.only_roles else None
    if (d / "cands_tr.parquet").exists() and not only_roles and not a.states:
        log("skip (cands_tr exists)"); return
    s1, pool, cands = load_country(a.split, c)
    if a.states:
        s1 = s1.filter(pl.col("addr_state").replace(STATE_MERGE).is_in(a.states.split(",")))
        cands = cands.join(s1.select("i"), on="i", how="semi")
    table = translit.tables(a.split, translit.valid_states_india() if a.split == "train" else [])["nv" if a.split == "train" else "all"]
    P = translit_pairs(s1, pool, table, a.cap)
    old = cands.select("i", "j")
    if (d / "cands_fz.parquet").exists():
        old = pl.concat([old, pl.read_parquet(d / "cands_fz.parquet", columns=["i", "j"])])
    new = P.join(old, on=["i", "j"], how="anti")
    log(f"{a.split}/{c}: translit path {P.height} pairs, {new.height} new ({new.height / max(1, s1.height):.3f} per S1)")
    if new.height == 0:
        return
    cfg = BlockCfgV2(verbose=False, k_stateless=K_STATELESS)
    ta, tb = texts(s1), texts(pool)
    V = fit_vectorizers(ta, tb, cfg.max_df)
    pos = np.full(int(s1["i"].max()) + 1, -1, np.int64)
    pos[s1["i"].to_numpy()] = np.arange(s1.height)
    ii, jj = new["i"].to_numpy(), new["j"].to_numpy()
    ui, inv_i = np.unique(ii, return_inverse=True)
    uj, inv_j = np.unique(jj, return_inverse=True)
    XA, XB = vecs_for(V, ta, pos[ui], cfg.w_name), vecs_for(V, tb, uj, cfg.w_name)
    new = pairs_df(ii, jj, np.full(len(ii), TR_BIT, np.int32), cos_all(XA, XB, inv_i, inv_j)).select(cands.columns)
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
        fdir = d / f"feats_{dname}" if not a.states else d / "smoke" / f"feats_{dname}"
        fdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for b, nb, f in iter_feature_chunks(s1, pool, ctx, idf, freq, s1_mask=mask, labels=lab):
            if schema is not None:
                assert set(f.columns) == set(schema), f"schema differs: {set(f.columns) ^ set(schema)}"
                f = f.select([pl.col(x).cast(t) for x, t in schema.items()])
            write_parquet_atomic(f, fdir / f"part_{800 + b}.parquet")
            n += f.height
        log(f"  {a.split}/{c}/{dname}: {n} new pairs with features ({nb} parts)")
    if not only_roles and not a.states:
        write_parquet_atomic(new, d / "cands_tr.parquet")


if __name__ == "__main__":
    main()
