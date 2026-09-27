"""v9: fuzzy name search path for records whose name came from a native script (recall analyst, 25 Sep 20:05).

Per merged state, native-script pool records (name_non_latin) are compared with every S1 of the same state:
top k by rapidfuzz ratio on name_phon (x1000) + name_core (tie-break), plus top k by name_core ratio. Pairs not in
cands.parquet become new candidates with the same TF-IDF cosines as p1_block and a search bit outside the f_via_*
list (all f_via_* flags 0 marks them). Their features are built with the record-side context of old + new pairs
and written as extra parts (part_9NN.parquet) next to the existing ones, per role, with the same schema.
Existing parts and cands.parquet are not changed; cands_fz.parquet keeps the new pairs.
Usage: python add_fuzzy.py --split train|test [--countries India] [--k 5] [--states ap] [--out-root DIR]
Train split needs ER_MODEL_DIR (roles.parquet), ER_FIT_MODE, ER_FIT_PCT and ER_S2_STATES as for p1_train.py.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from p1_block import K_STATELESS, STATE_MERGE, VIA_BITS, BlockCfgV2, cos_all, fit_vectorizers, pairs_df, texts, vecs_for
from p1_features import country_context, iter_feature_chunks, load_country
from utils import WORK, announce_pid, log, rss_gb, write_parquet_atomic

THREADS = int(os.environ.get("ER_THREADS", "7"))
FZ_BIT = 1 << len(VIA_BITS)          # x_fz: next free bit, not in features._VIA_ORDER


def merged(col):
    return np.array([STATE_MERGE.get(x, x) for x in col.fill_null("").to_list()])


def fuzzy_pairs(s1, pool, k, only_states=None):
    sa, sb = merged(s1["addr_state"]), merged(pool["addr_state"])
    nl = pool["name_non_latin"].fill_null(False).to_numpy()
    a_ph, a_co = s1["name_phon"].fill_null("").to_list(), s1["name_core"].fill_null("").to_list()
    b_ph, b_co = pool["name_phon"].fill_null("").to_list(), pool["name_core"].fill_null("").to_list()
    s1_i = s1["i"].to_numpy()                     # pipeline index of each S1 row (rows may be a subset)
    out = []
    for st in sorted(set(sb[nl]) - {""}):
        if only_states and st not in only_states:
            continue
        ai, bi = np.where(sa == st)[0], np.where((sb == st) & nl)[0]
        if len(ai) == 0:
            continue
        T = time.time()
        A_ph, A_co = [a_ph[x] for x in ai], [a_co[x] for x in ai]
        kk = min(k, len(ai))
        for s in range(0, len(bi), 500):
            r = bi[s:s + 500]
            Mc = cdist([b_co[x] for x in r], A_co, scorer=fuzz.ratio, workers=THREADS, dtype=np.float32)
            L = cdist([b_ph[x] for x in r], A_ph, scorer=fuzz.ratio, workers=THREADS, dtype=np.float32) * 1000 + Mc
            for M in (L, Mc):
                top = np.argpartition(-M, kk - 1, axis=1)[:, :kk] if kk < len(ai) else np.tile(np.arange(len(ai)), (len(r), 1))
                out.append(pl.DataFrame({"i": s1_i[ai[top].ravel()].astype(np.int32), "j": np.repeat(r, top.shape[1]).astype(np.int32)}))
            del Mc, L
        log(f"  fuzzy path '{st}': {len(bi)} native-script records x {len(ai)} S1 in {time.time() - T:.0f}s")
    return pl.concat(out).unique() if out else pl.DataFrame(schema={"i": pl.Int32, "j": pl.Int32})


def roles_for(split, c):
    if split == "test":
        return [("all", None)], None
    from p1_train import FIT_DIR, S2_STATES, labels, role_mask
    return ([(FIT_DIR, role_mask(c, "fit")), ("valid", role_mask(c, "valid"))]
            + ([("fit", role_mask(c, "s2fit"))] if S2_STATES else [])), labels()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--countries", default="India")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--states", default="", help="test runs only: limit to these merged states")
    ap.add_argument("--out-root", default="", help="test runs only: write parts under this folder instead of WORK")
    ap.add_argument("--only-roles", default="", help="train: build fuzzy parts only for these feature folders (e.g. fit_ent80); keeps cands_fz")
    a = ap.parse_args()
    announce_pid(f"add_fuzzy[{a.split}]")
    only = set(a.states.split(",")) if a.states else None
    for c in a.countries.split(","):
        d = WORK / "p1" / a.split / c
        o = Path(a.out_root) / a.split / c if a.out_root else d
        only_roles = set(a.only_roles.split(",")) if a.only_roles else None
        if (o / "cands_fz.parquet").exists() and not only_roles:
            log(f"skip {c} (cands_fz exists)"); continue
        s1, pool, cands = load_country(a.split, c)
        if only:          # smoke test: one state's S1, records and candidates
            s1 = s1.filter(pl.col("addr_state").replace(STATE_MERGE).is_in(list(only)))
            cands = cands.join(s1.select("i"), on="i", how="semi")
        P = fuzzy_pairs(s1, pool, a.k, only)
        new = P.join(cands.select("i", "j"), on=["i", "j"], how="anti")
        log(f"{a.split}/{c}: fuzzy path {P.height} pairs, {new.height} new ({new.height / max(1, s1.height):.2f} per S1)")
        cfg = BlockCfgV2(verbose=False, k_stateless=K_STATELESS)
        ta, tb = texts(s1), texts(pool)
        V = fit_vectorizers(ta, tb, cfg.max_df)
        pos = np.full(int(s1["i"].max()) + 1, -1, np.int64)
        pos[s1["i"].to_numpy()] = np.arange(s1.height)       # s1 may be a subset: map i -> row
        ii, jj = new["i"].to_numpy(), new["j"].to_numpy()
        ui, inv_i = np.unique(ii, return_inverse=True)
        uj, inv_j = np.unique(jj, return_inverse=True)
        XA, XB = vecs_for(V, ta, pos[ui], cfg.w_name), vecs_for(V, tb, uj, cfg.w_name)
        new = pairs_df(ii, jj, np.full(len(ii), FZ_BIT, np.int32), cos_all(XA, XB, inv_i, inv_j)).select(cands.columns)
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
            fdir = o / f"feats_{dname}"
            fdir.mkdir(parents=True, exist_ok=True)
            n = 0
            for b, nb, f in iter_feature_chunks(s1, pool, ctx, idf, freq, s1_mask=mask, labels=lab):
                if schema is not None:
                    assert set(f.columns) == set(schema), f"schema differs: {set(f.columns) ^ set(schema)}"
                    f = f.select([pl.col(x).cast(t) for x, t in schema.items()])
                write_parquet_atomic(f, fdir / f"part_{900 + b}.parquet")
                n += f.height
            log(f"  {a.split}/{c}/{dname}: {n} new pairs with features ({nb} parts), RSS {rss_gb():.1f} GB")
        if not only_roles:
            write_parquet_atomic(new, o / "cands_fz.parquet")


if __name__ == "__main__":
    main()
