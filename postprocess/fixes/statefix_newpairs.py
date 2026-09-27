"""State-parser rescue, candidate step. The address parser (normalize._state_of_component) takes a state from any comma
component that equals a state code once digits are removed, last match wins: 'Fl 3' (floor 3) becomes Florida,
'..., DC, ..., Washington' becomes WA, a split 'Nag Ar' becomes Arunachal. The S1 (or record) then sits in the wrong state
partition and within-state blocking never compares it with its twin. VALID holds one state per country, so it cannot show
this; TRAIN labels can (work/errfix/out/state_rules_<C>_{s1,rec}.parquet, selected combos in state_rules_sel.json).
S1 / records whose address carries exactly one other state code, in a selected (parsed, other) combo, get the other state
and are blocked against it with the p1_block vectorizers (top-k by name and by name+address, both directions).
The small scoring root also gets the existing cands of every touched record and of every touched unfixed S1, so the
rank / margin context features see the usual competitors (old pairs of fixed S1 are dropped: they were wrong-state).
Out (work/errfix/out/sf/): cands_<split>_<c>.parquet (small root cands: new + context pairs), newpairs_<split>_<c>.parquet
(new pairs only, i, j; the name a1_apply.py --newpairs-dir expects), fix_<split>_<c>.parquet (side, row, entity_id, old, new).
Usage: POLARS_MAX_THREADS=4 python postprocess/fixes/statefix_newpairs.py --split test --countries US,India"""
import argparse
import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from p1_block import VIA_BITS, cos_all, fit_vectorizers, texts, vecs_for  # noqa: E402
from blocking import STATE_MERGE, BlockCfgV2  # noqa: E402

US = "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy".split()
IN = "ap ar as br cg ga gj hr hp jh ka kl mp mh mn ml mz nl od pb rj sk tn ts tr up uk wb dl jk ch py an ld dn".split()
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["valid", "test"])
ap.add_argument("--countries", default="US,India")
ap.add_argument("--rules", default="work/errfix/out/state_rules_sel.json")
ap.add_argument("--k1", type=int, default=20, help="S1 side: top-k pool records per path")
ap.add_argument("--k2", type=int, default=5, help="record side: top-k S1 per path")
ap.add_argument("--paths", default="name,comb", help="search paths (round 2: name,comb,addr)")
ap.add_argument("--out", default="work/errfix/out/sf", help="output dir (round 2: work/errfix/out/sf2)")
a = ap.parse_args()
R = json.load(open(a.rules))
cfg = BlockCfgV2(verbose=False)
out_dir = a.out
import os  # noqa: E402
os.makedirs(out_dir, exist_ok=True)


def alt_state(df, codes):
    return (df.with_columns(pl.col("addr_text").fill_null("").str.extract_all(r"\b[a-z]{2}\b")
                            .list.eval(pl.element().filter(pl.element().is_in(codes))).list.unique().alias("toks"))
              .with_columns(pl.col("toks").list.set_difference(pl.concat_list(pl.col("addr_state").fill_null(""))).alias("alt"))
              .with_columns(pl.when(pl.col("alt").list.len() == 1).then(pl.col("alt").list.first()).alias("alt")).drop("toks"))


def topk(XA, XB, k, chunk=256):
    """row-wise top-k columns of XA @ XB.T (both l2-normalised sparse), score > 0."""
    ii, jj = [], []
    BT = XB.T.tocsc()
    for s in range(0, XA.shape[0], chunk):
        S = (XA[s:s + chunk] @ BT).toarray()
        kk = min(k, S.shape[1])
        if kk == 0:
            continue
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        for r in range(S.shape[0]):
            c = idx[r][S[r, idx[r]] > 0]
            ii.append(np.full(len(c), s + r)); jj.append(c)
    return (np.concatenate(ii), np.concatenate(jj)) if ii else (np.array([], int), np.array([], int))


for c in a.countries.split(","):
    T0 = time.time()
    codes = US if c == "US" else IN
    base = f"work_v4/p1/{'test' if a.split == 'test' else 'train'}/{c}"
    cols = ["entity_id", "name_core", "name_alts", "name_phon", "addr_text", "addr_state"]
    s1 = alt_state(pl.read_parquet(f"{base}/s1.parquet", columns=cols), codes)
    pool = alt_state(pl.read_parquet(f"{base}/pool.parquet", columns=cols), codes)
    rs1 = {tuple(x) for x in R[c]["s1"]}; rrec = {tuple(x) for x in R[c]["rec"]}
    f1 = [(p, q) in rs1 for p, q in zip(s1["addr_state"].fill_null("").to_list(), s1["alt"].to_list())]
    fp = [(p, q) in rrec for p, q in zip(pool["addr_state"].fill_null("").to_list(), pool["alt"].to_list())]
    f1, fp = np.array(f1), np.array(fp)
    mg = lambda s: np.array([STATE_MERGE.get(x or "", x or "") for x in s])  # noqa: E731
    st1 = mg(np.where(f1, s1["alt"].fill_null("").to_numpy(), s1["addr_state"].fill_null("").to_numpy()))
    stp = mg(np.where(fp, pool["alt"].fill_null("").to_numpy(), pool["addr_state"].fill_null("").to_numpy()))
    fix = pl.concat([
        df.with_row_index("row").filter(pl.Series(mask)).select(pl.lit(side).alias("side"), pl.col("row").cast(pl.Int64), "entity_id",
                                                                 pl.col("addr_state").alias("old"), pl.col("alt").alias("new"))
        for side, df, mask in (("s1", s1, f1), ("rec", pool, fp))])
    fix.write_parquet(f"{out_dir}/fix_{a.split}_{c}.parquet")
    print(f"{c}: fixed S1 {int(f1.sum())}, fixed records {int(fp.sum())}", flush=True)
    ta, tb = texts(s1), texts(pool)
    V = fit_vectorizers(ta, tb, cfg.max_df)
    I, J, VIA = [], [], []
    for X in sorted(set(st1[f1]) | set(stp[fp])):
        ia_fix, jb_all = np.flatnonzero(f1 & (st1 == X)), np.flatnonzero(stp == X)
        jb_fix, ia_all = np.flatnonzero(fp & (stp == X)), np.flatnonzero(st1 == X)
        for rows_a, rows_b, k, side in ((ia_fix, jb_all, a.k1, "s1"), (jb_fix, ia_all, a.k2, "rec")):
            if len(rows_a) == 0 or len(rows_b) == 0:
                continue
            ta_, tb_ = (ta, tb) if side == "s1" else (tb, ta)
            XA, XB = vecs_for(V, ta_, rows_a, cfg.w_name), vecs_for(V, tb_, rows_b, cfg.w_name)
            for m in a.paths.split(","):
                r, q = topk(XA[m], XB[m], k)
                ai, bj = rows_a[r], rows_b[q]
                if side == "s1":
                    I.append(ai); J.append(bj)
                else:
                    I.append(bj); J.append(ai)
                VIA.append(np.full(len(r), VIA_BITS[m] | (VIA_BITS["rev"] if side == "rec" else 0)))
    new = (pl.DataFrame({"i": np.concatenate(I).astype(np.int32), "j": np.concatenate(J).astype(np.int32),
                         "via": np.concatenate(VIA).astype(np.int32)})
             .group_by("i", "j").agg(pl.col("via").bitwise_or()))
    old = pl.read_parquet(f"{base}/cands.parquet")
    new = new.join(old.select("i", "j"), on=["i", "j"], how="anti")
    ii, jj = new["i"].to_numpy(), new["j"].to_numpy()
    ui, inv_i = np.unique(ii, return_inverse=True)
    uj, inv_j = np.unique(jj, return_inverse=True)
    cs = cos_all(vecs_for(V, ta, ui, cfg.w_name), vecs_for(V, tb, uj, cfg.w_name), inv_i, inv_j)
    new = new.with_columns(**{k: pl.Series(v) for k, v in cs.items()})
    new.select("i", "j").write_parquet(f"{out_dir}/newpairs_{a.split}_{c}.parquet")   # a1_apply / a1_gate name
    # context: existing pairs of touched records and touched unfixed S1 (never the old pairs of a fixed S1)
    fixed_i = pl.Series("i", np.flatnonzero(f1).astype(np.int32))
    ctx = old.filter((pl.col("j").is_in(new["j"].unique().implode()) | pl.col("i").is_in(new["i"].unique().implode()))
                     & ~pl.col("i").is_in(fixed_i.implode()))
    small = pl.concat([ctx, new.select(old.columns).cast(old.schema)])
    small.write_parquet(f"{out_dir}/cands_{a.split}_{c}.parquet")
    print(f"{c}: new pairs {new.height} (S1 {new['i'].n_unique()}, records {new['j'].n_unique()}), context pairs {ctx.height}, "
          f"cos_comb median {float(np.median(cs['cos_comb'])):.3f}, {time.time() - T0:.0f}s", flush=True)
