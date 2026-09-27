"""Big-state recall rescue, candidate step. In Maharashtra and Delhi blocking misses ~4.5% of same-state true pairs (TRAIN;
other states 1-2.5%): noisy transliterated names ('yunaited aiti pra li' <- 'united it') of very common names, so the
S1-side top-k fills up with look-alikes. For the missed train pairs the true S1 is in the record-side top-3 of at least one
path (comb / phon / addr / name) 69-70% of the time, top-10 76-77% (postprocess/fixes/mhdl_rank.py).
Record side = records that are free: test = unassigned in the shipped file (--records parquet of entity_id), train = no
true pair in the old cands (sampled --frac). Per state, record -> top-k S1 of the same state per path (sparse_dot_topn),
pairs not in the old cands. Small scoring root = new pairs + context: all old pairs of the touched records and the top
--ctx-s1 old pairs (by cos_comb) of each touched S1, so rank / margin features see the usual competitors.
Out (--out): cands_<split>_India.parquet, newpairs_<split>_India.parquet (i, j; the a1_apply / a1_gate name).
Usage: POLARS_MAX_THREADS=4 python postprocess/fixes/mhdl_newpairs.py --split test --records work/errfix/out/mhdl_unassigned.parquet"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, "src")
from p1_block import VIA_BITS, cos_all, fit_vectorizers, texts, vecs_for  # noqa: E402
from blocking import BlockCfgV2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["valid", "test"])
ap.add_argument("--country", default="India")
ap.add_argument("--states", default="mh,dl", help="comma list, or all")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--paths", default="comb,phon,addr,name")
ap.add_argument("--records", default="", help="test: parquet with entity_id of the free (unassigned) records")
ap.add_argument("--frac", type=float, default=0.3, help="train: share of free records searched")
ap.add_argument("--ctx-s1", type=int, default=10)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default="work/errfix/out/mhdl")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = BlockCfgV2(verbose=False)
T0 = time.time()
CTRY = a.country
base = f"work_v4/p1/{'test' if a.split == 'test' else 'train'}/{CTRY}"
cols = ["entity_id", "name_core", "name_alts", "name_phon", "addr_text", "addr_state"]
s1 = pl.read_parquet(f"{base}/s1.parquet", columns=cols)
pool = pl.read_parquet(f"{base}/pool.parquet", columns=cols)
old = pl.read_parquet(f"{base}/cands.parquet")
ta, tb = texts(s1), texts(pool)
V = fit_vectorizers(ta, tb, cfg.max_df)
print(f"fit {time.time() - T0:.0f}s", flush=True)
states = sorted(set(s1["addr_state"].drop_nulls().to_list()) - {""}) if a.states == "all" else a.states.split(",")
P = pool.select("entity_id", "addr_state", "addr_text").with_row_index("j").with_columns(pl.col("j").cast(pl.Int32))
P = P.filter(pl.col("addr_state").is_in(states) & (pl.col("addr_text").fill_null("").str.strip_chars() != ""))
if a.split == "test":
    P = P.join(pl.read_parquet(a.records, columns=["entity_id"]), on="entity_id", how="semi")
else:
    g = (pl.read_parquet("work/parquet/train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
           .explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"})
           .filter(pl.col("m_id").str.len_chars() > 0))
    S1i = s1.select(pl.col("entity_id").alias("s1_id")).with_row_index("i").with_columns(pl.col("i").cast(pl.Int32))
    tj = (g.join(S1i, on="s1_id").join(P.select(pl.col("entity_id").alias("m_id"), "j"), on="m_id")
           .join(old.select("i", "j"), on=["i", "j"], how="semi")["j"].unique())
    P = P.filter(~pl.col("j").is_in(tj.implode())).sample(fraction=a.frac, seed=0)
print(f"free records searched: {P.height:,} ({P.group_by('addr_state').len().sort('addr_state').rows()})", flush=True)
st_s1 = s1["addr_state"].to_numpy()
I, J, VIA = [], [], []
for st in states:
    srows = np.flatnonzero(st_s1 == st)
    rrows = P.filter(pl.col("addr_state") == st)["j"].to_numpy()
    if len(srows) == 0 or len(rrows) == 0:
        continue
    XS, XR = vecs_for(V, ta, srows, cfg.w_name), vecs_for(V, tb, rrows, cfg.w_name)
    for m in a.paths.split(","):
        C = sp_matmul_topn(XR[m].tocsr(), XS[m].T.tocsr(), top_n=a.k, threshold=1e-6, n_threads=a.threads).tocoo()
        I.append(srows[C.col]); J.append(rrows[C.row]); VIA.append(np.full(len(C.row), VIA_BITS[m] | VIA_BITS["rev"]))
        print(f"{st} {m}: {len(C.row):,} pairs, {time.time() - T0:.0f}s", flush=True)
new = (pl.DataFrame({"i": np.concatenate(I).astype(np.int32), "j": np.concatenate(J).astype(np.int32),
                     "via": np.concatenate(VIA).astype(np.int32)})
         .group_by("i", "j").agg(pl.col("via").bitwise_or()))
new = new.join(old.select("i", "j"), on=["i", "j"], how="anti")
ii, jj = new["i"].to_numpy(), new["j"].to_numpy()
ui, inv_i = np.unique(ii, return_inverse=True)
uj, inv_j = np.unique(jj, return_inverse=True)
cs = cos_all(vecs_for(V, ta, ui, cfg.w_name), vecs_for(V, tb, uj, cfg.w_name), inv_i, inv_j)
new = new.with_columns(**{k: pl.Series(v) for k, v in cs.items()})
new.select("i", "j").write_parquet(f"{a.out}/newpairs_{a.split}_{CTRY}.parquet")
ctx_r = old.filter(pl.col("j").is_in(new["j"].unique().implode()))
ctx_s = (old.filter(pl.col("i").is_in(new["i"].unique().implode())).sort("cos_comb", descending=True)
            .group_by("i", maintain_order=True).head(a.ctx_s1))
ctx = pl.concat([ctx_r, ctx_s.select(old.columns)]).unique(["i", "j"])
small = pl.concat([ctx, new.select(old.columns).cast(old.schema)])
small.write_parquet(f"{a.out}/cands_{a.split}_{CTRY}.parquet")
print(f"new pairs {new.height:,} (S1 {new['i'].n_unique():,}, records {new['j'].n_unique():,}), context {ctx.height:,}, "
      f"root {small.height:,}, cos_comb median {float(np.median(cs['cos_comb'])):.3f}, {time.time() - T0:.0f}s", flush=True)
