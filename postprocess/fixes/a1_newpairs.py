"""A1 bucket build step: turn the containment-key pairs into cands.parquet rows (i, j, via, cos_name, cos_phon, cos_addr,
cos_comb, cos_cat) with the SAME vectorizers as p1_block (fit_vectorizers is deterministic, seed 0, same corpora) and
via = the 'addr' bit (f_via_addr = 1, like address-path pairs). Keeps key pairs with rk <= 3, score >= .5, not in the
stage-2 set and not already in cands.parquet (keys/contain_<split>_<c>.parquet from a1_test_volume.py).
Out: work/errfix/out/a1/newpairs_<split>_<c>.parquet. The lead appends them to a copy of cands.parquet in a new
ER_WORK_DIR root and reruns the v10 feature + stage 1/2 chain there.
Usage: POLARS_MAX_THREADS=3 python postprocess/fixes/a1_newpairs.py --split valid|test --countries US,India[,France]"""
import argparse
import sys
import time
import numpy as np
import polars as pl
sys.path.insert(0, "src")
from p1_block import VIA_BITS, cos_all, fit_vectorizers, texts, vecs_for  # noqa: E402
from blocking import BlockCfgV2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["valid", "test"])
ap.add_argument("--countries", default="US,India")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--smin", type=float, default=0.5)
a = ap.parse_args()
cfg = BlockCfgV2(verbose=False)
for c in a.countries.split(","):
    T = time.time()
    base = f"work_v4/p1/{'test' if a.split == 'test' else 'train'}/{c}"
    k = (pl.read_parquet(f"work/errfix/out/a1/keys/contain_{a.split}_{c}.parquet")
           .filter((pl.col("rk") <= a.k) & (pl.col("score") >= a.smin) & ~pl.col("in2") & ~pl.col("inc")).select("i", "j").unique())
    cols = ["name_core", "name_alts", "name_phon", "addr_text"]
    s1 = pl.read_parquet(f"{base}/s1.parquet", columns=cols)
    pool = pl.read_parquet(f"{base}/pool.parquet", columns=cols)
    ta, tb = texts(s1), texts(pool)
    V = fit_vectorizers(ta, tb, cfg.max_df)
    ii, jj = k["i"].to_numpy(), k["j"].to_numpy()
    ui, inv_i = np.unique(ii, return_inverse=True)
    uj, inv_j = np.unique(jj, return_inverse=True)
    XA, XB = vecs_for(V, ta, ui, cfg.w_name), vecs_for(V, tb, uj, cfg.w_name)
    cs = cos_all(XA, XB, inv_i, inv_j)
    out = pl.DataFrame({"i": ii.astype(np.int32), "j": jj.astype(np.int32), "via": np.full(len(ii), VIA_BITS["addr"], np.int32), **cs})
    out.write_parquet(f"work/errfix/out/a1/newpairs_{a.split}_{c}.parquet")
    print(c, a.split, "new pairs", out.height, "cos_addr median", float(np.median(cs["cos_addr"])), f"{time.time() - T:.0f}s", flush=True)
