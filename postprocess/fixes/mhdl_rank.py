"""Big-state blocking misses (Maharashtra / Delhi lose ~4.5% of same-state true pairs at blocking vs 1-2% elsewhere, TRAIN).
For the missed same-state true pairs (record has an address), rank of the true S1 among ALL S1 of the state for the record,
per search path (name / phon / addr / comb, p1_block vectorizers). Share recovered at k = 3, 5, 10, 20, 50 per path and for
the union; plus the cos of the true pair. Runs on worker 2 (TF-IDF fit).
Usage: POLARS_MAX_THREADS=4 python postprocess/fixes/mhdl_rank.py [--states mh,dl]"""
import argparse
import sys
import time
import numpy as np
import polars as pl
sys.path.insert(0, "src")
from p1_block import fit_vectorizers, texts, vecs_for  # noqa: E402
from blocking import BlockCfgV2  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--states", default="mh,dl"); ap.add_argument("--chunk", type=int, default=512)
a = ap.parse_args()
cfg = BlockCfgV2(verbose=False)
T0 = time.time()
base = "work_v4/p1/train/India"
cols = ["entity_id", "name_core", "name_alts", "name_phon", "addr_text", "addr_state"]
s1 = pl.read_parquet(f"{base}/s1.parquet", columns=cols); pool = pl.read_parquet(f"{base}/pool.parquet", columns=cols)
ta, tb = texts(s1), texts(pool)
V = fit_vectorizers(ta, tb, cfg.max_df); print(f"fit {time.time() - T0:.0f}s", flush=True)
g = (pl.read_parquet("work/parquet/train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"}).filter(pl.col("m_id").str.len_chars() > 0))
S1 = s1.select("entity_id", "addr_state").with_row_index("i").with_columns(pl.col("i").cast(pl.Int32))
P = pool.select("entity_id", "addr_state", "addr_text").with_row_index("j").with_columns(pl.col("j").cast(pl.Int32))
cands = pl.read_parquet(f"{base}/cands.parquet", columns=["i", "j"])
t = (g.join(S1.rename({"entity_id": "s1_id", "addr_state": "st1"}), on="s1_id").join(P.rename({"entity_id": "m_id", "addr_state": "st2"}), on="m_id")
      .join(cands.with_columns(pl.lit(1).alias("inc")), on=["i", "j"], how="left")
      .filter(pl.col("inc").is_null() & (pl.col("st1") == pl.col("st2")) & (pl.col("addr_text").fill_null("").str.strip_chars() != "")))
KS = [3, 5, 10, 20, 50, 100]
for st in a.states.split(","):
    tt = t.filter(pl.col("st1") == st)
    srows = S1.filter(pl.col("addr_state") == st)["i"].to_numpy()
    pos = np.full(s1.height, -1, np.int64); pos[srows] = np.arange(len(srows))
    XS = vecs_for(V, ta, srows, cfg.w_name)
    rj, ti = tt["j"].to_numpy(), pos[tt["i"].to_numpy()]
    XR = vecs_for(V, tb, rj, cfg.w_name)
    ranks = {}; coss = {}
    for m in ("name", "phon", "addr", "comb"):
        BT = XS[m].T.tocsc(); rk = np.empty(len(rj), np.int64); cs = np.empty(len(rj), np.float32)
        for s in range(0, len(rj), a.chunk):
            D = (XR[m][s:s + a.chunk] @ BT).toarray()
            tc = D[np.arange(D.shape[0]), ti[s:s + a.chunk]]
            rk[s:s + a.chunk] = (D > tc[:, None]).sum(1); cs[s:s + a.chunk] = tc
        ranks[m] = rk; coss[m] = cs
        print(f"{st} {m:5s} missed {len(rj):,}: " + " ".join(f"k{k} {np.mean(rk < k):.3f}" for k in KS) + f" | cos median {np.median(cs):.3f}, {time.time() - T0:.0f}s", flush=True)
    mn = np.minimum.reduce([ranks[m] for m in ranks])
    print(f"{st} UNION missed {len(rj):,}: " + " ".join(f"k{k} {np.mean(mn < k):.3f}" for k in KS), flush=True)
    for combo in (("comb", "phon"), ("comb", "addr"), ("phon", "addr")):
        mm = np.minimum(ranks[combo[0]], ranks[combo[1]])
        print(f"{st} {'+'.join(combo):10s}: " + " ".join(f"k{k} {np.mean(mm < k):.3f}" for k in KS), flush=True)
    pl.DataFrame({"i": tt["i"], "j": tt["j"], **{f"rk_{m}": ranks[m] for m in ranks}, **{f"cos_{m}": coss[m] for m in coss}}).write_parquet(f"work/errfix/out/mhdl_rank_{st}.parquet")
