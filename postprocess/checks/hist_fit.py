"""Label-free recall / false-pair fit of the kept-count histogram. Truth n ~ train-truth histogram (US=India). Model: with prob
z an S1 loses all its records (hopeless: address-less ties, parser bugs), else each true record is kept with prob r; plus
Poisson(lam) false pairs per S1. Multinomial ML fit per country and file; cells 0..7, 8+.
Usage: python postprocess/checks/hist_fit.py file1 file2 ..."""
import sys
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import binom, poisson
R = "."  # run from the repo root
gt = (pl.read_parquet(f"{R}/work/parquet/train_ground_truth.parquet")
        .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n")))
T = np.bincount(gt["n"].to_numpy()); T = T / T.sum()
def model(r, lam, z, K=9):
    kept = np.zeros(len(T) + 1)
    for n, pn in enumerate(T):
        kept[:n + 1] += pn * (1 - z) * binom.pmf(np.arange(n + 1), n, r)
    kept[0] += z
    fp = poisson.pmf(np.arange(len(kept)), lam)
    o = np.convolve(kept, fp)[:60]
    return np.r_[o[:K - 1], max(1 - o[:K - 1].sum(), 1e-12)]
def fit(cnt):
    nll = lambda x: -np.dot(cnt, np.log(np.clip(model(*x), 1e-12, None)))
    best = minimize(nll, [0.97, 0.02, 0.005], bounds=[(0.8, 1), (0, 0.5), (0, 0.1)], method="L-BFGS-B")
    return best.x, best.fun
ids = {c: pl.read_parquet(f"{R}/work_v4/p1/test/{c}/s1.parquet", columns=["entity_id"])["entity_id"] for c in ("France", "US", "India")}
print(f"truth hist: {' '.join(f'{x*100:.2f}' for x in T[:9])}  mean {np.dot(np.arange(len(T)), T):.3f}")
print(f"{'file':38s} {'country':7s}  recall r   false/S1 lam   hopeless z   fit P(k=0..8) vs obs (x100)")
for f in sys.argv[1:]:
    sub = (pl.read_csv(f"{R}/output_bucket/{f}/matching_results.tsv", separator="\t", infer_schema=False, quote_char=None).fill_null("")
             .with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))
             .rename({"source1_entity_id": "s1_id"}))
    for c in ("US", "India", "France"):
        k = pl.DataFrame({"s1_id": ids[c]}).join(sub.select("s1_id", "k"), on="s1_id", how="left")["k"].fill_null(0).clip(0, 8).to_numpy()
        cnt = np.bincount(k, minlength=9).astype(float)
        (r, lam, z), _ = fit(cnt)
        m = model(r, lam, z); o = cnt / cnt.sum()
        print(f"{f[-38:]:38s} {c:7s}  {r:.4f}     {lam:.4f}        {z:.4f}      " + " ".join(f"{(m[i]-o[i])*100:+.2f}" for i in range(9)))
