"""Label-free threshold scan per country with the team's calibrated histogram model (postprocess/checks/hist_fit.py model/fit and
hist_score.py score, verbatim logic). For US / India (stack_e5hyb2 stage-3 test p) and France (stack_ce p, minus the swap
removal list), apply the team's decision at global threshold t (group thresholds and decoy rules unchanged), count kept
records per S1, fit (recall r, false pairs lam, hopeless z), score. Also MH+DL alone for India."""
import os
import numpy as np, polars as pl
from scipy.optimize import minimize
from scipy.stats import binom, poisson
W = os.environ.get("ER_WORK_DIR", "work"); E = f"{W}/e5hyb2"
gt = (pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
        .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n")))
T = np.bincount(gt["n"].to_numpy()); T = T / T.sum()
def model(r, lam, z, K=9):
    kept = np.zeros(len(T) + 1)
    for n, pn in enumerate(T): kept[:n + 1] += pn * (1 - z) * binom.pmf(np.arange(n + 1), n, r)
    kept[0] += z
    o = np.convolve(kept, poisson.pmf(np.arange(len(kept)), lam))[:60]
    return np.r_[o[:K - 1], max(1 - o[:K - 1].sum(), 1e-12)]
def fit(cnt):
    nll = lambda x: -np.dot(cnt, np.log(np.clip(model(*x), 1e-12, None)))
    return minimize(nll, [0.97, 0.02, 0.005], bounds=[(0.8, 1), (0, 0.5), (0, 0.1)], method="L-BFGS-B").x
def score(r, lam, z):
    e = 0.0; fpp = poisson.pmf(np.arange(40), lam)
    for n, pn in enumerate(T):
        if pn == 0: continue
        if n == 0: e += pn * fpp[0]; continue
        for tp in range(1, n + 1):
            ptp = (1 - z) * binom.pmf(tp, n, r)
            for fp, pf in enumerate(fpp):
                if pf < 1e-12: break
                P, Rr = tp / (tp + fp), tp / n
                e += pn * ptp * pf * 1.25 * P * Rr / (0.25 * P + Rr)
    return e
BASE = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98}
def kept(d, t):
    key = pl.when(pl.col("sup") == "corr").then(pl.col("grp") + "|corr").otherwise(pl.col("grp")); e = pl.lit(t)
    for g_, x in BASE.items():
        if "|" in g_: e = pl.when(key == g_).then(pl.lit(max(x, t))).otherwise(e)
    for g_, x in BASE.items():
        if "|" not in g_: e = pl.when((pl.col("grp") == g_) & ~key.is_in([k for k in BASE if "|" in k])).then(pl.lit(max(x, t))).otherwise(e)
    keep = (pl.col("r") == 1) & (pl.col("p") >= e) & ~pl.col("dtok").fill_null(False)
    restore = (pl.col("country") == "India") & pl.col("grp").is_in(["shift_pure", "shift_ms_only"]) & (pl.col("p") >= 0.98) & (pl.col("r") == 1) & ~pl.col("dtok").fill_null(False)
    return d.filter(keep | restore).select("s1_id", "m_id")
flags = pl.read_parquet(f"{E}/rule_flags_pairs.parquet", columns=["s1_id", "m_id", "dtok", "split"]).filter(pl.col("split") == "test").drop("split")
s1 = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=["entity_id", "country", "addr_state"])
rem = pl.read_parquet(f"{W}/france_swap/fr_swap_remove.parquet").select("s1_id", "m_id")
for c, f in (("US", "stage3_test_pairs_US"), ("India", "stage3_test_pairs_India"), ("France", "stage3_test_pairs_France_stack_ce_USED")):
    d = pl.read_parquet(f"{E}/{f}.parquet").join(flags, on=["s1_id", "m_id"], how="left")
    if "country" not in d.columns: d = d.with_columns(pl.lit(c).alias("country"))
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r"))
    groups = {c: s1.filter(pl.col("country") == c)["entity_id"]}
    if c == "India": groups["India MH+DL"] = s1.filter((pl.col("country") == c) & pl.col("addr_state").is_in(["mh", "dl"]))["entity_id"]
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        k = kept(d, t)
        if c == "France": k = k.join(rem, on=["s1_id", "m_id"], how="anti")
        cnt_s1 = k.group_by("s1_id").len()
        for gname, ids in groups.items():
            kk = pl.DataFrame({"s1_id": ids}).join(cnt_s1, on="s1_id", how="left")["len"].fill_null(0).clip(0, 8).to_numpy()
            cnt = np.bincount(kk, minlength=9).astype(float)
            r_, lam, z = fit(cnt)
            print(f"{gname:12s} t={t:.2f}  kept {int(sum(np.arange(9) * cnt)):8d}  r {r_:.4f}  lam {lam:.4f}  z {z:.4f}  score {score(r_, lam, z):.5f}", flush=True)
print("TSCAN_DONE")
