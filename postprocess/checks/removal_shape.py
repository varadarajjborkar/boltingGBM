"""Label-free removal test on the WHOLE after-removal count histogram.
False pairs attached to random S1 leave those S1 with the country's general k histogram after removal (flat in r = pairs
removed per S1). Real copies leave them well below it (size-biased minus removed, falling with r). Fits the false share f of the
r=1 S1: after-hist = f * general + (1 - f) * real reference (US exact-name moved pairs, VALID precision 0.996, on file A).
Control: the LB-confirmed French swap removal reads f 0.84-0.90; the French moved-number class reads 0.00.
CAVEAT: on VALID India it reads f 0.41-0.76 for classes that are really 0.001-0.064 false (typo, swap,
reorder, no-address, p 0.90-0.98). Always read the same class on VALID next to the test figure; trust only a clear gap.
Usage: python postprocess/checks/removal_shape.py <output_bucket dir> <removal parquet (s1_id, m_id)> <country>"""
import sys, numpy as np, polars as pl
from scipy.optimize import minimize_scalar
R = "."  # run from the repo root; K = 7
base, lst, C = sys.argv[1:4]
sub = pl.read_csv(f"{R}/output_bucket/{base}/matching_results.tsv", separator="\t", infer_schema=False, quote_char=None).fill_null("")
kc = sub.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k")).select(pl.col("source1_entity_id").alias("s1_id"), "k")
kp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"})
ids = pl.read_parquet(f"{R}/work_v4/p1/test/{C}/s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"})
G = ids.join(kc, on="s1_id", how="left").fill_null(0)["k"].to_numpy()
d = pl.read_parquet(lst).select("s1_id", "m_id").join(kp, on=["s1_id", "m_id"]).join(ids, on="s1_id")
a = d.group_by("s1_id").len("r").join(kc, on="s1_id").with_columns((pl.col("k") - pl.col("r")).alias("post"))
hist = lambda x: np.bincount(np.clip(x, 0, K - 1), minlength=K).astype(float)
print(f"{C}: pairs in file {d.height:,} on {a.height:,} S1 | general mean k {G.mean():.3f} P(k=0) {(G == 0).mean() * 100:.2f} P(k=1) {(G == 1).mean() * 100:.2f}")
for r in (1, 2, 3):
    x = a.filter(pl.col("r") == r)["post"].to_numpy()
    if len(x): print(f"  r={r}: n {len(x):6,}  after mean {x.mean():.3f} +- {x.std() / np.sqrt(len(x)):.3f}  P(after=0) {(x == 0).mean() * 100:.1f}  P(after=1) {(x == 1).mean() * 100:.1f}")
obs = hist(a.filter(pl.col("r") == 1)["post"].to_numpy()); F = hist(G) / len(G); Rr = np.load(f"{R}/work/advisor/real_ref_r1.npy"); Rr = Rr / Rr.sum()
nll = lambda f: -np.dot(obs, np.log(np.clip(f * F + (1 - f) * Rr, 1e-9, None)))
f = minimize_scalar(nll, bounds=(0, 1), method="bounded").x; g = np.linspace(0, 1, 201); ok = g[[nll(v) - nll(f) < 1.92 for v in g]]
print(f"  false share f (r=1) {f:.2f} [{ok.min():.2f}, {ok.max():.2f}]   (removal break-even ~0.33; model error not included)")
