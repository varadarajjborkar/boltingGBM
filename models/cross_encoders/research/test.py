"""Residual tests: baseline [logit p, ce1, ce2, flag] vs baseline + each hypothesis group. Also seed-noise floor."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
d = load_base()
F = pl.read_parquet(f"{R}/feats_valid.parquet")
assert F.height == d.height
# ---- H4: S1-side competition (label-free: team p and e5 scores only)
g = "s1_id"
s = d.with_columns(pl.len().over(g).cast(pl.Float64).alias("s1_ncand"), pl.col("p").rank("ordinal", descending=True).over(g).cast(pl.Float64).alias("p_rank_s1"),
                   pl.col("c1").rank("ordinal", descending=True).over(g).cast(pl.Float64).alias("c_rank_s1"),
                   (pl.col("p") > 0.5).sum().over(g).cast(pl.Float64).alias("s1_n05"), pl.col("p").sum().over(g).alias("s1_psum"))
s = s.with_columns((pl.col("s1_psum") - pl.col("p")).alias("s1_psum_other"), (pl.col("s1_n05") - (pl.col("p") > 0.5).cast(pl.Float64)).alias("s1_n05_other"))
H4 = {c: s[c].to_numpy() for c in ("s1_ncand", "p_rank_s1", "c_rank_s1", "s1_n05_other", "s1_psum_other")}
grp = lambda pre: {c: F[c].to_numpy() for c in F.columns if c.startswith(pre)}
tests = {"H1 house-number": grp("hn_"), "H2 name-edit": grp("ne_"), "H3 distinctive+namecnt": grp("dw_"),
         "H3a distinctive only": {k: v for k, v in grp("dw_").items() if "namecnt" not in k}, "H3b namecnt only": {k: v for k, v in grp("dw_").items() if "namecnt" in k},
         "H4 S1-competition": H4}
only = sys.argv[1:]
res = {}
for name, fe in tests.items():
    if only and not any(name.startswith(o) for o in only): continue
    t0 = time.time(); A, B = run_test(d, fe, name); res[name] = B; log(f"  ({time.time() - t0:.0f}s)")
if not only:
    # seed-noise floor: baseline features, different LightGBM seed
    yy, grp_ = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
    B = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(np.column_stack(base_X(d)), yy, grp_, seed=8))))
    log(f"NOISE baseline seed 8 vs 7: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f}")
json.dump(res, open(f"{R}/test_result_{'_'.join(only) or 'all'}.json", "w"), default=float, indent=1)
print("TEST_DONE", flush=True)
