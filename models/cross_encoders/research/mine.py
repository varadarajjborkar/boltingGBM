"""Baseline OOF + error mining. Saves research/base_oof.parquet and prints error examples."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
d = load_base()
log(f"SANITY raw stack p (must be ~0.9873): {macro(d.select('s1_id','m_id','p'))}")
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
po = oof(np.column_stack(base_X(d)), yy, grp)
A = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", po))); log(f"BASE OOF {A}")
json.dump(A, open(f"{R}/base_res.json", "w"))
t = A["t"]
e = d.with_columns(pl.Series("po", po))
# one S1 per record
e = e.with_columns((pl.col("po") == pl.col("po").max().over("m_id")).alias("top"))
e = e.with_columns(((pl.col("po") >= t) & pl.col("top")).alias("pred"))
e.write_parquet(f"{R}/base_oof.parquet")
fp = e.filter(pl.col("pred") & (pl.col("y") == 0)); fn = e.filter(~pl.col("pred") & (pl.col("y") == 1))
# true pairs never in candidates
cand_true = e.filter(pl.col("y") == 1).select("s1_id", "m_id")
miss = gv.join(cand_true, on=["s1_id", "m_id"], how="anti")
log(f"threshold {t}: FP {fp.height} FN(in cand) {fn.height} (FN because outranked by other S1: {fn.filter(~pl.col('top')).height}); true pairs missing from candidates: {miss.height}; total true {gv.height}")
