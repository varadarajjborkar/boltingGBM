# Save stage-3 cross-fitted VALID scores (same design/params/halves as stage3.fit) as model_v10_s3/valid_scored.parquet
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, polars as pl, lightgbm as lgb
import stage3 as S
M = S.WORK / "p1" / sys.argv[1]
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=4, seed=7)
sc = pl.read_parquet(M / "valid_scored.parquet")
ex = {x.split("=")[0]: pl.read_parquet(x.split("=")[1]) for x in filter(None, (sys.argv[2] if len(sys.argv) > 2 else "").split(";"))}
d, cols = S.design(sc.select("s1_id", "m_id", "y", "p"), pl.read_parquet(M / "stage3" / "cn_valid.parquet"), ex)
y, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
X = d.select(cols).to_numpy().astype(np.float64)
oof = np.zeros(len(y))
for k in (0, 1):
    m = lgb.train(params, lgb.Dataset(X[grp != k], y[grp != k]), 300)
    oof[grp == k] = m.predict(X[grp == k])
out = sc.drop("p").join(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof)), on=["s1_id", "m_id"], how="left")
O = S.WORK / "p1" / f"{sys.argv[1]}_s3"
O.mkdir(exist_ok=True)
out.write_parquet(O / "valid_scored.parquet")
print(out.shape, out.columns, "null p", out["p"].null_count(), "->", O / "valid_scored.parquet")
