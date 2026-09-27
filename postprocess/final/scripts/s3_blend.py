# Stage-3 blend of small rejected tweaks (ff0.5/600, bagged seeds, address-split) on the pkf2 package inputs.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "stage3"))
import numpy as np, polars as pl, lightgbm as lgb
import stage3 as S
M = S.WORK / "p1" / "model_v10pkf2"
specs = {}
for x in filter(None, sys.argv[1].split(";")):
    n, rest = x.split("=", 1); v, t = rest.split("@", 1); specs[n] = (v, t)
s1v, truth = S.universe()
sc = pl.read_parquet(M / "valid_scored.parquet").select("s1_id", "m_id", "y", "p")
d, cols = S.design(sc, pl.read_parquet(M / "stage3" / "cn_valid.parquet"), {n: pl.read_parquet(v) for n, (v, _) in specs.items()})
X = d.select(cols).to_numpy().astype(np.float64); y = d["y"].to_numpy(); grp = (d["s1_id"].hash(13) % 2).to_numpy()
na = d["x_rec_noaddr"].fill_null(0).to_numpy() > 0.5
B = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=4, seed=7)
bag = {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}
V = {"v0": ({}, 300, False), "ff05r600": ({"feature_fraction": 0.5}, 600, False),
     "bag1": ({**bag, "seed": 1}, 300, False), "bag2": ({**bag, "seed": 2}, 300, False), "bag3": ({**bag, "seed": 3}, 300, False),
     "split": ({}, 300, True), "ff05split": ({"feature_fraction": 0.5}, 600, True)}
def fitpred(prm, r, split, Xtr, ytr, natr, Xte, nate):
    o = np.zeros(len(Xte))
    for g in ((True, False) if split else (None,)):
        tr = np.ones(len(ytr), bool) if g is None else natr == g
        te = np.ones(len(Xte), bool) if g is None else nate == g
        m = lgb.train({**B, **prm}, lgb.Dataset(Xtr[tr], ytr[tr]), r); o[te] = m.predict(Xte[te])
    return o
def mac(o): return S.macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", o)), s1v, truth)
O = {}
for n, (prm, r, sp) in V.items():
    o = np.zeros(len(y))
    for k in (0, 1):
        o[grp == k] = fitpred(prm, r, sp, X[grp != k], y[grp != k], na[grp != k], X[grp == k], na[grp == k])
    O[n] = o
R0 = mac(O["v0"]); print("v0", json.dumps({k: round(float(v), 5) for k, v in R0.items()}), flush=True)
def show(n, o):
    r = mac(o); print(f"{n:28s} macro {r['macro']:.5f} gain {r['macro']-R0['macro']:+.5f} h0 {r['h0']-R0['h0']:+.5f} h1 {r['h1']-R0['h1']:+.5f} US {r['US']-R0['US']:+.5f} IN {r['India']-R0['India']:+.5f} t {r['t']}", flush=True); return r
res = {}
for n in V:
    if n != "v0": res[n] = show(n, O[n])
blends = {"all": list(V), "v0+bags": ["v0", "bag1", "bag2", "bag3"], "v0+ff05+split": ["v0", "ff05r600", "split"],
          "no_v0": [n for n in V if n != "v0"], "v0+ff05+bags+split": ["v0", "ff05r600", "bag1", "bag2", "bag3", "split"]}
for n, mem in blends.items():
    res["B_" + n] = show("B_" + n, np.mean([O[m] for m in mem], axis=0))
ok = {n: r for n, r in res.items() if n.startswith("B_") and r["h0"] > R0["h0"] and r["h1"] > R0["h1"]}
best = max(ok, key=lambda n: ok[n]["macro"]) if ok else "B_all"
mem = blends[best[2:]]; print("CHOSEN", best, mem, flush=True)
pb = np.mean([O[m] for m in mem], axis=0)
d.select("s1_id", "m_id", "y").with_columns(pl.Series("p0", O["v0"]), pl.Series("pb", pb)).write_parquet(M / "stage3" / "blend_oof.parquet")
json.dump({"chosen": best, "members": mem, "t0": R0["t"], "tb": res[best]["t"]}, open(M / "stage3" / "blend.json", "w"))
cn_test = pl.read_parquet(M / "stage3" / "cn_test.parquet")
for c in ["US", "India"]:
    te = pl.read_parquet(S.WORK / "p1" / "test" / c / "stage2_scored_model_v10pkf2.parquet").select("s1_id", "m_id", "p")
    ex = {n: pl.read_parquet(Path(t.format(c=c))) for n, (_, t) in specs.items()}
    dt, _ = S.design(te, cn_test.filter(pl.col("country") == c), ex)
    Xt = dt.select(cols).to_numpy().astype(np.float64); nat = dt["x_rec_noaddr"].fill_null(0).to_numpy() > 0.5
    pt = np.mean([fitpred(V[m][0], V[m][1], V[m][2], X, y, na, Xt, nat) for m in mem], axis=0)
    dt.select("s1_id", "m_id").with_columns(pl.Series("p", pt)).write_parquet(S.WORK / "p1" / "test" / c / "stage2_scored_model_v10pkf2b_s3.parquet")
    print("test", c, dt.height, flush=True)
print("BLEND DONE", flush=True)
