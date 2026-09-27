# Shared setup (v2): code from the bgbm-code dataset, data from bgbm-fetch, block outputs found by search.
import glob, os, shutil, subprocess, sys, time
def find(pattern, prefer=None):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    if prefer:
        hits = [h for h in hits if prefer in h] or hits
    return hits
REQ = find("requirements.txt", prefer="bgbm-code")[0]
CODE = os.path.dirname(os.path.dirname(os.path.dirname(REQ)))
SRC = f"{CODE}/src"
DATA = os.path.dirname(os.path.dirname(find("work/norm_v2/train_source1.parquet")[0]))
W = "/tmp/work"
os.makedirs(f"{W}/p1", exist_ok=True)
for d in ["norm_v2", "parquet"]:
    if not os.path.exists(f"{W}/{d}"):
        os.symlink(f"{DATA}/{d}", f"{W}/{d}")
def link_split(split, countries):
    for c in countries:
        src = os.path.dirname(find(f"p1/{split}/{c}/cands.parquet")[0])
        dst = f"{W}/p1/{split}/{c}"
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not os.path.exists(f"{dst}/{f}"):
                os.symlink(f"{src}/{f}", f"{dst}/{f}")
# ---- bgbm-k1x: (1) export k1 scores for the team (VALID + test, p >= 0.005); (2) PIPELINE UNION on VALID: pairs k1 scored
# that the team's stack never compared -> how many are true, at what precision, and the VALID macro when added.
import json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
link_split("train", ["US", "India"])
os.environ.update({"ER_WORK_DIR": W, "ER_NORM_SUBDIR": "norm_v2", "ER_THREADS": "4"})
sys.path[:0] = [SRC, f"{CODE}/tools"]
import numpy as np, polars as pl
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
OUT = "/kaggle/working"
k1v = pl.read_parquet(find("model_k1/valid_scored.parquet")[0]).select("s1_id", "m_id", "y", pl.col("p").alias("pk"))
k1v.filter(pl.col("pk") >= 0.005).select("s1_id", "m_id", pl.col("pk").alias("p")).write_parquet(f"{OUT}/k1_valid.parquet")
for c in ["France", "India", "US"]:
    t = pl.read_parquet(find(f"p1/test/{c}/stage2_scored_model_k1.parquet", prefer="bgbm-merge")[0]).select("s1_id", "m_id", "p")
    t.filter(pl.col("p") >= 0.005).write_parquet(f"{OUT}/k1_test_{c}.parquet"); log(f"k1 test {c}: {t.height} pairs, {t.filter(pl.col('p') >= 0.005).height} kept")
team = pl.read_parquet(find("best_valid_scored.parquet", prefer="bgbm-exports")[0]).select("s1_id", "m_id", "y", "p")
log(f"team VALID pairs {team.height} (true {team['y'].sum()}), k1 VALID pairs {k1v.height} (true {k1v['y'].sum()})")
new = k1v.join(team.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti")
log(f"k1-only pairs (never compared by the team): {new.height}, of them true {new['y'].sum()}")
for t_ in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
    x = new.filter(pl.col("pk") >= t_)
    log(f"  k1 p >= {t_}: {x.height} pairs, true {x['y'].sum()}, precision {x['y'].mean() if x.height else 0:.3f}")
from quick_residual import macro, universe
s1v, truth = universe()
base = macro(team.select("s1_id", "m_id", "p"), s1v, truth)
log(f"team stack_s3a2 VALID (quick macro): {json.dumps({k: round(float(v), 5) for k, v in base.items()})}")
res = {"base": base}
for t_ in (0.8, 0.9, 0.95, 0.98):
    add = new.filter(pl.col("pk") >= t_).select("s1_id", "m_id", pl.col("pk").alias("p"))
    r = macro(pl.concat([team.select("s1_id", "m_id", "p"), add]), s1v, truth); res[f"union_{t_}"] = r
    log(f"union k1 p >= {t_}: {r['macro'] - base['macro']:+.5f} | halves {r['h0'] - base['h0']:+.5f} {r['h1'] - base['h1']:+.5f} | US {r['US'] - base['US']:+.5f} India {r['India'] - base['India']:+.5f}")
json.dump(res, open(f"{OUT}/union_result.json", "w"), default=float, indent=1)
print("K1X_DONE", flush=True)
