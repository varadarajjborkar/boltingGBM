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
# ---- bgbm-comp: record-level competition on cross-encoder scores (address-less records, record given to the wrong S1).
# Base = the team's stack_s3a2 VALID p (cross-fitted); + e5 tagged CE; + per-record rank/margin of p and CE across the
# record's candidate S1s, record address-emptiness. Residual (2 folds by S1 hash), both halves.
import json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
link_split("train", ["US", "India"])
os.environ.update({"ER_WORK_DIR": W, "ER_NORM_SUBDIR": "norm_v2", "ER_THREADS": "4"})
sys.path[:0] = [SRC, f"{CODE}/tools"]
import numpy as np, polars as pl
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
sc = pl.read_parquet(find("best_valid_scored.parquet", prefer="bgbm-exports")[0]).select("s1_id", "m_id", "y", "p")
ce = pl.read_parquet(find("ce2_team_valid.parquet")[0]).rename({"p": "ce"})
ce1 = pl.read_parquet(find("ce_team_valid.parquet", prefer="bgbm-ce-team")[0]).rename({"p": "ce1"})
emp = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=["entity_id", "addr_empty"]) for s in (2, 3)])
d = (sc.join(ce, on=["s1_id", "m_id"], how="left", maintain_order="left").join(ce1, on=["s1_id", "m_id"], how="left", maintain_order="left")
       .join(emp.rename({"entity_id": "m_id"}), on="m_id", how="left", maintain_order="left"))
d = d.with_columns(pl.col("ce").is_not_null().cast(pl.Float64).alias("ce_flag"), pl.col("ce").fill_null(pl.col("p")).alias("ce_f"), pl.col("ce1").fill_null(pl.col("p")).alias("ce1_f"),
                   pl.col("addr_empty").fill_null(False).cast(pl.Float64).alias("rec_noaddr"))
g = "m_id"
d = d.with_columns(
    pl.len().over(g).alias("rec_ncand"),
    pl.col("p").rank("ordinal", descending=True).over(g).alias("p_rank_rec"),
    pl.col("ce_f").rank("ordinal", descending=True).over(g).alias("ce_rank_rec"),
    pl.col("p").max().over(g).alias("_pmax"), pl.col("ce_f").max().over(g).alias("_cmax"),
)
two = d.group_by(g).agg(pl.col("p").sort(descending=True).head(2).alias("_tp"), pl.col("ce_f").sort(descending=True).head(2).alias("_tc"))
two = two.with_columns(pl.col("_tp").list.get(1, null_on_oob=True).alias("_p2"), pl.col("_tc").list.get(1, null_on_oob=True).alias("_c2")).drop("_tp", "_tc")
d = d.join(two, on=g, how="left", maintain_order="left")
d = d.with_columns(
    pl.when(pl.col("p") >= pl.col("_pmax")).then(pl.col("_p2")).otherwise(pl.col("_pmax")).fill_null(0.0).alias("p_best_other"),
    pl.when(pl.col("ce_f") >= pl.col("_cmax")).then(pl.col("_c2")).otherwise(pl.col("_cmax")).fill_null(0.0).alias("ce_best_other"))
d = d.with_columns((pl.col("p") - pl.col("p_best_other")).alias("p_margin_rec"), (pl.col("ce_f") - pl.col("ce_best_other")).alias("ce_margin_rec"))
from quick_residual import macro, universe, oof
s1v, truth = universe()
p = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
F = lambda cols: np.column_stack([lp] + [d[c].cast(pl.Float64).to_numpy() for c in cols])
comp = ["rec_ncand", "p_rank_rec", "ce_rank_rec", "p_best_other", "ce_best_other", "p_margin_rec", "ce_margin_rec", "rec_noaddr"]
sets = {"A stack_s3a2 p": [], "B + e5 v1": ["ce1_f", "ce_flag"], "C + e5 v2 tagged": ["ce_f", "ce_flag"],
        "D + v1 + v2": ["ce1_f", "ce_f", "ce_flag"], "E + v1 + v2 + record competition": ["ce1_f", "ce_f", "ce_flag"] + comp}
res = {}
for name, cols in sets.items():
    res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(F(cols), yy, grp))), s1v, truth)
    log(f"{name:30s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
names = list(res)
for a_, b_ in ((names[0], names[1]), (names[0], names[2]), (names[0], names[3]), (names[3], names[4])):
    A, B = res[a_], res[b_]
    log(f"{b_} vs {a_}: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
json.dump(res, open("/kaggle/working/comp_result.json", "w"), default=float, indent=1)
print("COMP_DONE", flush=True)
