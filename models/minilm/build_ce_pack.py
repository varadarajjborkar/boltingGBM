import os
"""Builds the cross-encoder fine-tuning package (runs on worker 2). Training pairs come ONLY from training states
(no VALID labels): (1) stage-2 training states tx/up/ka, every pair with v10 p1 >= 0.01 (the char net's set, uncapped);
(2) all other FIT states (feats_fit_ent80 = stage-1 FIT, US + India): a sample of true pairs and of the hardest non-pairs (highest f_cos_comb).
Scoring sets = the stage-3 uncertain pairs of VALID and test (cn_valid / cn_test lists).
Writes $ER_WORK_DIR/ce_pack/: train_pairs, valid_pairs, test_pairs (ids), train_records, test_records (entity_id, text).
Usage: python build_ce_pack.py [N_POS N_NEG]"""
import glob
import sys
from pathlib import Path

import polars as pl

A = Path(os.environ.get("ER_ROOT", "."))
W, P = A / "work_v4", A / "work" / "parquet"
O = W / "ce_pack"
O.mkdir(exist_ok=True)
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 700_000
N_NEG = int(sys.argv[2]) if len(sys.argv) > 2 else 900_000

s2 = (pl.read_parquet(W / "p1" / "model_v10" / "s2fit_p1.parquet", columns=["s1_id", "m_id", "y", "p1"])
        .filter(pl.col("p1") >= 0.01).select("s1_id", "m_id", pl.col("y").cast(pl.Int8), pl.lit("s2fit").alias("src")))
fit = pl.concat([pl.read_parquet(f, columns=["s1_id", "m_id", "y", "f_cos_comb"])
                 for c in ("US", "India") for f in sorted(glob.glob(str(W / "p1" / "train" / c / "feats_fit_ent80" / "*.parquet")))])
fit = fit.join(s2.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti")
pos = fit.filter(pl.col("y") == 1)
pos = pos.sample(n=min(N_POS, pos.height), seed=7)
neg = fit.filter(pl.col("y") == 0).sort("f_cos_comb", descending=True).head(N_NEG * 3)
neg = neg.sample(n=min(N_NEG, neg.height), seed=7)
fitp = pl.concat([pos, neg]).select("s1_id", "m_id", pl.col("y").cast(pl.Int8), pl.lit("fit").alias("src"))
tr = pl.concat([s2, fitp]).unique(["s1_id", "m_id"]).sample(fraction=1.0, shuffle=True, seed=7)
tr = tr.with_columns((pl.col("s1_id").hash(29) % 50 == 0).alias("holdout"))
print(f"train pairs {tr.height}: s2fit {s2.height} (pos {s2['y'].mean():.3f}), fit pos {pos.height}, fit hard neg {neg.height} "
      f"(fit pool {fit.height}); holdout {tr['holdout'].sum()}")
va = pl.read_parquet(W / "p1" / "model_v10" / "stage3" / "cn_valid.parquet").select("s1_id", "m_id")
te = pl.read_parquet(W / "p1" / "model_v10" / "stage3" / "cn_test.parquet").select("s1_id", "m_id", "country")


def records(split, ids):
    r = pl.concat([pl.read_parquet(P / f"{split}_source{i}.parquet", columns=["entity_id", "business_name", "business_address"])
                   for i in (1, 2, 3)]).join(ids.to_frame("entity_id"), on="entity_id", how="semi")
    addr = r["business_address"].fill_null("").str.replace_all(r"(?i)<?null>?", "").str.strip_chars(" ,")
    return r.select("entity_id", (pl.col("business_name").fill_null("") + " | " + addr).alias("text"))


tri = pl.concat([tr["s1_id"], tr["m_id"], va["s1_id"], va["m_id"]]).unique()
tei = pl.concat([te["s1_id"], te["m_id"]]).unique()
rtr, rte = records("train", tri), records("test", tei)
assert rtr.height == tri.len() and rte.height == tei.len(), "missing texts"
for name, d in [("train_pairs", tr), ("valid_pairs", va), ("test_pairs", te), ("train_records", rtr), ("test_records", rte)]:
    d.write_parquet(O / f"{name}.parquet", compression="zstd")
    print(name, d.shape)
