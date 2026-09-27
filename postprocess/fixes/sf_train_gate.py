"""State-parser rescue, TRAIN calibration of the add-only gate (VALID holds one state per country and cannot show the bug).
Same acceptance as a1_apply.py (a1_gate.candidates + accept: new pair, record argmax over its pairs in the small root,
p >= T, shipped rules, one S1 per record), on the train small root scored by model_v10 (w2_sf.sh SPLIT=valid).
'Record free' proxy for 'unassigned in the shipped file': the record has no true pair in the old train cands (records whose
true pair was compared are nearly always assigned by the base). Prints accepted adds and precision per T and country.
Usage: POLARS_MAX_THREADS=3 python postprocess/fixes/sf_train_gate.py [--scored 'work_sf_out/stage2_scored_model_v10sftr_{c}.parquet']"""
import argparse
import sys

import polars as pl

sys.path.insert(0, "postprocess/fixes")
from a1_gate import accept, candidates, load_new  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scored", default="work_sf_out/stage2_scored_model_v10sftr_{c}.parquet")
ap.add_argument("--newpairs-dir", default="work/errfix/out/sf")
ap.add_argument("--ts", default="0.5,0.75,0.9,0.95")
ap.add_argument("--countries", default="US,India")
a = ap.parse_args()
g = (pl.read_parquet("work/parquet/train_ground_truth.parquet").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"})
       .filter(pl.col("m_id").str.len_chars() > 0).with_columns(pl.lit(1).alias("y")))
for c in a.countries.split(","):
    d = f"work_v4/p1/train/{c}"
    sc = pl.read_parquet(a.scored.format(c=c), columns=["s1_id", "m_id", "p"])
    new = load_new(a.newpairs_dir, "valid", c)
    s1 = pl.read_parquet(f"{d}/s1.parquet", columns=["entity_id"]).with_row_index("i").with_columns(pl.col("i").cast(pl.Int32))
    pool = pl.read_parquet(f"{d}/pool.parquet", columns=["entity_id"]).with_row_index("j").with_columns(pl.col("j").cast(pl.Int32))
    old = (pl.read_parquet(f"{d}/cands.parquet", columns=["i", "j"])
             .join(s1.rename({"entity_id": "s1_id"}), on="i").join(pool.rename({"entity_id": "m_id"}), on="j").select("s1_id", "m_id"))
    blocked = old.join(g, on=["s1_id", "m_id"], how="semi")["m_id"].unique()
    nt = new.join(g, on=["s1_id", "m_id"], how="semi").height
    print(f"{c}: new pairs {new.height:,} (true {nt:,}), scored {sc.join(new, on=['s1_id', 'm_id'], how='semi').height:,}, "
          f"records with a compared true pair (blocked) {blocked.len():,}", flush=True)
    for T in map(float, a.ts.split(",")):
        cand, info = candidates(sc, new, blocked, T, "valid", c)
        acc = accept(cand, T).join(g, on=["s1_id", "m_id"], how="left").with_columns(pl.col("y").fill_null(0))
        print(f"  T {T}: argmax & free & p >= T {info['cand']:,} -> accepted {acc.height:,}, precision {acc['y'].mean() if acc.height else float('nan'):.4f}, "
              f"true adds {int(acc['y'].sum()):,} of {nt:,} true new pairs", flush=True)
