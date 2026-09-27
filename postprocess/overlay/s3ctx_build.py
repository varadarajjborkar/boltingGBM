import os
"""Stage-3 context inputs: selected v10 pair features as stage-3 extras, one score file per feature (s1_id, m_id, p),
so the stack can learn where the cross-encoder or the stage-2 GBM is more reliable (address missing, number relation,
source, name rarity ...). VALID from train/<c>/feats_valid (restricted to model_v10 stage-2 pairs),
test from test/<c>/kept_feats_model_v10 (= the stage-2 input set).
Writes $ER_WORK_DIR/s3ctx/<feature>_valid.parquet and <feature>_test_<C>.parquet."""
from pathlib import Path

import polars as pl

W = Path(os.environ.get("ER_WORK_DIR", "work"))
O = W / "s3ctx"
O.mkdir(exist_ok=True)
F = ["f_addr_empty", "f_hrel", "f_hn_shift", "f_hdiff_log", "f_is_s3", "f_b_stateless", "f_state_conflict", "f_num_missing",
     "f_num_conflict", "f_legal_fam_conflict", "f_name_tset", "f_addr_tset", "f_rank_comb_s1", "f_n_same_name_s1", "f_non_latin",
     "f_freq_s1name_cty", "f_freq_bname_in_s1_cty", "f_ncand_s1", "f_len_a", "f_len_b", "f_cos_comb"]


def read(parts):
    return pl.concat([pl.read_parquet(p, columns=["s1_id", "m_id"] + F) for p in parts]).unique(["s1_id", "m_id"])


keep = pl.read_parquet(W / "p1" / "model_v10" / "valid_scored.parquet", columns=["s1_id", "m_id"])
v = read([p for c in ("US", "India") for p in sorted((W / "p1" / "train" / c / "feats_valid").glob("part_*.parquet"))])
v = keep.join(v, on=["s1_id", "m_id"], how="left")
print(f"VALID: {keep.height} stage-2 pairs, features missing on {v['f_addr_empty'].is_null().sum()}", flush=True)
for f in F:
    v.select("s1_id", "m_id", pl.col(f).cast(pl.Float64).alias("p")).write_parquet(O / f"{f}_valid.parquet")
for c in ("US", "India", "France"):
    s2 = pl.read_parquet(W / "p1" / "test" / c / "stage2_scored_model_v10.parquet", columns=["s1_id", "m_id"])
    t = s2.join(read(sorted((W / "p1" / "test" / c / "kept_feats_model_v10").glob("part_*.parquet"))), on=["s1_id", "m_id"], how="left")
    print(f"test {c}: {s2.height} pairs, features missing on {t['f_addr_empty'].is_null().sum()}", flush=True)
    for f in F:
        t.select("s1_id", "m_id", pl.col(f).cast(pl.Float64).alias("p")).write_parquet(O / f"{f}_test_{c}.parquet")
print("done")
