"""Phase 0: build faithful regional dev samples from TRAIN (see docs/PROBLEM_DECOMPOSITION.md).

Whole states are taken so that the local look-alike density equals the real one.
  dev_train : states A  (used to fit models)
  dev_valid : states B  (disjoint from A; used to score)
  dev_valid_dense : dev_valid with a fraction of S1 dropped so S2/S3-per-S1 matches the test files
Each sample dir holds: s1.parquet, pool.parquet (S2+S3 records), gt.parquet (s1_id, m_id pairs), meta.json

State of an S2/S3 record: its matched S1's state (train labels) if matched, else parsed from its address.
Unparseable-state unmatched records are included at the region's share of the country (density-preserving).
"""
import argparse
import json
import random

import polars as pl

from utils import WORK, announce_pid, done, log, write_parquet_atomic

TEST_RATIO = {"US": 5.76, "India": 5.82}      # (test S2+S3) / test S1, measured in EDA


def load():
    n = lambda f: pl.read_parquet(WORK / "norm" / f"{f}.parquet")
    s1 = n("train_source1")
    s23 = pl.concat([n("train_source2").with_columns(pl.lit("S2").alias("src")),
                     n("train_source3").with_columns(pl.lit("S3").alias("src"))])
    gt = (pl.read_parquet(WORK / "parquet" / "train_ground_truth.parquet")
            .filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
            .explode("m").select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
    return s1, s23, gt


def pick_states(counts: pl.DataFrame, target: int, rng: random.Random, exclude=set()):
    """Random whole states (excluding the 2 largest to keep samples small) until ~target S1 records."""
    rows = [r for r in counts.sort("n", descending=True).iter_rows() if r[0] not in exclude][2:]
    rng.shuffle(rows)
    chosen, tot = [], 0
    for st, n in rows:
        if tot >= target:
            break
        if n > target * 0.7:          # avoid one state dominating the sample
            continue
        chosen.append(st)
        tot += n
    return chosen, tot


def build(s1, s23, gt, states_by_country, name, drop_to_test_density=False, seed=0):
    rng = random.Random(seed)
    s1_state = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("addr_state").alias("s1_state"), "country")
    # state of each S2/S3 record: matched S1's state, else parsed
    m_state = gt.join(s1_state, on="s1_id").select(pl.col("m_id").alias("entity_id"), "s1_state")
    rec = (s23.join(m_state, on="entity_id", how="left")
              .with_columns(pl.when(pl.col("s1_state").is_not_null()).then(pl.col("s1_state"))
                              .otherwise(pl.col("addr_state")).alias("region_state")))
    S1s, pools, meta = [], [], {}
    for country, states in states_by_country.items():
        c_s1 = s1.filter(pl.col("country") == country)
        known = c_s1.filter(pl.col("addr_state") != "").height
        sel_s1 = c_s1.filter(pl.col("addr_state").is_in(states))
        frac = sel_s1.height / known
        c_rec = rec.filter(pl.col("country") == country)
        in_region = c_rec.filter(pl.col("region_state").is_in(states))
        unknown = c_rec.filter(pl.col("region_state") == "").sample(fraction=frac, seed=seed)
        pool = pl.concat([in_region, unknown])
        ratio = pool.height / sel_s1.height
        dropped = 0
        if drop_to_test_density:
            d = max(0.0, 1 - ratio / TEST_RATIO[country])
            keep = sel_s1.sample(fraction=1 - d, seed=seed + 1)
            dropped = sel_s1.height - keep.height
            sel_s1 = keep
        meta[country] = {"states": states, "s1": sel_s1.height, "pool": pool.height, "unknown_state_added": unknown.height,
                         "pool_per_s1": round(pool.height / sel_s1.height, 3), "dropped_s1": dropped, "region_frac": round(frac, 4)}
        S1s.append(sel_s1)
        pools.append(pool)
    S1 = pl.concat(S1s)
    POOL = pl.concat(pools).drop("s1_state", "region_state")
    GT = gt.join(S1.select(pl.col("entity_id").alias("s1_id")), on="s1_id").join(
        POOL.select(pl.col("entity_id").alias("m_id")), on="m_id")
    n_true_total = gt.join(S1.select(pl.col("entity_id").alias("s1_id")), on="s1_id").height
    meta["gt_pairs_in_pool"] = GT.height
    meta["gt_pairs_lost_outside_pool"] = n_true_total - GT.height
    meta["singleton_frac"] = round(1 - GT["s1_id"].n_unique() / S1.height, 4)
    out = WORK / "phase0" / name
    write_parquet_atomic(S1, out / "s1.parquet")
    write_parquet_atomic(POOL, out / "pool.parquet")
    write_parquet_atomic(GT, out / "gt.parquet")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{name}: {json.dumps(meta)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train_target", type=int, default=60000)
    ap.add_argument("--valid_target", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    announce_pid("phase0_sample")
    if done(WORK / "phase0" / "dev_valid_dense" / "meta.json", a.force):
        log("samples exist; use --force to rebuild")
        return
    s1, s23, gt = load()
    rng = random.Random(a.seed)
    tr, va = {}, {}
    for country in ["US", "India"]:
        counts = (s1.filter((pl.col("country") == country) & (pl.col("addr_state") != ""))
                    .group_by("addr_state").agg(pl.len().alias("n")))
        va[country], nv = pick_states(counts, a.valid_target, rng)
        tr[country], nt = pick_states(counts, a.train_target, rng, exclude=set(va[country]))
        log(f"{country}: valid states {va[country]} ({nv} S1) | train states {tr[country]} ({nt} S1)")
    build(s1, s23, gt, tr, "dev_train", seed=a.seed)
    build(s1, s23, gt, va, "dev_valid", seed=a.seed)
    build(s1, s23, gt, va, "dev_valid_dense", drop_to_test_density=True, seed=a.seed)


if __name__ == "__main__":
    main()
