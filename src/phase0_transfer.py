"""France proxy: how much do we lose on a country the model never saw?

Train stage-1 on ONE country of dev_train, score the OTHER country of dev_valid, compare with the model trained on
both. Also reports which features' importance shifts, to spot features that will not transfer.
Usage: python phase0_transfer.py --tag v3
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import by_threshold, one_s1_per_record
from run_phase0 import f32_matrix, feat_cols, load_set, score_report, tied_distractors
from utils import WORK, announce_pid, log, timed

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=7, verbose=-1, seed=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    announce_pid("phase0_transfer")
    tr = pl.read_parquet(WORK / "phase0" / a.tag / "dev_train" / "feats.parquet")
    va = pl.read_parquet(WORK / "phase0" / a.tag / "dev_valid" / "feats.parquet")
    cols = feat_cols(tr)
    s1, pool, gt = load_set("dev_valid")
    kmap = tied_distractors(s1, pool, gt)
    rounds = 525
    res = {}
    for src, dst in [("US", "India"), ("India", "US"), ("both", "India"), ("both", "US")]:
        t = tr if src == "both" else tr.filter(pl.col("a_country") == src)
        v = va.filter(pl.col("a_country") == dst)
        with timed(f"train on {src}, score {dst}"):
            m = lgb.train(PARAMS, lgb.Dataset(f32_matrix(t, cols), t["y"].to_numpy()), num_boost_round=rounds)
        sc = v.select("s1_id", "m_id", "y").with_columns(pl.Series("p", m.predict(f32_matrix(v, cols), num_threads=7)))
        s1d = s1.filter(pl.col("country") == dst)
        best = max(((t_, score_report(by_threshold(one_s1_per_record(sc), t_), s1d, gt, kmap))
                    for t_ in np.arange(0.4, 0.96, 0.05).round(2)), key=lambda x: x[1]["weighted_f05"])
        res[f"{src}->{dst}"] = {"t": float(best[0]), "macro": best[1]["macro_f05"], "weighted": best[1]["weighted_f05"]}
        log(f"{src} -> {dst}: {res[f'{src}->{dst}']}")
    (WORK / "phase0" / a.tag / "transfer_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
