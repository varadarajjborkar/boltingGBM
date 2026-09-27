"""Pick the (parsed state, other state code) combos to re-block, from the TRAIN tables written by the state-rule scan
(work/errfix/out/state_rules_<C>_{s1,rec}.parquet): true partners mostly in the other state (in_alt >= --min-alt and
in_alt > 2 x in_parsed) and enough missed pairs (alt_missed >= --min-missed). Writes work/errfix/out/state_rules_sel.json."""
import argparse
import json

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--min-alt", type=float, default=0.6)
ap.add_argument("--min-missed", type=int, default=10)
ap.add_argument("--ratio", type=float, default=2.0, help="in_alt > ratio x in_parsed")
ap.add_argument("--out", default="work/errfix/out/state_rules_sel.json")
a = ap.parse_args()
out = {}
for c in ("US", "India"):
    out[c] = {}
    for side in ("s1", "rec"):
        t = pl.read_parquet(f"work/errfix/out/state_rules_{c}_{side}.parquet")
        k = t.filter((pl.col("in_alt") >= a.min_alt) & (pl.col("in_alt") > a.ratio * pl.col("in_parsed")) & (pl.col("alt_missed") >= a.min_missed))
        out[c][side] = [[p, q] for p, q in zip(k["parsed"].to_list(), k["alt"].to_list())]
        print(f"{c} {side}: {k.height} combos, missed true pairs covered {k['alt_missed'].sum():,} of {t['alt_missed'].sum():,}; "
              f"top {k.sort('alt_missed', descending=True).head(6).select('parsed', 'alt', 'alt_missed', 'in_alt').rows()}")
json.dump(out, open(a.out, "w"), indent=1)
