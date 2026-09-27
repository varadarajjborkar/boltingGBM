"""A/B variant: remove the pairs of one country whose decision score lies in [lo, hi) (records become unassigned).
France decision scores = stack_ce's model_v10e_v10_s3. Rider pairs without a score (rescue adds) are kept.
Usage: python postprocess/overlay/drop_band.py --base output_bucket/<in> --out output_bucket/<out> [--country France --lo .75 --hi .9]"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--country", default="France"); ap.add_argument("--lo", type=float, default=0.75); ap.add_argument("--hi", type=float, default=0.9)
ap.add_argument("--scores", default="stage2_scored_model_v10e_v10_s3.parquet")
a = ap.parse_args()
base = pl.read_csv(f"{a.base}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
sc = pl.read_parquet(f"work_v4/p1/test/{a.country}/{a.scores}", columns=["s1_id", "m_id", "p"])
drop = sc.filter((pl.col("p") >= a.lo) & (pl.col("p") < a.hi))
drop = set(zip(drop["s1_id"].to_list(), drop["m_id"].to_list()))
n = 0; new = []
for s, v in zip(base["source1_entity_id"].to_list(), base["matched_entity_ids"].to_list()):
    keep = []
    for m in [x for x in v.split(",") if x]:
        if (s, m) in drop: n += 1
        else: keep.append(m)
    new.append(",".join(keep))
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
base.with_columns(pl.Series("matched_entity_ids", new)).write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
shutil.copy(f"{a.base}/candidate_pairs.tsv", o / "candidate_pairs.tsv")
print(f"dropped {n} {a.country} pairs with score in [{a.lo}, {a.hi})")
sys.path.insert(0, "postprocess/fixes")
from sub_check import check  # noqa: E402
assert check(str(o / "matching_results.tsv"), str(o / "candidate_pairs.tsv")), "sub_check FAIL"
r = subprocess.run([sys.executable, "data/student_resource/utils/validate_submission.py", "--matching", str(o / "matching_results.tsv"), "--candidate",
                    str(o / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test", "--check-ids"], capture_output=True, text=True)
print("\n".join((r.stdout + r.stderr).strip().splitlines()[-1:]), f"| validator exit code {r.returncode}")
