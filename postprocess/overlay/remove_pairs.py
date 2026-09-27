"""Remove listed (s1_id, m_id) pairs from a submission (records become unassigned); candidates copied unchanged.
Used for the teammate's France descriptor-swap removal (exports/france_swap/fr_swap_remove.parquet, no moves).
Usage: python postprocess/overlay/remove_pairs.py --base output_bucket/<in> --remove <pairs.parquet> --out output_bucket/<out>"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--remove", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
base = pl.read_csv(f"{a.base}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
r = pl.read_parquet(a.remove, columns=["s1_id", "m_id"])
drop = set(zip(r["s1_id"].to_list(), r["m_id"].to_list()))
n = 0; new = []
for s, v in zip(base["source1_entity_id"].to_list(), base["matched_entity_ids"].to_list()):
    keep = [m for m in v.split(",") if m and (s, m) not in drop]
    n += len([m for m in v.split(",") if m]) - len(keep); new.append(",".join(keep))
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
base.with_columns(pl.Series("matched_entity_ids", new)).write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
shutil.copy(f"{a.base}/candidate_pairs.tsv", o / "candidate_pairs.tsv")
print(f"removed {n} of {len(drop)} listed pairs")
sys.path.insert(0, "postprocess/fixes")
from sub_check import check  # noqa: E402
assert check(str(o / "matching_results.tsv"), str(o / "candidate_pairs.tsv")), "sub_check FAIL"
rr = subprocess.run([sys.executable, "data/student_resource/utils/validate_submission.py", "--matching", str(o / "matching_results.tsv"), "--candidate",
                     str(o / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test", "--check-ids"], capture_output=True, text=True)
print("\n".join((rr.stdout + rr.stderr).strip().splitlines()[-1:]), f"| validator exit code {rr.returncode}")
