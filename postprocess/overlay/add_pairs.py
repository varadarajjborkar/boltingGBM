"""Add-only merge: base submission + extra (s1_id, m_id) pairs (records not used in the base, one S1 per record).
Copies the base candidate_pairs.tsv (adds must already be candidates; sub_check verifies), runs sub_check and the validator.
Usage: python postprocess/overlay/add_pairs.py --base output_bucket/stack_e5hyb2_a1 --adds pairs.parquet --out output_bucket/<new>"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adds", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = pl.read_csv(f"{a.base}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
used = {x for v in base["matched_entity_ids"].to_list() for x in v.split(",") if x}
add = pl.read_parquet(a.adds, columns=["s1_id", "m_id"]).unique("m_id")
add = add.filter(~pl.col("m_id").is_in(list(used)))
ad = {}
for s, m in zip(add["s1_id"].to_list(), add["m_id"].to_list()):
    ad.setdefault(s, []).append(m)
new = [",".join([x for x in v.split(",") if x] + ad.get(s, [])) for s, v in zip(base["source1_entity_id"].to_list(), base["matched_entity_ids"].to_list())]
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
base.with_columns(pl.Series("matched_entity_ids", new)).write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
shutil.copy(f"{a.base}/candidate_pairs.tsv", o / "candidate_pairs.tsv")
print(f"added {add.height} pairs to {len(ad)} S1 (records already used in base skipped)", flush=True)
sys.path.insert(0, "postprocess/fixes")
from sub_check import check  # noqa: E402
assert check(str(o / "matching_results.tsv"), str(o / "candidate_pairs.tsv")), "sub_check FAIL"
r = subprocess.run([sys.executable, "data/student_resource/utils/validate_submission.py", "--matching", str(o / "matching_results.tsv"), "--candidate",
                    str(o / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test", "--check-ids"], capture_output=True, text=True)
print("\n".join((r.stdout + r.stderr).strip().splitlines()[-4:]), f"\nvalidator exit code {r.returncode}")
