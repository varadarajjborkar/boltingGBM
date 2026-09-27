"""Build a submission whose rows for the given countries come from file B and all other rows from file A (whole S1 rows,
so one-S1-per-record holds inside each country). Copies A's candidate_pairs.tsv and runs the official validator.
Usage: python postprocess/fixes/country_swap.py --a output_bucket/stack_l12_fr --b output_bucket/stack_ce_fr \
          --countries France --out work/errfix/sub/stack_l12_frce   (then the lead copies it into output_bucket)"""
import argparse
import shutil
import subprocess
from pathlib import Path
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--countries", default="France")
ap.add_argument("--out", required=True)
x = ap.parse_args()
C = x.countries.split(",")
rd = lambda d: pl.read_csv(Path(d) / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
A, B = rd(x.a), rd(x.b)
s1 = pl.read_parquet("work/parquet/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
ids = set(s1.filter(pl.col("country").is_in(C))["source1_entity_id"].to_list())
bm = dict(zip(B["source1_entity_id"].to_list(), B["matched_entity_ids"].to_list()))
new = [bm.get(s, "") if s in ids else v for s, v in zip(A["source1_entity_id"].to_list(), A["matched_entity_ids"].to_list())]
out = A.with_columns(pl.Series("matched_entity_ids", new))
o = Path(x.out); o.mkdir(parents=True, exist_ok=True)
out.write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
if (Path(x.a) / "candidate_pairs.tsv").exists():
    shutil.copy(Path(x.a) / "candidate_pairs.tsv", o / "candidate_pairs.tsv")
changed = sum(a != b for a, b in zip(A["matched_entity_ids"].to_list(), new))
(o / "NOTES.md").write_text(f"# {o.name}\n\nRows of {C} from {x.b}, others from {x.a}. S1 rows changed vs A: {changed}.\n")
print(f"rows {out.height}, S1 rows changed vs A {changed}")
r = subprocess.run([".venv/bin/python", "data/student_resource/utils/validate_submission.py", "--matching", str(o / "matching_results.tsv"),
                    "--candidate", str(o / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test"], capture_output=True, text=True)
print("validator", "PASS" if r.returncode == 0 else "FAIL", r.stdout.strip()[-200:])
