"""Apply the teammate's France exact-match fix (exports/france_fix on msrit-aws-setup) to a submission, with our safeguards.
adds  (s1_id, m_id, why): record unassigned in the base -> add to s1_id. Kept kinds: --adds-why (default unique,group).
moves (from_s1_id, s1_id, m_id, why): record currently at from_s1_id -> move to s1_id. Kept kinds: --moves-why (default unique),
      and only when our stage-2 model never scored (s1_id, m_id) (blocking missed the exact S1), unless --moves-seen.
Evidence (27 Sep): unique exact key (same name core, state, house numbers; one S1 with that core) = owner for 99.9% of train
records even with different streets; group key picked by legal form = owner 0.845 (US) / 0.926 (India); on VALID our decision
never disagrees with the unique key (0 of 102,430), so moves where our model saw both S1 have no analogue and are skipped.
New pairs join candidate_pairs.tsv (the exact-match rescue path is a candidate generator). sub_check + validator.
review (13:45): French names are generic, so only same-street adds are precise (.99 by a chance control; different street ~.25)
and moves hurt: use --adds-file work/advisor/fr_exact_adds_samestreet.parquet --moves-why "".
Usage: python postprocess/overlay/apply_fr_fix.py --base output_bucket/<in> --out output_bucket/<out>"""
import os
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

E = Path(os.environ.get("ER_EXPORTS", "exports")) / "france_fix"
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--adds-why", default="unique,group")
ap.add_argument("--moves-why", default="unique")
ap.add_argument("--moves-seen", action="store_true")
ap.add_argument("--adds-file", default="", help="use this adds list instead (e.g. advisor same-street filter)")
a = ap.parse_args()
base = pl.read_csv(f"{a.base}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
lists = {s: [x for x in v.split(",") if x] for s, v in zip(base["source1_entity_id"].to_list(), base["matched_entity_ids"].to_list())}
where = {m: s for s, v in lists.items() for m in v}
seen = pl.read_parquet("work_v4/p1/test/France/stage2_scored_model_v10.parquet", columns=["s1_id", "m_id"])
seen = set(zip(seen["s1_id"].to_list(), seen["m_id"].to_list()))

ad = pl.read_parquet(a.adds_file or E / "fr_exact_adds.parquet").filter(pl.col("why").is_in(a.adds_why.split(",")))
mv = pl.read_parquet(E / "fr_exact_moves.parquet").filter(pl.col("why").is_in(a.moves_why.split(",")) if a.moves_why else pl.lit(False))
n_add = n_mv = skip = 0
new_pairs = {}
for s, m in zip(ad["s1_id"].to_list(), ad["m_id"].to_list()):
    if m in where or s not in lists:
        skip += 1; continue
    lists[s].append(m); where[m] = s; n_add += 1
    new_pairs.setdefault(s, []).append(m)
for f, s, m in zip(mv["from_s1_id"].to_list(), mv["s1_id"].to_list(), mv["m_id"].to_list()):
    if where.get(m) != f or s not in lists or ((s, m) in seen and not a.moves_seen):
        skip += 1; continue
    lists[f].remove(m); lists[s].append(m); where[m] = s; n_mv += 1
    new_pairs.setdefault(s, []).append(m)
assert len(where) == sum(len(v) for v in lists.values()), "record in two S1"
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
base.with_columns(pl.Series("matched_entity_ids", [",".join(lists[s]) for s in base["source1_entity_id"].to_list()])).write_csv(
    o / "matching_results.tsv", separator="\t", quote_style="never")
print(f"France fix: adds {n_add} (kinds {a.adds_why}), moves {n_mv} (kinds {a.moves_why or 'none'}{', seen allowed' if a.moves_seen else ', unseen only'}), skipped {skip}", flush=True)
sys.path.insert(0, "postprocess/fixes")
from sub_check import check, merge_candidates  # noqa: E402
st = merge_candidates(f"{a.base}/candidate_pairs.tsv", new_pairs, str(o / "candidate_pairs.tsv"))
print(f"candidate_pairs.tsv: {st}", flush=True)
assert check(str(o / "matching_results.tsv"), str(o / "candidate_pairs.tsv")), "sub_check FAIL"
r = subprocess.run([sys.executable, "data/student_resource/utils/validate_submission.py", "--matching", str(o / "matching_results.tsv"), "--candidate",
                    str(o / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test", "--check-ids"], capture_output=True, text=True)
print("\n".join((r.stdout + r.stderr).strip().splitlines()[-2:]), f"\nvalidator exit code {r.returncode}")
