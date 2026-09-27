"""Apply an msrit overlay (adds / removals as s1_id, m_id, action, src, country) onto YOUR current submission, safely:
  - removal: only if that exact pair is in your file
  - add: only if the record is assigned NOWHERE in your file (after removals), one S1 per record, and the S1 exists
  - every added pair is also appended to candidate_pairs (matches stay a subset of candidates)
  - --only-src / --skip-src / --countries restrict which overlay rows are used
Writes <out>/matching_results.tsv + candidate_pairs.tsv and prints what was applied / skipped per src and country.
Then run models/cross_encoders/final_check.py on the output.
Usage: python models/cross_encoders/apply_overlay.py <your matching_results.tsv> <your candidate_pairs.tsv> <overlay.parquet> <out dir>
       [--only-src stage4_cells,tie_q75] [--skip-src tie_q75] [--countries US,India] [--block removed1.parquet,removed2.parquet]"""
import argparse
from collections import Counter
from pathlib import Path
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("matching"); ap.add_argument("candidates"); ap.add_argument("overlay"); ap.add_argument("out")
ap.add_argument("--only-src", default=""); ap.add_argument("--skip-src", default=""); ap.add_argument("--countries", default="")
ap.add_argument("--block", default="", help="comma list of parquet files with an m_id column (your removal lists): never re-add those records")
a = ap.parse_args()
ov = pl.read_parquet(a.overlay)
if a.only_src: ov = ov.filter(pl.col("src").is_in(a.only_src.split(",")))
if a.skip_src: ov = ov.filter(~pl.col("src").is_in(a.skip_src.split(",")))
if a.countries: ov = ov.filter(pl.col("country").is_in(a.countries.split(",")))


def read(path):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return d.columns, {s: [x for x in ids.split(",") if x] for s, ids in zip(d[d.columns[0]].to_list(), d[d.columns[1]].to_list())}


mh, M = read(a.matching); ch, Cd = read(a.candidates)
owner = {x: s for s, ids in M.items() for x in ids}
block = set()
for f in [x for x in a.block.split(",") if x]: block |= set(pl.read_parquet(f)["m_id"].to_list())
st = Counter()
for s, m, src, c in zip(*[ov.filter(pl.col("action") == "remove")[k].to_list() for k in ("s1_id", "m_id", "src", "country")]):
    if s in M and m in M[s]:
        M[s].remove(m); owner.pop(m, None); st[(src, c, "remove", "applied")] += 1
    else:
        st[(src, c, "remove", "skipped_not_present")] += 1
for s, m, src, c in zip(*[ov.filter(pl.col("action") == "add")[k].to_list() for k in ("s1_id", "m_id", "src", "country")]):
    if s not in M:
        st[(src, c, "add", "skipped_unknown_s1")] += 1; continue
    if m in owner:
        st[(src, c, "add", "skipped_record_assigned")] += 1; continue
    if m in block:
        st[(src, c, "add", "skipped_blocked_record")] += 1; continue
    M[s].append(m); owner[m] = s; st[(src, c, "add", "applied")] += 1
    if m not in Cd.setdefault(s, []): Cd[s].append(m)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("\t".join(mh) + "\n")
    for s, ids in M.items(): f.write(f"{s}\t{','.join(ids)}\n")
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("\t".join(ch) + "\n")
    for s, ids in Cd.items(): f.write(f"{s}\t{','.join(ids)}\n")
for k in sorted(st): print(f"   {k[0]:14s} {k[1]:7s} {k[2]:7s} {k[3]:26s} {st[k]}")
print(f"wrote {out}/matching_results.tsv and candidate_pairs.tsv; now run models/cross_encoders/final_check.py on them")
