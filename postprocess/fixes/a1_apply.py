"""A1 test apply: add the VALID-gated new pairs to a shipped submission file (US + India only; France untouched).
Same acceptance as a1_gate.py on the test A1 stage-2 scores: new pair (listed in newpairs_test_<c>), p >= T, record argmax
over all its A1 pairs, record NOT already used anywhere in the base file, shipped rules (enrich grp / sup, BASE, decoy
tokens), one S1 per record. Writes <out>/matching_results.tsv (base rows in base order, new ids appended), runs the
official validator with --check-ids, and prints adds per country and per 1000 S1 in the VALID states (US ny, India ap+ts)
against VALID adds per 1000 VALID S1 from gate_summary.json (FLAG if test > 2x VALID).
Usage: POLARS_MAX_THREADS=3 python postprocess/fixes/a1_apply.py --t 0.9 --base output_bucket/stack_e5hyb2 \
         --out work/errfix/sub/stack_e5hyb2_a1 [--scored 'work_a1_out/stage2_scored_model_v10a1_{c}.parquet']
         [--newpairs-dir work/errfix/out/a1] [--gate-json work/errfix/out/a1/gate_summary.json] [--no-validate]
         [--cand-mode scored|adds]
candidate_pairs.tsv (audit 27 Sep, patch 01): <out>/candidate_pairs.tsv = base candidate_pairs.tsv + the new A1 pairs the
v10 A1 stage 2 scored (--cand-mode scored, default: the exact set the rescue model scores) or only the accepted adds
(--cand-mode adds: smallest set that keeps matched <= candidates). Checked with sub_check.py and the validator --candidate.
(~3-5 min incl. validator, < 4 GB)"""
import argparse
import gc
import json
import os
import subprocess
import warnings
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = os.environ.get("ER_ROOT", ".")
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/postprocess/fixes")
import polars as pl  # noqa: E402
from a1_gate import SAME, accept, candidates, load_new, texts_dir  # noqa: E402
from sub_check import check, merge_candidates  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, required=True)
ap.add_argument("--base", required=True, help="dir with the base matching_results.tsv")
ap.add_argument("--out", required=True)
ap.add_argument("--scored", default=f"{ROOT}/work_a1_out/stage2_scored_model_v10a1_{{c}}.parquet", help="path template with {c}")
ap.add_argument("--newpairs-dir", default=f"{ROOT}/work/errfix/out/a1")
ap.add_argument("--gate-json", default=f"{ROOT}/work/errfix/out/a1/gate_summary.json")
ap.add_argument("--no-validate", action="store_true")
ap.add_argument("--cand-mode", choices=["scored", "adds"], default="scored")
ap.add_argument("--countries", default="US,India", help="e.g. India for the Maharashtra / Delhi re-search")
a = ap.parse_args()

base = pl.read_csv(f"{a.base}/matching_results.tsv", separator="\t", schema={"s1": pl.String, "m": pl.String}, has_header=True)
base = base.with_columns(pl.col("m").fill_null(""))
used = base.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m").str.len_chars() > 0)["m"]
print(f"base {a.base}: S1 rows {base.height:,}, matched records {used.len():,} (unique {used.n_unique():,})", flush=True)
gate = json.load(open(a.gate_json)) if os.path.exists(a.gate_json) else None
gT = gate["T"].get(str(a.t)) if gate else None
if gate and gate.get("chosen") is not None and float(gate["chosen"]) != a.t:
    print(f"WARNING: --t {a.t} differs from the VALID gate's chosen T {gate['chosen']}", flush=True)
if gate and gate.get("chosen") is None:
    print("WARNING: the VALID gate FAILED (no T passed on both halves)", flush=True)

adds, flag, cand_new = [], False, {}
for c in a.countries.split(","):
    sc = pl.read_parquet(a.scored.format(c=c), columns=["s1_id", "m_id", "p"])
    new = load_new(a.newpairs_dir, "test", c)
    cand, info = candidates(sc, new, used, a.t, "test", c)
    if a.cand_mode == "scored":   # every new pair the A1 stage 2 scored is a candidate of the final system
        for s_, m_ in zip(*sc.join(new, on=["s1_id", "m_id"], how="semi").select("s1_id", "m_id").get_columns()):
            cand_new.setdefault(s_, []).append(m_)
    acc = accept(cand, a.t)
    del sc, cand
    gc.collect()
    st = pl.read_parquet(f"{texts_dir('test', c)}/s1.parquet", columns=["entity_id", "addr_state"])
    nS1, nsame = st.height, st.filter(pl.col("addr_state").is_in(SAME[c])).height
    acc = acc.join(st.rename({"entity_id": "s1_id", "addr_state": "state"}), on="s1_id", how="left")
    ns = acc.filter(pl.col("state").is_in(SAME[c])).height
    t1k, s1k = 1000 * acc.height / nS1, 1000 * ns / max(nsame, 1)
    v1k = gT[c]["per1k"] if gT else float("nan")
    ratio = s1k / v1k if gT and v1k > 0 else float("nan")
    bad = gT is not None and ratio > 2.0
    flag |= bad
    print(f"{c}: newpairs {new.height:,}, scored {info['new_scored']:,}, argmax & record free & p >= {a.t}: {info['cand']:,}, "
          f"accepted {acc.height:,} ({t1k:.2f}/1k test S1); same states {'+'.join(SAME[c])}: {ns:,} adds / {nsame:,} S1 = {s1k:.2f}/1k "
          f"vs VALID {v1k:.2f}/1k -> ratio {ratio:.2f}{'  FLAG (> 2x VALID)' if bad else ''}", flush=True)
    print(f"  {c} adds by group: {acc.group_by('grp').len().sort('len', descending=True).head(6).rows()}", flush=True)
    adds.append(acc.select("s1_id", "m_id", "p", "grp", "sup", "state", pl.lit(c).alias("country")))
    del st
adds = pl.concat(adds)
assert adds["m_id"].n_unique() == adds.height and not adds["m_id"].is_in(used.implode()).any(), "record reuse"
os.makedirs(a.out, exist_ok=True)
adds.write_parquet(f"{a.out}/a1_adds.parquet")
ag = adds.sort(["s1_id", "p"], descending=[False, True]).group_by("s1_id", maintain_order=True).agg(pl.col("m_id").str.join(",").alias("add"))
out = base.join(ag.rename({"s1_id": "s1"}), on="s1", how="left", maintain_order="left")
out = out.with_columns(pl.when(pl.col("add").is_null()).then(pl.col("m"))
                       .when(pl.col("m") == "").then(pl.col("add")).otherwise(pl.col("m") + "," + pl.col("add")).alias("m"))
miss = ag.join(base.rename({"s1": "s1_id"}), on="s1_id", how="anti").height
assert miss == 0, f"{miss} add S1 not in base file"
with open(f"{a.out}/matching_results.tsv", "w") as fh:
    fh.write("source1_entity_id\tmatched_entity_ids\n")
    for s, m in zip(out["s1"].to_list(), out["m"].to_list()):
        fh.write(f"{s}\t{m}\n")
tot = out.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m").str.len_chars() > 0)["m"]
print(f"wrote {a.out}/matching_results.tsv: rows {out.height:,}, matched {tot.len():,} (+{tot.len() - used.len():,}), unique {tot.n_unique():,}; "
      f"adds US {adds.filter(pl.col('country') == 'US').height:,}, India {adds.filter(pl.col('country') == 'India').height:,}, France 0", flush=True)
print("DENSITY CHECK: " + ("FLAG, test same-state adds per 1k S1 exceed 2x VALID; hold the file" if flag else
                           ("ok (<= 2x VALID)" if gT else "no VALID reference (run a1_gate.py first)")), flush=True)
if a.cand_mode == "adds":
    for s_, m_ in zip(adds["s1_id"].to_list(), adds["m_id"].to_list()):
        cand_new.setdefault(s_, []).append(m_)
del base, out, adds, ag, used, tot
gc.collect()
bc = f"{a.base}/candidate_pairs.tsv"
assert os.path.exists(bc), f"{bc} missing: the A1 file needs the base candidate set"
st = merge_candidates(bc, cand_new, f"{a.out}/candidate_pairs.tsv")
print(f"candidate_pairs.tsv ({a.cand_mode}): {st}; +{st['added_pairs'] / max(st['rows'], 1):.3f} candidates per test S1", flush=True)
del cand_new
gc.collect()
ok = check(f"{a.out}/matching_results.tsv", f"{a.out}/candidate_pairs.tsv")
assert ok, "sub_check FAIL"
if not a.no_validate:
    r = subprocess.run([sys.executable, f"{ROOT}/data/student_resource/utils/validate_submission.py", "--matching", f"{a.out}/matching_results.tsv",
                        "--candidate", f"{a.out}/candidate_pairs.tsv", "--test-dir", f"{ROOT}/data/student_resource/dataset/test", "--check-ids"], capture_output=True, text=True)
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-12:]))
    print(f"validator exit code {r.returncode}")
