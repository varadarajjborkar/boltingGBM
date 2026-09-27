"""India shift-rule relaxation: keep India pairs in groups shift_pure / shift_ms_only (dropped by the shipped rules) when
stage-3 p >= PMIN, the record's best S1 is this S1 (am) and no decoy token. Evidence (postprocess/fixes/shift_rule_density.py):
in the same states (ap+ts) test density / VALID density = 0.82 (pure) and 0.86 (ms_only) at p >= 0.98, VALID precision ~0.99,
so no test decoy excess in India. US / France unchanged (US shift_pure is decoy-enriched, ratio 1.4-5.5).
--split valid: VALID gain by S1-hash half. --split test: base submission + restored India pairs -> --out, validator.
Usage: python postprocess/fixes/india_shift_restore.py --split valid [--pmin 0.98]
       python postprocess/fixes/india_shift_restore.py --split test --base output_bucket/stack_ce_fr --out work/errfix/sub/stack_ce_fr_ins"""
import argparse
import subprocess
import sys
from pathlib import Path
import polars as pl
sys.path.insert(0, "postprocess/decoy")
import pi_eval  # noqa: E402
from compare_models import BASE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["valid", "test"], required=True)
ap.add_argument("--model", default="model_v10e_v10_s3")
ap.add_argument("--pmin", type=float, default=0.98)
ap.add_argument("--groups", default="shift_pure,shift_ms_only")
ap.add_argument("--base", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()
G = a.groups.split(",")
cols = ["s1_id", "m_id", "p", "am", "grp", "sup", "country"] + (["y"] if a.split == "valid" else [])
d = pl.read_parquet(f"work/director/pi_cache_{a.model}_{a.split}.parquet", columns=cols)
fl = pl.read_parquet(f"work/director/flags_{a.model}.parquet", columns=["s1_id", "m_id", "dtok", "split"]).filter(pl.col("split") == a.split).drop("split")
d = d.join(fl, on=["s1_id", "m_id"], how="left").with_columns(pl.col("dtok").fill_null(False))
d = d.with_columns(pl.when(pl.col("dtok") & ~pl.col("grp").fill_null("").str.starts_with("shift")).then(pl.lit("dtok")).otherwise(pl.col("grp")).alias("grp"))
d = d.with_columns(pi_eval.decide(d, {"t": 0.75, "tg": {**BASE, "dtok": 1.01}}).alias("dec"))
restore = (pl.col("country") == "India") & pl.col("grp").is_in(G) & (pl.col("p") >= a.pmin) & pl.col("am") & ~pl.col("dtok") & ~pl.col("dec")
d = d.with_columns(restore.fill_null(False).alias("add"))
print(f"{a.split}: restored India pairs {int(d['add'].sum())} (" + ", ".join(f"{g} {int(d.filter(pl.col('add') & (pl.col('grp') == g)).height)}" for g in G) + ")")
if a.split == "valid":
    N = {"US": 102314, "India": 73301}
    y = pl.col("y") == 1
    ad = d.filter(pl.col("add"))
    print(f"precision of restored VALID pairs {ad['y'].mean():.4f}")
    def f05(tp, fp, fn):
        return pl.when((tp + fp + fn) == 0).then(1.0).otherwise(1.25 * tp / (1.25 * tp + 0.25 * fn + fp))
    def macro(dec):
        e = d.with_columns(dec.alias("D")).group_by("s1_id", "country").agg((pl.col("D") & y).sum().alias("tp"), (pl.col("D") & ~y).sum().alias("fp"),
                                                                            (~pl.col("D") & y).sum().alias("fn"))
        e = e.with_columns(f05(pl.col("tp"), pl.col("fp"), pl.col("fn")).alias("F"), (pl.col("s1_id").hash(13) % 2).alias("h"))
        tot = sum(N.values())
        return {"all": 1 - (1 - e["F"]).sum() / tot, "h0": 1 - (1 - e.filter(pl.col("h") == 0)["F"]).sum() / (tot / 2),
                "h1": 1 - (1 - e.filter(pl.col("h") == 1)["F"]).sum() / (tot / 2), "India": 1 - (1 - e.filter(pl.col("country") == "India")["F"]).sum() / N["India"]}
    b0, b1 = macro(pl.col("dec")), macro(pl.col("dec") | pl.col("add"))
    print("VALID gain: " + " ".join(f"{k} {b1[k] - b0[k]:+.5f}" for k in b0) + f" (base all {b0['all']:.5f})")
else:
    rd = pl.read_csv(Path(a.base) / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    used = set(rd.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")["matched_entity_ids"].to_list())
    ad = d.filter(pl.col("add") & ~pl.col("m_id").is_in(list(used))).unique("m_id")
    extra = ad.group_by("s1_id").agg(pl.col("m_id"))
    em = dict(zip(extra["s1_id"].to_list(), extra["m_id"].to_list()))
    new = [",".join([x for x in v.split(",") if x] + em.get(s, [])) for s, v in zip(rd["source1_entity_id"].to_list(), rd["matched_entity_ids"].to_list())]
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    rd.with_columns(pl.Series("matched_entity_ids", new)).write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
    import shutil
    if (Path(a.base) / "candidate_pairs.tsv").exists():
        shutil.copy(Path(a.base) / "candidate_pairs.tsv", o / "candidate_pairs.tsv")
    (o / "NOTES.md").write_text(f"# {o.name}\n\n{a.base} + {ad.height} India pairs restored (groups {G}, p >= {a.pmin}, am, no decoy token; "
                                f"postprocess/fixes/india_shift_restore.py). US and France unchanged.\n")
    print(f"added {ad.height} pairs to {extra.height} S1 (skipped {int(d['add'].sum()) - ad.height} whose record is already used)")
    r = subprocess.run([".venv/bin/python", "data/student_resource/utils/validate_submission.py", "--matching", str(o / "matching_results.tsv"),
                        "--candidate", str(o / "candidate_pairs.tsv"), "--test-dir", "data/student_resource/dataset/test"], capture_output=True, text=True)
    print("validator", "PASS" if r.returncode == 0 else "FAIL")
