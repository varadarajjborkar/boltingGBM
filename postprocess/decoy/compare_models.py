"""Compare models on test with pooled pi_eval, each model's own blocking-miss rate, and the decoy-token group (the LB proxy
that forecast v8_final within 0.0002 and credits blocking gains).

Usage: python postprocess/decoy/compare_models.py --models model_v10,model_v10_decoy \
           --valid work_v4/p1/model_v10/valid_scored.parquet,work_v4/p1/model_v10_decoy/scored_valid_new.parquet
Steps per model (skipped when outputs exist): pi_eval cache (postprocess/decoy/pi_eval.py --cache), decoy-token flags
(postprocess/decoy/flag_rule_eval.py --stage flags). Then the variants below per model.
Inputs: <valid> scores (s1_id, m_id, p), work_v4/p1/test/<c>/stage2_scored_<model>.parquet, texts.
Outputs: prints a table (VALID, EF per country, LB_raw, forecast = LB_raw - 0.0045); writes work/director/compare_<models>.parquet.
Runs as separate processes per step; each under about 3 GB.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, "postprocess/decoy")
import dlib  # noqa: E402
import pi_eval  # noqa: E402
import polars as pl  # noqa: E402

R = dlib.ROOT
PY = sys.executable
OFFSET = 0.0045
BASE = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98}
VARIANTS = {
    "rules+tok t0.70 (current)": {"t": 0.70, "tg": {**BASE, "dtok": 1.01}},
    "rules+tok t0.75": {"t": 0.75, "tg": {**BASE, "dtok": 1.01}},
    "rules+tok t0.65": {"t": 0.65, "tg": {**BASE, "dtok": 1.01}},
    "no shift_12 rules": {"t": 0.70, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "missing|corr": 0.98, "dtok": 1.01}},
    "no shift_12, no missing|corr": {"t": 0.70, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "dtok": 1.01}},
    "only pure+ms+tok": {"t": 0.70, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "dtok": 1.01}},
    "no rules at all": {"t": 0.70},
}


def prepare(m, valid):
    if not (os.path.exists(f"{R}/work/director/pi_cache_{m}_valid.parquet") and os.path.exists(f"{R}/work/director/pi_cache_{m}_test.parquet")):
        subprocess.run([PY, f"{R}/postprocess/decoy/pi_eval.py", "--valid", valid, "--test-root", "work_v4", "--model", m, "--t", "0.70",
                        "--variants", f"{R}/work/director/v9_min.json", "--cache", "--pool", "--out", "_min"], check=True)
    if not os.path.exists(f"{R}/work/director/flags_{m}.parquet"):
        subprocess.run([PY, f"{R}/postprocess/decoy/flag_rule_eval.py", "--model", m, "--flag", "dtok", "--stage", "flags", "--valid", valid], check=True)


def evaluate(m):
    v = pl.read_parquet(f"{R}/work/director/pi_cache_{m}_valid.parquet", columns=["s1_id", "m_id", "y", "p", "am", "country", "state", "grp", "sup"])
    t = pl.read_parquet(f"{R}/work/director/pi_cache_{m}_test.parquet", columns=["s1_id", "m_id", "p", "am", "country", "state", "grp", "sup"])
    f = pl.read_parquet(f"{R}/work/director/flags_{m}.parquet", columns=["s1_id", "m_id", "dtok", "split"])
    v = v.join(f.filter(pl.col("split") == "valid").drop("split"), on=["s1_id", "m_id"], how="left").with_columns(pl.col("dtok").fill_null(False))
    t = t.join(f.filter(pl.col("split") == "test").drop("split"), on=["s1_id", "m_id"], how="left").with_columns(pl.col("dtok").fill_null(False))
    del f
    rel = pl.when(pl.col("dtok") & ~pl.col("grp").fill_null("").str.starts_with("shift")).then(pl.lit("dtok")).otherwise(pl.col("grp"))
    v, t = v.with_columns(rel.alias("grp")).drop("dtok", "m_id"), t.with_columns(rel.alias("grp")).drop("dtok", "m_id")
    NVS = {"US": 102314, "India": 73301}
    um = v.filter((pl.col("y") == 1) & pl.col("p").is_null()).group_by("country").len()
    pi_eval.LAM = {c: n / NVS[c] for c, n in um.iter_rows()}
    rc, fc, band, lab = pi_eval.cells(v, t, pool=True)
    tq = {}
    for c in ["US", "India", "France"]:
        for src in ([c] if c != "France" else ["US", "India"]):
            tq[(c, src)] = pi_eval.assign_q(t.filter(pl.col("country") == c), rc, fc, band, lab, src)
    rows = []
    for nm, spec in VARIANTS.items():
        row = {"model": m, "variant": nm, "lam_IN": pi_eval.LAM.get("India")}
        mm = dlib.macro(dlib.entity_scores(v.with_columns(pi_eval.decide(v, spec).alias("dec")), "dec"))
        row.update({"VALID": mm["all"], "VALID_US": mm["US"], "VALID_IN": mm["India"]})
        for (c, src), d in tq.items():
            row[f"EF_{c}_{src}"] = pi_eval.dlib_expected(d.with_columns(pi_eval.decide(d, spec).alias("dec")), pi_eval.LAM[src], pi_eval.NTEST[c], 4)
        row["LB_raw"] = (pi_eval.W["US"] * row["EF_US_US"] + pi_eval.W["India"] * row["EF_India_India"]
                         + pi_eval.W["France"] * 0.5 * (row["EF_France_US"] + row["EF_France_India"]))
        row["forecast"] = row["LB_raw"] - OFFSET
        rows.append(row)
        print(m, nm, round(row["LB_raw"], 5), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--valid", required=True, help="comma list, one VALID scores parquet per model")
    ap.add_argument("--stage", default="all", choices=["all", "eval"])
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    ms, vs = a.models.split(","), a.valid.split(",")
    if a.stage == "eval":
        rows = evaluate(a.model)
        pl.DataFrame(rows).write_parquet(f"{R}/work/director/compare_part_{a.model}.parquet")
        return
    for m, v in zip(ms, vs):
        prepare(m, v)
        subprocess.run([PY, __file__, "--models", a.models, "--valid", a.valid, "--stage", "eval", "--model", m], check=True)
    r = pl.concat([pl.read_parquet(f"{R}/work/director/compare_part_{m}.parquet") for m in ms])
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_cols(12)
    print(r.select("model", "variant", "VALID", "EF_US_US", "EF_India_India", "EF_France_US", "LB_raw", "forecast").with_columns(pl.col(pl.Float64).round(5)))
    r.write_parquet(f"{R}/work/director/compare_{'_'.join(ms)}.parquet")


if __name__ == "__main__":
    main()
