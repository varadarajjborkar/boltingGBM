"""Phase 1: score the TEST split and write the two submission files.

Per country: full-universe context -> chunked features -> p1 (fold-ensemble, as in training) -> stage-2 context over
all pairs -> stage-2 on pairs with p1 >= P1_MIN (this set IS candidate_pairs.tsv) -> decisions -> matching_results.tsv.
Checkpoints: per-country p1 table and stage-2 scores under work/p1/test/<country>/; re-run skips finished parts.
Usage: python p1_score.py --stage feats   (optional, no model needed; can run while training)
       caffeinate -i -s -m python p1_score.py --version v1 --decision threshold --t 0.75   (score + write)
       python p1_score.py --stage score  /  --stage write --version v1 --t 0.75   (the two halves separately)
       python p1_score.py --stage rescore --alt work_v4/p1/model_v10_decoy   (another stage 2 on the saved stage-2 inputs)
"""
import argparse
import json

import numpy as np
import polars as pl

from p1_features import country_context, iter_feature_chunks, load_country
from p1_train import M, P1_MIN, f32, load_predictor, model_file, predict_p1
from pathlib import Path

from utils import DATA, ROOT, WORK, announce_pid, done, log, resource_guard, rss_gb, timed, write_parquet_atomic

BUCKET = ROOT / "output_bucket"   # one versioned folder per full-data submission
SC = f"stage2_scored_{M.name}.parquet"   # per-model score file, so v1/v2 models never overwrite or skip each other


def feats_country(c):
    """Model-free half of scoring: build and save every test pair's features. It does not need the trained model,
    so it can run on a second machine while training is still going. score_country() then only predicts."""
    from p1_features import save_features
    fdir = WORK / "p1" / "test" / c / "feats_all"
    if done(fdir / "DONE"):
        log(f"skip {c} (features saved)"); return
    resource_guard(f"test feats {c}")
    with timed(f"test features {c}"):
        save_features("test", c, "all", None)
    (fdir / "DONE").touch()


def score_country(c):
    from stack import stage2_features
    d = WORK / "p1" / "test" / c
    if done(d / SC):
        log(f"skip {c} (scored)"); return
    resource_guard(f"score {c}")
    cols = json.loads((M / "feature_cols.json").read_text())
    allc = json.loads((M / "stage2_cols.json").read_text())
    fdir = d / "feats_all"
    if (fdir / "DONE").exists():          # features precomputed by feats_country(): read them back
        parts = sorted(fdir.glob("part_*.parquet"))
        n_s1 = pl.scan_parquet(d / "s1.parquet").select(pl.len()).collect().item()
        chunks = ((b, len(parts), pl.read_parquet(p)) for b, p in enumerate(parts))
        what = f"{len(parts)} precomputed chunks"
    else:
        s1, pool, cands = load_country("test", c)
        c_ctx, idf, freq = country_context(s1, pool, cands)
        n_s1 = s1.height
        chunks = iter_feature_chunks(s1, pool, c_ctx, idf, freq)
        what = f"{cands.height} pairs"
    kept_dir = d / f"kept_feats_{M.name}"
    kept_dir.mkdir(exist_ok=True)
    p1_parts = []
    with timed(f"pass 1 {c}: features + p1 for {what}"):
        for b, n, f in chunks:
            p1 = predict_p1(f32(f, cols))
            t = f.select("s1_id", "m_id").with_columns(pl.Series("p1", p1.astype(np.float32)))
            p1_parts.append(t)
            keep = f.with_columns(pl.Series("p1", p1)).filter(pl.col("p1") >= P1_MIN).drop("p1")
            write_parquet_atomic(keep, kept_dir / f"part_{b:03d}.parquet")
            if b % 5 == 0:
                log(f"  chunk {b + 1}/{n}: kept {keep.height}/{f.height}, RSS {rss_gb():.1f} GB")
    p1t = pl.concat(p1_parts)
    write_parquet_atomic(p1t, d / f"p1_{M.name}.parquet")
    s2 = stage2_features(p1t).filter(pl.col("p1") >= P1_MIN)
    if any(c.startswith("s3_sib") for c in allc):     # model trained with sibling evidence
        from stack import pool_texts, sibling_features
        s2 = s2.join(sibling_features(p1t, pool_texts([d / "pool.parquet"]), P1_MIN), on=["s1_id", "m_id"], how="left")
    kept = pl.read_parquet(str(kept_dir / "part_*.parquet"))
    x = s2.join(kept, on=["s1_id", "m_id"], how="inner")
    m2 = load_predictor(model_file("stage2"), fast=False)   # ~10% of pairs: LightGBM is quicker than compiling
    x = x.select("s1_id", "m_id").with_columns(pl.Series("p", m2(f32(x, allc))))
    write_parquet_atomic(x, d / SC)
    # kept_dir stays: rescore_country() applies another stage 2 (e.g. the decoy-trained one) without a stage-1 pass
    log(f"{c}: {x.height} stage-2 pairs scored ({x.height / n_s1:.2f} per S1)")


def rescore_country(c, alt):
    """Stage 2 of model folder `alt` (same stage-2 columns as M) on M's saved p1 table and kept features."""
    from stack import stage2_features
    d = WORK / "p1" / "test" / c
    out = d / f"stage2_scored_{alt.name}.parquet"
    if done(out):
        log(f"skip {c} (rescored)"); return
    allc = json.loads((M / "stage2_cols.json").read_text())
    p1t = pl.read_parquet(d / f"p1_{M.name}.parquet")
    s2 = stage2_features(p1t).filter(pl.col("p1") >= P1_MIN)
    if any(c.startswith("s3_sib") for c in allc):
        from stack import pool_texts, sibling_features
        s2 = s2.join(sibling_features(p1t, pool_texts([d / "pool.parquet"]), P1_MIN), on=["s1_id", "m_id"], how="left")
    x = s2.join(pl.read_parquet(str(d / f"kept_feats_{M.name}" / "part_*.parquet")), on=["s1_id", "m_id"], how="inner")
    m2 = load_predictor(model_file("stage2", alt), fast=False)
    x = x.select("s1_id", "m_id").with_columns(pl.Series("p", m2(f32(x, allc))))
    write_parquet_atomic(x, out)
    log(f"{c}: {x.height} stage-2 pairs rescored with {alt.name}")


def write_outputs(decision, t, mu, r, version, note, t_country=None):
    from decide import by_expected_f_entity, by_threshold, one_s1_per_record
    s1_all = pl.read_parquet(WORK / "parquet" / "test_source1.parquet", columns=["entity_id"])
    cs = sorted(p.name for p in (WORK / "p1" / "test").iterdir() if (p / SC).exists())
    if cs != ["France", "India", "US"]:
        raise SystemExit(f"scores for model {M.name} missing: have {cs}")
    sc = pl.concat([pl.read_parquet(WORK / "p1" / "test" / c / SC) for c in cs])
    if r != 1.0:   # prior-shift odds scaling
        sc = sc.with_columns((pl.col("p") * r / (pl.col("p") * r + 1 - pl.col("p"))).alias("p"))
    sc1 = one_s1_per_record(sc)
    if decision == "threshold" and t_country:     # per-country thresholds, e.g. {"US": 0.80, "India": 0.83}
        cty = pl.read_parquet(WORK / "parquet" / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
        thr = pl.DataFrame({"country": list(t_country), "tc": [float(v) for v in t_country.values()]})
        pred = (sc1.join(cty, on="s1_id").join(thr, on="country", how="left").with_columns(pl.col("tc").fill_null(t))
                   .filter(pl.col("p") >= pl.col("tc")).select("s1_id", "m_id"))
    elif decision == "threshold":
        pred = by_threshold(sc1, t)
    else:
        pred = by_expected_f_entity(sc1, pl.read_parquet(M / "test_p_match.parquet"), mu)
    OUT = BUCKET / version
    OUT.mkdir(parents=True, exist_ok=True)

    def write(pairs, colname, path):
        g = pairs.group_by("s1_id").agg(pl.col("m_id").unique().sort().str.join(",").alias(colname))
        full = s1_all.rename({"entity_id": "source1_entity_id"}).join(
            g.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
        tmp = path.with_suffix(".tsv.tmp")
        full.write_csv(tmp, separator="\t", quote_style="never")
        tmp.replace(path)
        log(f"wrote {path} ({full.height} rows, {(full[colname] != '').sum()} non-empty)")

    write(sc.select("s1_id", "m_id"), "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    write(pred, "matched_entity_ids", OUT / "matching_results.tsv")
    finalize_bucket(OUT, version, dict(decision=decision, t=t, t_country=t_country, mu=mu, r=r, note=note, pred_pairs=pred.height,
                                       candidate_pairs=sc.height))


def finalize_bucket(OUT, version, info):
    """Official validator, code zip for the portal, notes, and the version-history table."""
    import subprocess
    import sys
    import time
    import zipfile
    val = subprocess.run([sys.executable, str(ROOT / "data" / "student_resource" / "utils" / "validate_submission.py"),
                          "--matching", str(OUT / "matching_results.tsv"), "--candidate", str(OUT / "candidate_pairs.tsv"),
                          "--test-dir", str(DATA / "test")], capture_output=True, text=True)
    code_dir = ROOT / "code" / "business_entity_resolution"
    with zipfile.ZipFile(OUT / "amazingRIVER_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(code_dir.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                z.write(f, Path("code") / "business_entity_resolution" / f.relative_to(code_dir))
    rep = {}
    vr = M / "valid_report.json"
    if vr.exists():
        rep = json.loads(vr.read_text()).get("best", {})
    notes = [f"# {version}", "", f"Created: {time.strftime('%Y-%m-%d %H:%M')} IST", "",
             "Upload to the portal: `matching_results.tsv` only (portal changed on 25 Sep: one TSV upload box). "
             "`candidate_pairs.tsv` and `amazingRIVER_code.zip` are kept for the final package.", "",
             "## Settings", "```", json.dumps(info, indent=1), "```", "", "## Full-scale validation (train, held-out states)",
             "```", json.dumps(rep, indent=1), "```", "", "## Official validator", "```", val.stdout.strip()[-1500:], "```"]
    (OUT / "NOTES.md").write_text("\n".join(notes))
    hist = BUCKET / "SUBMISSIONS.md"
    if not hist.exists():
        hist.write_text("# Submission version history\n\n| Version | Created | Validation macro F0.5 | Validator | Uploaded? | Public LB |\n|---|---|---|---|---|---|\n")
    full = json.loads(vr.read_text()) if vr.exists() else {}
    at = full.get("threshold", {}).get(str(info["t"])) if info["decision"] == "threshold" else full.get("entity", {}).get(str(info["mu"]))
    vscore = f'{at["macro"]} (at {"t" if info["decision"] == "threshold" else "mu"}={info["t"] if info["decision"] == "threshold" else info["mu"]})' if at else "n/a"
    status = "PASS" if val.returncode == 0 else "FAIL"
    with open(hist, "a") as fh:
        fh.write(f"| {version} | {time.strftime('%d %b %H:%M')} | {vscore} | {status} | not yet | |\n")
    log(f"bucket {OUT}: validator {status}, code zip {(OUT / 'amazingRIVER_code.zip').stat().st_size // 1024} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="")
    ap.add_argument("--decision", default="threshold", choices=["threshold", "entity"])
    ap.add_argument("--t", type=float, default=0.75)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--r", type=float, default=1.0, help="odds scaling for prior shift (1 = none)")
    ap.add_argument("--only_write", action="store_true")
    ap.add_argument("--stage", default="all", choices=["feats", "score", "write", "all", "rescore"],
                    help="feats = precompute test features (no model needed); score = predict only; "
                         "write = submission files from saved scores; all = score + write")
    ap.add_argument("--alt", default="", help="rescore: model folder whose stage2.txt is applied")
    ap.add_argument("--version", default="", help="folder name under output_bucket/, e.g. v1 (needed for score)")
    ap.add_argument("--note", default="", help="what changed in this version")
    ap.add_argument("--t_country", default="", help='per-country thresholds, e.g. "US:0.80,India:0.83,France:0.85"')
    a = ap.parse_args()
    announce_pid("p1_score")
    countries = a.countries.split(",") if a.countries else sorted(p.name for p in (WORK / "p1" / "test").iterdir() if p.is_dir())
    if a.stage == "feats":
        for c in countries:
            feats_country(c)
        return
    if a.stage == "rescore":      # --alt DIR: its stage2.txt on this model's (ER_MODEL_DIR) saved p1 + kept features
        for c in countries:
            rescore_country(c, Path(a.alt))
        return
    if a.stage in ("score", "all") and not a.only_write:
        for c in countries:
            score_country(c)
    if a.stage == "score":
        return
    if not a.version:
        raise SystemExit("--version is required to write submission files")
    tc = {k: float(v) for k, v in (x.split(":") for x in a.t_country.split(","))} if a.t_country else None
    write_outputs(a.decision, a.t, a.mu, a.r, a.version, a.note, tc)


if __name__ == "__main__":
    main()
