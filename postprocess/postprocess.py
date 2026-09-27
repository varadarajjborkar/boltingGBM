"""Submission from saved test scores with house-number group thresholds (tuned in review, 25 Sep).

Scores -> one S1 per record -> per-pair threshold: the global --t, or a group threshold for the pair's house-number
class from postprocess/checks/fingerprint.num_rel (shift_pure, shift_ms_only, shift_12, ...; a value > 1 drops the class).
A key "grp|sup" overrides "grp" for one support class: corr = the record's new house numbers are
also carried by another scored record of the same S1 (clustered look-alikes), lone = they are not, none = no new number.
Writes output_bucket/<dst>/ (matching_results.tsv, candidate_pairs.tsv, NOTES.md), runs the official validator and
adds a row to output_bucket/SUBMISSIONS.md.
Usage: python postprocess/postprocess.py --scores work_fr --model model_v2b --t 0.75 \
           --tg shift_pure=1.01,shift_ms_only=1.01,shift_12=0.98 --dst v4_rule2 [--check v4]
       --tg shift_pure=0.85,shift_pure|corr=1.01,shift_12=0.95,shift_12|corr=0.98   (support-aware)
--drop-vocab file.json drops every pair whose record name adds a listed token (name_core tokens of the record minus
those of the S1; director decoy vocabulary, data/decoy_vocab_validated.json).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "postprocess" / "checks"))
from decide import one_s1_per_record  # noqa: E402
from fingerprint import num_rel  # noqa: E402

CS = ["France", "India", "US"]
NUM = re.compile(r"\d+")


def nset(t):
    return {str(int(v)) for v in NUM.findall(t or "") if len(v) <= 7}


def _ed_le2(a, b, k=2):
    """edit distance <= k (banded)."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > k:
            return False
        prev = cur
    return prev[-1] <= k


def decoy_flag(na, nb, voc):
    """record name adds a decoy token that is not a near-duplicate of an S1 word (typo, plural, two S1 words joined)."""
    sa = [w for w in (na or "").split(" ") if w]
    joined = {u + v for u in sa for v in sa if u != v}
    for w in set(w for w in (nb or "").split(" ") if w) - set(sa):
        if w in voc and w not in joined and not any(_ed_le2(w, u, 2 if min(len(w), len(u)) >= 5 else 1) for u in sa):
            return True
    return False


def support(x, sc, texts):
    """corr / lone / none per candidate pair, as in postprocess/decoy/pi_eval.enrich."""
    others = {}
    for a, b in zip(*sc.join(x.select("s1_id").unique(), on="s1_id", how="semi").select("s1_id", "m_id").get_columns()):
        others.setdefault(a, []).append(b)
    nb, out = {}, []
    for a, b, ta, tb in zip(x["s1_id"].to_list(), x["m_id"].to_list(), x["sa"].to_list(), x["sb"].to_list()):
        new = nset(tb) - nset(ta)
        if not new:
            out.append("none"); continue
        k = "lone"
        for bb in others.get(a, []):
            if bb != b:
                if bb not in nb:
                    nb[bb] = nset(texts.get(bb, ""))
                if new <= nb[bb]:
                    k = "corr"; break
        out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="work folder holding p1/test/<country>/stage2_scored_<model>.parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--t", type=float, default=0.75)
    ap.add_argument("--tg", default="", help="group thresholds, e.g. shift_pure=1.01,shift_12=0.98")
    ap.add_argument("--texts", default="work_v4", help="work folder with test s1/pool tables (addr_text)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--check", default="", help="existing version: report the plain-threshold difference against it")
    ap.add_argument("--note", default="")
    ap.add_argument("--drop-vocab", default="", help="JSON list of decoy name tokens: drop pairs whose record adds one")
    a = ap.parse_args()
    tg = {k: float(v) for k, v in (x.split("=") for x in a.tg.split(","))} if a.tg else {}
    sc = pl.concat([pl.read_parquet(ROOT / a.scores / "p1" / "test" / c / f"stage2_scored_{a.model}.parquet").with_columns(
        pl.lit(c).alias("country")) for c in CS])
    sc1 = one_s1_per_record(sc)
    tmin = min([a.t] + list(tg.values()))
    cand = sc1.filter(pl.col("p") >= tmin)
    T = ROOT / a.texts / "p1" / "test"
    s1 = pl.concat([pl.read_parquet(T / c / "s1.parquet", columns=["entity_id", "addr_text", "name_core"]) for c in CS])
    pool = pl.concat([pl.read_parquet(T / c / "pool.parquet", columns=["entity_id", "addr_text", "name_core"]) for c in CS])
    x = (cand.join(s1.rename({"entity_id": "s1_id", "addr_text": "sa", "name_core": "na"}), on="s1_id", how="left")
             .join(pool.rename({"entity_id": "m_id", "addr_text": "sb", "name_core": "nb"}), on="m_id", how="left"))
    pool = pool.select("entity_id", "addr_text")
    assert x.height == cand.height and x["sa"].null_count() == 0 and x["sb"].null_count() == 0, "text lookup failed"
    x = x.with_columns(pl.Series("grp", [num_rel(u, v) for u, v in zip(x["sa"].to_list(), x["sb"].to_list())]))
    if any("|" in k for k in tg):
        need = sc.join(x.select("s1_id").unique(), on="s1_id", how="semi").select("m_id").unique()
        texts = dict(zip(*pool.rename({"entity_id": "m_id"}).join(need, on="m_id", how="semi").get_columns()))
        x = x.with_columns(pl.Series("sup", support(x, sc, texts)))
    else:
        x = x.with_columns(pl.lit("-").alias("sup"))
    thr = pl.lit(a.t)
    for k, v in sorted(tg.items(), key=lambda kv: "|" in kv[0]):   # grp keys first, grp|sup keys win
        g, _, sp = k.partition("|")
        thr = pl.when((pl.col("grp") == g) & ((pl.col("sup") == sp) if sp else pl.lit(True))).then(pl.lit(v)).otherwise(thr)
    if a.drop_vocab:
        voc = json.load(open(a.drop_vocab))
        voc = set(voc)
        x = x.with_columns(pl.Series("decoy_tok", [decoy_flag(u, v, voc) for u, v in zip(x["na"].to_list(), x["nb"].to_list())]))
        thr = pl.when(pl.col("decoy_tok")).then(pl.lit(1.01)).otherwise(thr)
        x = x.with_columns(pl.when(pl.col("decoy_tok")).then(pl.lit("decoy_tok")).otherwise(pl.col("sup")).alias("sup"))
    x = x.with_columns(thr.alias("tgrp"))
    x = x.with_columns((pl.col("p") >= pl.col("tgrp")).alias("keep"), (pl.col("p") >= a.t).alias("plain"))
    pred = x.filter("keep").select("s1_id", "m_id")
    tab = (x.group_by("country", "grp", "sup").agg(pl.col("plain").sum().alias("plain_pred"), pl.col("keep").sum().alias("kept"))
             .filter(pl.col("plain_pred") != pl.col("kept")).sort("country", "grp", "sup"))
    if a.check:
        old = (pl.read_csv(ROOT / "output_bucket" / a.check / "matching_results.tsv", separator="\t",
                           schema_overrides={"matched_entity_ids": pl.Utf8})
                 .filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
                 .explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"}))
        plain = x.filter("plain").select("s1_id", "m_id")
        print(f"check vs {a.check}: plain {plain.height}, {a.check} {old.height}, only-new {plain.join(old, on=['s1_id', 'm_id'], how='anti').height}, "
              f"only-old {old.join(plain, on=['s1_id', 'm_id'], how='anti').height}")
    s1_all = pl.read_parquet(ROOT / "work" / "parquet" / "test_source1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
    O = ROOT / "output_bucket" / a.dst
    O.mkdir(parents=True, exist_ok=True)

    def write(pairs, col, path):
        g = pairs.group_by("s1_id").agg(pl.col("m_id").unique().sort().str.join(",").alias(col)).rename({"s1_id": "source1_entity_id"})
        s1_all.join(g, on="source1_entity_id", how="left", maintain_order="left").with_columns(pl.col(col).fill_null("")).write_csv(
            path, separator="\t", quote_style="never")

    write(sc.select("s1_id", "m_id"), "candidate_entity_ids", O / "candidate_pairs.tsv")
    write(pred, "matched_entity_ids", O / "matching_results.tsv")
    val = subprocess.run([sys.executable, str(ROOT / "data" / "student_resource" / "utils" / "validate_submission.py"),
                          "--matching", str(O / "matching_results.tsv"), "--candidate", str(O / "candidate_pairs.tsv"),
                          "--test-dir", str(ROOT / "data" / "student_resource" / "dataset" / "test")], capture_output=True, text=True)
    status = "PASS" if val.returncode == 0 else "FAIL"
    setting = f"{a.model} scores ({a.scores}), t={a.t}, group thresholds {tg or 'none'}" + (f", decoy tokens dropped ({a.drop_vocab})" if a.drop_vocab else "")
    (O / "NOTES.md").write_text("\n".join([f"# {a.dst}", "", f"Created: {time.strftime('%Y-%m-%d %H:%M')} IST", "", setting,
                                           a.note, "", "Upload `matching_results.tsv` only.", "", "Pairs changed by group thresholds:",
                                           "```", str(tab), "```", "", f"Predicted pairs: {pred.height}", "",
                                           "## Official validator", "```", val.stdout.strip()[-1500:], "```"]))
    with open(ROOT / "output_bucket" / "SUBMISSIONS.md", "a") as fh:
        fh.write(f"| {a.dst} | {time.strftime('%d %b %H:%M')} | {setting} {a.note} | {status} | not yet | |\n")
    pl.Config.set_tbl_rows(40)
    print(tab)
    print(f"predicted pairs {pred.height}; validator {status}")
    print(val.stdout.strip()[-400:])


if __name__ == "__main__":
    main()
