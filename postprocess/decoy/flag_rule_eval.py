"""Evaluate a candidate decoy-family flag as a drop rule with pi_eval (VALID actual + test expected F, LB delta).

Purpose: after postprocess/decoy/shift_classifier.py finds a test-inflated family, turn it into a rule and measure it. The flag
is computed for every argmax pair with p >= 0.5 on VALID (all VALID states) and test (all states); flagged pairs that are
not already in a shift group get their own pi_eval group "flag", so its test precision comes from same-state densities.
Flags (token-based, cheap): dtok (adds a decoy-vocabulary token), name_corr (a name token the record adds is also carried by
another candidate record of the same S1), street_corr (same for non-number address words), and combinations.

Usage: python postprocess/decoy/flag_rule_eval.py --model model_v8 --flag name_corr [--t 0.70]
Inputs: work/director/pi_cache_<model>_{valid,test}.parquet, stage-2 scores, texts, data/decoy_vocab_validated.json.
Outputs: prints cells for the flag and variant table; writes work/director/flag_eval_<model>_<flag>.parquet.
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "3")
sys.path.insert(0, "postprocess/decoy")
import dlib  # noqa: E402
import pi_eval  # noqa: E402
import polars as pl  # noqa: E402

R = dlib.ROOT
TOK = re.compile(r"[a-z]+")
STOP = {"rd", "st", "ave", "road", "street", "no", "nr", "near", "opp", "ngr", "nagar", "fl", "floor", "unit", "apt", "suite", "ste",
        "dr", "ln", "blvd", "the", "of", "and", "plot", "h", "d", "door", "flat", "sec", "sector", "colony", "main", "cross"}


def flags(target, allpairs, s1p, poolp, vocab):
    ap_ = allpairs.join(target.select("s1_id").unique(), on="s1_id", how="semi")
    s1 = pl.scan_parquet(s1p).select("entity_id", "name_core", "addr_text").join(
        target.select(pl.col("s1_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect()
    pool = pl.scan_parquet(poolp).select("entity_id", "name_core", "addr_text").join(
        ap_.select(pl.col("m_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect()
    sn = dict(zip(s1["entity_id"].to_list(), s1["name_core"].fill_null("").to_list()))
    sa = dict(zip(s1["entity_id"].to_list(), s1["addr_text"].fill_null("").to_list()))
    bn = dict(zip(pool["entity_id"].to_list(), pool["name_core"].fill_null("").to_list()))
    ba = dict(zip(pool["entity_id"].to_list(), pool["addr_text"].fill_null("").to_list()))
    del s1, pool
    others = {}
    for a, b in zip(ap_["s1_id"].to_list(), ap_["m_id"].to_list()):
        others.setdefault(a, []).append(b)
    nt, at = {}, {}
    def ntok(m):
        if m not in nt:
            nt[m] = set(bn.get(m, "").split())
        return nt[m]
    def atok(m):
        if m not in at:
            at[m] = {w for w in TOK.findall(ba.get(m, "")) if len(w) >= 3 and w not in STOP}
        return at[m]
    out = {"n_add": [], "dtok": [], "name_corr": [], "street_corr": []}
    for a, b in zip(target["s1_id"].to_list(), target["m_id"].to_list()):
        add = ntok(b) - set(sn.get(a, "").split())
        sadd = atok(b) - {w for w in TOK.findall(sa.get(a, "")) if len(w) >= 3 and w not in STOP}
        oth = [o for o in others.get(a, []) if o != b]
        out["n_add"].append(len(add))
        out["dtok"].append(bool(add & vocab))
        out["name_corr"].append(bool(add) and any(add <= ntok(o) for o in oth))
        out["street_corr"].append(bool(sadd) and any(sadd <= atok(o) for o in oth))
    return target.select("s1_id", "m_id").with_columns(*[pl.Series(k, v) for k, v in out.items()])


FLAGS = {
    "name_corr": pl.col("name_corr") & ~pl.col("dtok"),
    "street_corr": pl.col("street_corr"),
    "name_or_street_corr": (pl.col("name_corr") & ~pl.col("dtok")) | pl.col("street_corr"),
    "dtok": pl.col("dtok"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_v8")
    ap.add_argument("--flag", default="name_corr", choices=list(FLAGS))
    ap.add_argument("--t", type=float, default=0.70)
    ap.add_argument("--sims", type=int, default=4)
    ap.add_argument("--valid", default="", help="VALID scores parquet (s1_id, m_id, p); default <model dir>/valid_scored.parquet")
    ap.add_argument("--stage", default="all", choices=["all", "flags", "eval"], help="flags and eval run as separate processes (memory)")
    a = ap.parse_args()
    FP = f"{R}/work/director/flags_{a.model}.parquet"
    if a.stage == "all":
        import subprocess
        if not os.path.exists(FP):
            subprocess.run([sys.executable, __file__, "--model", a.model, "--flag", a.flag, "--stage", "flags"], check=True)
        subprocess.run([sys.executable, __file__, "--model", a.model, "--flag", a.flag, "--t", str(a.t), "--sims", str(a.sims), "--stage", "eval"], check=True)
        return
    vocab = set(json.load(open(f"{R}/data/decoy_vocab_validated.json")))
    v = pl.read_parquet(f"{R}/work/director/pi_cache_{a.model}_valid.parquet", columns=["s1_id", "m_id", "y", "p", "am", "country", "state", "grp", "sup"])
    t = pl.read_parquet(f"{R}/work/director/pi_cache_{a.model}_test.parquet", columns=["s1_id", "m_id", "p", "am", "country", "state", "grp", "sup"])
    if a.stage == "eval":
        f = pl.read_parquet(FP)
        v = v.join(f.filter(pl.col("split") == "valid").drop("split"), on=["s1_id", "m_id"], how="left")
        t = t.join(f.filter(pl.col("split") == "test").drop("split"), on=["s1_id", "m_id"], how="left")
        del f
    if a.stage == "flags":
        vdir = "work/p1/model_v2b" if a.model == "model_v2b" else f"work_v4/p1/{a.model}"
        vpath = a.valid if a.valid else f"{vdir}/valid_scored.parquet"
        compute_flags(a, v, t, vpath, vocab, FP)
        return
    run_eval(a, v, t)


def compute_flags(a, v, t, vpath, vocab, FP):
    vsc = pl.read_parquet(vpath if vpath.startswith("/") else f"{R}/{vpath}", columns=["s1_id", "m_id", "p"])
    fl = []
    for c in ["US", "India"]:
        tg = v.filter((pl.col("country") == c) & pl.col("am") & (pl.col("p") >= 0.5))
        fl.append(flags(tg, vsc, f"{R}/work/p1/train/{c}/s1.parquet", f"{R}/work/p1/train/{c}/pool.parquet", vocab))
    v = v.join(pl.concat(fl), on=["s1_id", "m_id"], how="left")
    fl = []
    for c in ["US", "India", "France"]:
        tg = t.filter((pl.col("country") == c) & pl.col("am") & (pl.col("p") >= 0.5))
        allp = pl.read_parquet(f"{R}/work_v4/p1/test/{c}/stage2_scored_{a.model}.parquet", columns=["s1_id", "m_id"]).join(
            tg.select("s1_id").unique(), on="s1_id", how="semi")
        for k in range(12):  # S1 chunks keep the text dictionaries small
            tk = tg.filter(pl.col("s1_id").hash(3) % 12 == k)
            fl.append(flags(tk, allp.join(tk.select("s1_id").unique(), on="s1_id", how="semi"),
                            f"{R}/work_v4/p1/test/{c}/s1.parquet", f"{R}/work_v4/p1/test/{c}/pool.parquet", vocab))
        print(c, "flags done", flush=True)
    tf = pl.concat(fl)
    vf = v.select("s1_id", "m_id", "n_add", "dtok", "name_corr", "street_corr").filter(pl.col("n_add").is_not_null())
    pl.concat([vf.with_columns(pl.lit("valid").alias("split")), tf.with_columns(pl.lit("test").alias("split"))]).write_parquet(FP)
    print("flags written", FP, flush=True)


def run_eval(a, v, t):
    fill = [pl.col(k).fill_null(False) for k in ["dtok", "name_corr", "street_corr"]]
    v, t = v.with_columns(fill), t.with_columns(fill)
    F = FLAGS[a.flag]
    NS = F & ~pl.col("grp").fill_null("").str.starts_with("shift")
    # decoy-token pairs keep their own group so the base includes the token rule
    relabel = (pl.when(pl.col("dtok") & ~pl.col("grp").fill_null("").str.starts_with("shift")).then(pl.lit("dtok"))
               .when(NS).then(pl.lit("flag")).otherwise(pl.col("grp")))
    v2, t2 = v.with_columns(relabel.alias("grp")), t.with_columns(relabel.alias("grp"))
    del v, t
    import gc; gc.collect()
    rc, fc, band, lab = pi_eval.cells(v2, t2)
    print(f"cells for group 'flag' ({a.flag}) and 'dtok':")
    print(rc.filter(pl.col("grp").is_in(["flag", "dtok"])).sort("grp", "country", "sup", "cb").with_columns(pl.col(pl.Float64).round(3)))
    print("VALID true rate of flagged pairs p>=0.70 (all VALID states):",
          v2.filter((pl.col("grp") == "flag") & (pl.col("p") >= 0.70)).group_by("country").agg(pl.len(), pl.col("y").mean().round(4)).rows())
    print("test flagged pairs p>=0.70 per 1000 S1:", t2.filter((pl.col("grp") == "flag") & (pl.col("p") >= 0.70)).group_by("country").len().with_columns(
        (pl.col("len") * 1000 / pl.col("country").replace_strict(pi_eval.NTEST, return_dtype=pl.Float64)).round(2)).rows())
    base = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98, "dtok": 1.01}
    V = {"base(corr_strict+dtok)": {"t": a.t, "tg": dict(base)}}
    for nm, extra in {"flag_all": {"flag": 1.01}, "flag<0.98": {"flag": 0.98}, "flag<0.95": {"flag": 0.95},
                      "flag_corr_all": {"flag|corr": 1.01}, "flag_lone+none_all": {"flag|lone": 1.01, "flag|none": 1.01}}.items():
        V[nm] = {"t": a.t, "tg": {**base, **extra}}
    tq = {}
    for c in ["US", "India", "France"]:
        for src in ([c] if c != "France" else ["US", "India"]):
            tq[(c, src)] = pi_eval.assign_q(t2.filter(pl.col("country") == c), rc, fc, band, lab, src)
    t2 = None; gc.collect()
    res = []
    for nm, spec in V.items():
        row = {"variant": nm}
        m = dlib.macro(dlib.entity_scores(v2.with_columns(pi_eval.decide(v2, spec).alias("dec")), "dec"))
        row["VALID"] = m["all"]
        for (c, src), d in tq.items():
            row[f"EF_{c}_{src}"] = pi_eval.dlib_expected(d.with_columns(pi_eval.decide(d, spec).alias("dec")), pi_eval.LAM[src], pi_eval.NTEST[c], a.sims)
        row["LB_raw"] = (pi_eval.W["US"] * row["EF_US_US"] + pi_eval.W["India"] * row["EF_India_India"]
                         + pi_eval.W["France"] * 0.5 * (row["EF_France_US"] + row["EF_France_India"]))
        res.append(row)
    r = pl.DataFrame(res)
    r = r.with_columns((pl.col("LB_raw") - r["LB_raw"][0]).alias("dLB"), (pl.col("VALID") - r["VALID"][0]).alias("dVALID"))
    pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
    print(r.with_columns(pl.col(pl.Float64).round(5)))
    r.write_parquet(f"{R}/work/director/flag_eval_{a.model}_{a.flag}.parquet")


if __name__ == "__main__":
    main()
