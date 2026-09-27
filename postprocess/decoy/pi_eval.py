"""Leaderboard-change estimator for post-processing rules (positive-invariance model, corroboration-aware).

Purpose: estimate the TEST effect of decision rules such as group thresholds on house-number shift
classes, for any model with VALID and test stage-2 scores. It predicted rule 2 at +0.0025 LB (actual +0.0020),
where dense_s1 predicted +0.006.
Model: within a cell (country x num_rel group x corroboration x score band) the number of TRUE pairs per S1 is the
same on VALID and test ("positive invariance"); test adds negatives. Test precision q = VALID true per S1 / test pairs
per S1, measured on the VALID states only (NY; TS+AP), then applied to every test pair of that cell in all states.
France uses US or India cells (both reported). Per-S1 F0.5 is simulated with independent pair labels.
Corroboration: 'corr' = the record's new house number (absent from S1) is also carried by another candidate record
of the same S1; 'lone' = it is not; 'none' = the record has no new number.

Usage:
  python postprocess/decoy/pi_eval.py --valid work_v4/p1/model_v7/valid_scored.parquet \
      --test-root work_v4 --model model_v7 --t 0.70 [--texts work_v4] [--sims 4] [--variants file.json]
  variants JSON: {"name": {"t": 0.70, "tg": {"shift_pure": 1.01, "shift_12|corr": 0.98}}, ...}; a key "grp|sup"
  overrides "grp". Without --variants a built-in family is scored.
Inputs: VALID labels from work/advisor/valid_pairs_model_v2b.parquet (every true VALID pair); scores parquet
  (s1_id, m_id, p); test scores <test-root>/p1/test/<c>/stage2_scored_<model>.parquet; texts from <texts>/p1/test and
  work/p1/train (addr_text, addr_state).
Outputs: prints VALID actual macro, test expected F per country and LB delta estimate per variant; writes
  work/director/pi_eval_<model>.parquet. Runtime ~4-6 min, ~2.5 GB peak, 3 threads.
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "3")
sys.path.insert(0, "postprocess/decoy")
sys.path.insert(0, "postprocess/checks")
import dlib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from fingerprint import num_rel  # noqa: E402

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
R = dlib.ROOT
NUM = re.compile(r"\d+")
W = {"US": 0.3827, "India": 0.4675, "France": 0.1497}
NTEST = {"US": 663106, "India": 809986, "France": 259452}
LAM = {"US": 0.0556, "India": 0.1076}
FINE = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999]
COARSE = [0.75, 0.98]


def nset(t):
    return {str(int(x)) for x in NUM.findall(t or "") if len(x) <= 7}


def enrich(pairs, allpairs, s1p, poolp):
    s1 = pl.scan_parquet(s1p).select("entity_id", "addr_text").join(
        pairs.select(pl.col("s1_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect()
    ap_ = allpairs.join(pairs.select("s1_id").unique(), on="s1_id", how="semi")
    pool = pl.scan_parquet(poolp).select("entity_id", "addr_text").join(
        ap_.select(pl.col("m_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect()
    sa = dict(zip(s1["entity_id"].to_list(), s1["addr_text"].to_list()))
    sb = dict(zip(pool["entity_id"].to_list(), pool["addr_text"].to_list()))
    others = {}
    for a, b in zip(ap_["s1_id"].to_list(), ap_["m_id"].to_list()):
        others.setdefault(a, []).append(b)
    nb = {}
    grp, sup = [], []
    for a, b in zip(pairs["s1_id"].to_list(), pairs["m_id"].to_list()):
        grp.append(num_rel(sa.get(a, ""), sb.get(b, "")))
        new = nset(sb.get(b, "")) - nset(sa.get(a, ""))
        if not new:
            sup.append("none"); continue
        k = 0
        for bb in others.get(a, []):
            if bb == b:
                continue
            if bb not in nb:
                nb[bb] = nset(sb.get(bb, ""))
            if new <= nb[bb]:
                k += 1
                break
        sup.append("corr" if k else "lone")
    return pairs.with_columns(pl.Series("grp", grp), pl.Series("sup", sup))


def argmax(d):
    return d.with_columns((pl.col("p") == pl.col("p").max().over("m_id")).alias("am"))


def load_valid(path):
    lab = pl.read_parquet(f"{R}/work/advisor/valid_pairs_model_v2b.parquet", columns=["s1_id", "m_id", "y", "country"])
    s1c = lab.select("s1_id", "country").unique("s1_id")
    tr = lab.filter(pl.col("y") == 1).select("s1_id", "m_id", pl.lit(1).alias("yt"))
    sc = argmax(pl.read_parquet(path, columns=["s1_id", "m_id", "p"]))
    d = tr.join(sc, on=["s1_id", "m_id"], how="full", coalesce=True).with_columns(
        pl.col("yt").fill_null(0).alias("y"), pl.col("am").fill_null(False)).drop("yt")
    d = d.join(s1c, on="s1_id", how="left").filter(pl.col("country").is_not_null())
    st = pl.concat([pl.read_parquet(f"{R}/work/p1/train/{c}/s1.parquet", columns=["entity_id", "addr_state"]) for c in ["US", "India"]])
    d = d.join(st.rename({"entity_id": "s1_id", "addr_state": "state"}), on="s1_id", how="left")
    parts = []
    for c in ["US", "India"]:
        dc = d.filter((pl.col("country") == c) & pl.col("p").is_not_null())
        tgt = dc.filter(pl.col("am") & (pl.col("p") >= 0.5)).select("s1_id", "m_id")
        parts.append(enrich(tgt, dc.select("s1_id", "m_id"), f"{R}/work/p1/train/{c}/s1.parquet", f"{R}/work/p1/train/{c}/pool.parquet"))
    return d.join(pl.concat(parts), on=["s1_id", "m_id"], how="left")


def load_test(root, model, texts):
    out = []
    for c in ["US", "India", "France"]:
        f = f"{R}/{root}/p1/test/{c}/stage2_scored_{model}.parquet"
        t = argmax(pl.read_parquet(f, columns=["s1_id", "m_id", "p"])).filter(pl.col("am"))
        tx = texts if os.path.exists(f"{R}/{texts}/p1/test/{c}/s1.parquet") else "work_v4"
        allp = pl.read_parquet(f, columns=["s1_id", "m_id"])
        tgt = t.filter(pl.col("p") >= 0.5).select("s1_id", "m_id")
        es = []
        for k in range(8):  # S1 chunks keep the python text dictionaries small (v9 India has 7.4 candidates per S1)
            tk = tgt.filter(pl.col("s1_id").hash(3) % 8 == k)
            es.append(enrich(tk, allp.join(tk.select("s1_id").unique(), on="s1_id", how="semi"),
                             f"{R}/{tx}/p1/test/{c}/s1.parquet", f"{R}/{tx}/p1/test/{c}/pool.parquet"))
        e = pl.concat(es)
        del allp, es
        st = pl.read_parquet(f"{R}/{tx}/p1/test/{c}/s1.parquet", columns=["entity_id", "addr_state"])
        t = t.join(e, on=["s1_id", "m_id"], how="left").join(st.rename({"entity_id": "s1_id", "addr_state": "state"}), on="s1_id", how="left")
        out.append(t.with_columns(pl.lit(c).alias("country")))
        print(f"test {c}: argmax pairs {t.height}", flush=True)
    return pl.concat(out)


def matched(d):
    return d.filter(((pl.col("country") == "US") & (pl.col("state") == "ny")) |
                    ((pl.col("country") == "India") & pl.col("state").is_in(["ts", "ap"])))


def cells(v, t, pool=False):
    """pool=True: for groups that are not shift/decoy groups, use one density ratio over p >= 0.75 (score migration of true
    pairs between bands on test otherwise looks like excess in one band and a clipped deficit in the other)."""
    va = matched(v.filter(pl.col("am") & pl.col("p").is_not_null()))
    ta = matched(t)
    nV = {"US": 100256, "India": 72116}
    nT = {"US": 50761, "India": 66934}
    def lab(d):
        return d.with_columns(pl.col("grp").fill_null("lowp"), pl.col("sup").fill_null("-"),
                              pl.col("p").cut(FINE, left_closed=True).cast(pl.Utf8).alias("fb"),
                              pl.col("p").cut(COARSE, left_closed=True).cast(pl.Utf8).alias("cb"))
    va, ta = lab(va), lab(ta)
    key_c = ["country", "grp", "sup", "cb"]
    rc = (va.group_by(key_c).agg(pl.len().alias("nV"), pl.col("y").sum().alias("tV"))
            .join(ta.group_by(key_c).agg(pl.len().alias("nT")), on=key_c, how="full", coalesce=True).fill_null(0))
    rc = rc.with_columns(((pl.col("nT") / pl.col("country").replace_strict(nT, return_dtype=pl.Float64)) /
                          (pl.col("nV") / pl.col("country").replace_strict(nV, return_dtype=pl.Float64))).alias("ratio"))
    rc = rc.with_columns(pl.when((pl.col("nV") >= 15) & (pl.col("nT") >= 15)).then(pl.col("ratio")).otherwise(1.0).clip(1.0, None).alias("ratio"),
                         ((pl.col("tV") + 0.5) / (pl.col("nV") + 1)).alias("precC"))
    if pool:
        hi = rc.filter(pl.col("cb") != "[-inf, 0.75)").group_by("country", "grp", "sup").agg(pl.col("nV").sum().alias("pV"), pl.col("nT").sum().alias("pT"))
        hi = hi.with_columns(((pl.col("pT") / pl.col("country").replace_strict(nT, return_dtype=pl.Float64)) /
                              (pl.col("pV") / pl.col("country").replace_strict(nV, return_dtype=pl.Float64))).alias("pr"))
        hi = hi.with_columns(pl.when((pl.col("pV") >= 15) & (pl.col("pT") >= 15)).then(pl.col("pr")).otherwise(1.0).clip(1.0, None).alias("pr"))
        rc = rc.join(hi.select("country", "grp", "sup", "pr"), on=["country", "grp", "sup"], how="left").with_columns(
            pl.when(~pl.col("grp").str.starts_with("shift") & (pl.col("grp") != "dtok") & (pl.col("cb") != "[-inf, 0.75)") & pl.col("pr").is_not_null())
            .then(pl.col("pr")).otherwise(pl.col("ratio")).alias("ratio")).drop("pr")
    key_f = ["country", "grp", "sup", "fb"]
    fc = va.group_by(key_f + ["cb"]).agg(pl.len().alias("nVf"), pl.col("y").sum().alias("tVf"))
    band = va.group_by("country", "fb").agg((pl.col("y").sum() / pl.len()).alias("precB"))
    return rc, fc, band, lab


def assign_q(d, rc, fc, band, lab, src=None):
    """Adds q; keeps only the columns decisions and the simulation need (memory)."""
    d = lab(d.select("s1_id", "p", "am", "grp", "sup", "country"))
    if src:
        d = d.drop("country").with_columns(pl.lit(src).alias("country"))
    d = (d.join(rc.select("country", "grp", "sup", "cb", "ratio", "precC"), on=["country", "grp", "sup", "cb"], how="left")
          .join(fc, on=["country", "grp", "sup", "fb", "cb"], how="left").join(band, on=["country", "fb"], how="left"))
    prec = pl.when(pl.col("nVf") >= 20).then((pl.col("tVf") + 0.5) / (pl.col("nVf") + 1)).when(pl.col("precC").is_not_null()).then(pl.col("precC")).otherwise(pl.col("precB"))
    return d.with_columns((prec.fill_null(0.0) / pl.col("ratio").fill_null(1.0)).clip(0, 1).alias("q")).select(
        "s1_id", "p", "am", "grp", "sup", "q")


def decide(d, spec):
    t = spec.get("t", 0.75)
    tg = spec.get("tg", {})
    # grp keys first, then grp|sup keys (the more specific key wins)
    thr2 = pl.lit(t)
    for k, val in tg.items():
        if "|" not in k:
            thr2 = pl.when(pl.col("grp") == k).then(pl.lit(val)).otherwise(thr2)
    for k, val in tg.items():
        if "|" in k:
            g, s = k.split("|")
            thr2 = pl.when((pl.col("grp") == g) & (pl.col("sup") == s)).then(pl.lit(val)).otherwise(thr2)
    return (pl.col("am") & (pl.col("p") >= thr2)).fill_null(False)


def dlib_expected(d, lam, n_total, sims, seed=0):
    d = d.select("s1_id", "q", "dec").sort("s1_id")
    sid = d["s1_id"].to_numpy()
    _, idx = np.unique(sid, return_index=True)
    g = np.zeros(len(sid), np.int64); g[idx[1:]] = 1; g = np.cumsum(g)
    G = int(g.max()) + 1
    q = d["q"].to_numpy(); pr = d["dec"].to_numpy().astype(bool)
    npred = np.bincount(g, weights=pr, minlength=G)
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(sims):
        y = rng.random(len(q)) < q
        tp = np.bincount(g, weights=y & pr, minlength=G)
        nt = np.bincount(g, weights=y, minlength=G) + rng.poisson(lam, G)
        P = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
        Rr = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
        f = np.where(nt == 0, (npred == 0).astype(float), np.where(tp == 0, 0.0, 1.25 * P * Rr / np.maximum(0.25 * P + Rr, 1e-12)))
        acc += f.sum()
    return (acc / sims + (n_total - G) * np.exp(-lam)) / n_total


def builtin(t):
    return {
        "plain": {"t": t},
        "rule2": {"t": t, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.98}},
        "rule2_lone_pure_0.98": {"t": t, "tg": {"shift_pure": 1.01, "shift_pure|lone": 0.98, "shift_ms_only": 1.01, "shift_12": 0.98}},
        "rule2_lone_pure_0.85": {"t": t, "tg": {"shift_pure": 1.01, "shift_pure|lone": max(t, 0.85), "shift_ms_only": 1.01, "shift_12": 0.98}},
        "rule2_ms_lone_0.98": {"t": t, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_ms_only|lone": 0.98, "shift_12": 0.98}},
        "shift12_0.95": {"t": t, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95}},
        "shift12_0.99": {"t": t, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.99}},
        "rule2+missing_corr0.98": {"t": t, "tg": {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.98, "missing|corr": 0.98}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", default="work_v4")
    ap.add_argument("--t", type=float, default=0.75)
    ap.add_argument("--sims", type=int, default=4)
    ap.add_argument("--variants", default="")
    ap.add_argument("--cache", action="store_true", help="reuse/write enriched tables work/director/pi_cache_<model>_{valid,test}.parquet")
    ap.add_argument("--cells", action="store_true", help="print every cell with ratio >= 1.4 in the predicted range")
    ap.add_argument("--out", default="", help="output parquet name suffix")
    ap.add_argument("--pool", action="store_true", help="pool score bands (p >= 0.75) for non-shift groups; recommended since 25 Sep 21:25")
    a = ap.parse_args()
    variants = json.load(open(a.variants)) if a.variants else builtin(a.t)
    cv, ct = f"{R}/work/director/pi_cache_{a.model}_valid.parquet", f"{R}/work/director/pi_cache_{a.model}_test.parquet"
    if a.cache and os.path.exists(cv) and os.path.exists(ct):
        v, t = pl.read_parquet(cv), pl.read_parquet(ct)
        print("using cached enriched tables", flush=True)
    else:
        v = load_valid(f"{R}/{a.valid}" if not a.valid.startswith("/") else a.valid)
        t = load_test(a.test_root, a.model, a.texts)
        if a.cache:
            v.write_parquet(cv); t.write_parquet(ct)
    rc, fc, band, lab = cells(v, t, pool=a.pool)
    print("test/VALID density ratio, predicted-range shift cells (same states):")
    print(rc.filter(pl.col("grp").str.starts_with("shift") & (pl.col("cb") != "[-inf, 0.75)")).sort("country", "grp", "sup", "cb")
            .with_columns(pl.col(pl.Float64).round(3)))
    if a.cells:
        print("ALL cells p>=0.5 with ratio >= 1.4 (nV, nT >= 15):")
        print(rc.filter((pl.col("ratio") >= 1.4) & (pl.col("grp") != "lowp")).sort("country", "grp", "sup", "cb").with_columns(pl.col(pl.Float64).round(3)))
    vv = v.filter(pl.col("p").is_not_null())
    res = []
    tq = {}
    for c in ["US", "India", "France"]:
        for src in ([c] if c != "France" else ["US", "India"]):
            tq[(c, src)] = assign_q(t.filter(pl.col("country") == c), rc, fc, band, lab, src)
    for nm, spec in variants.items():
        row = {"variant": nm}
        dv = v.with_columns(decide(v, spec).alias("dec"))
        m = dlib.macro(dlib.entity_scores(dv, "dec"))
        row.update({"VALID": m["all"], "VALID_US": m["US"], "VALID_IN": m["India"]})
        lb = 0.0
        for (c, src), d in tq.items():
            x = d.with_columns(decide(d, spec).alias("dec"))
            e = dlib_expected(x, LAM[src], NTEST[c], a.sims)
            row[f"EF_{c}_{src}"] = e
        for c in ["US", "India"]:
            lb += W[c] * row[f"EF_{c}_{c}"]
        lb += W["France"] * 0.5 * (row["EF_France_US"] + row["EF_France_India"])
        row["LB_raw"] = lb
        res.append(row)
        print(nm, {k: round(x, 5) for k, x in row.items() if k != "variant"}, flush=True)
    r = pl.DataFrame(res)
    base = r["LB_raw"][0]
    r = r.with_columns((pl.col("LB_raw") - base).alias("LB_delta_vs_first"), (pl.col("VALID") - r["VALID"][0]).alias("VALID_delta"))
    print(r.select("variant", "VALID", "VALID_delta", "LB_raw", "LB_delta_vs_first").with_columns(pl.col(pl.Float64).round(5)))
    r.write_parquet(f"{R}/work/director/pi_eval_{a.model}{a.out}.parquet")


if __name__ == "__main__":
    main()
