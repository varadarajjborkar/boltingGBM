"""A1 VALID gate (A1_NEWKEY report section 4 step 6). Base = current best shipped decision on VALID
(noaddr_ceiling.base_decision: model_v10pc_s3 stage 3, t .75 + BASE + dtok + India shift restore; VALID .98918).
A NEW pair (listed in newpairs_valid_<c>, not in the base stage-2 set) is added when, in the A1 v10 stage-2 scores:
  p >= T, it is the record's argmax over ALL its A1 pairs (old + new), the record has no accepted pair in the base,
  and it passes the shipped rules: grp / sup from pi_eval.enrich (the S1's other candidates = its A1 pairs), threshold
  max(T, BASE group threshold), decoy-vocabulary token (data/decoy_vocab_validated.json) -> reject unless shift group.
  No India shift restore for new pairs. One S1 per record (highest p, then s1_id).
Prints VALID delta vs base (all, S1-hash halves hash(13) % 2, US, India), adds and precision per T in (.8, .9, .95), and
picks the best T whose delta is >= +0.0003 on BOTH halves (else FAIL). Writes <outdir>/gate_summary.json (read by
a1_apply.py for the same-state density check) and <outdir>/gate_adds_valid_T<T>.parquet.
Usage: POLARS_MAX_THREADS=3 python postprocess/fixes/a1_gate.py --valid <valid_scored_a1.parquet>
       [--newpairs-dir work/errfix/out/a1] [--outdir work/errfix/out/a1]   (~2-3 min, < 3 GB)"""
import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "3")
ROOT = os.environ.get("ER_ROOT", ".")
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/postprocess/fixes")
sys.path.insert(0, f"{ROOT}/postprocess/decoy")
import polars as pl  # noqa: E402
import pi_eval  # noqa: E402
from compare_models import BASE  # noqa: E402
from flag_rule_eval import flags  # noqa: E402

TS = [0.8, 0.9, 0.95]
MIN_GAIN = 0.0003
NVALID = {"US": 102314, "India": 73301}  # VALID S1 = all NY (US) / AP+TS (India)
SAME = {"US": ["ny"], "India": ["ap", "ts"], "France": []}
VOCAB = set(json.load(open(f"{ROOT}/data/decoy_vocab_validated.json")))


def texts_dir(split, c):
    return f"{ROOT}/work_v4/p1/{'test' if split == 'test' else 'train'}/{c}"


def load_new(ndir, split, c):
    """newpairs (i, j row indexes into work_v4 s1 / pool) -> (s1_id, m_id)."""
    d = texts_dir(split, c)
    k = pl.read_parquet(f"{ndir}/newpairs_{split}_{c}.parquet", columns=["i", "j"]).unique()
    s1 = pl.read_parquet(f"{d}/s1.parquet", columns=["entity_id"]).with_row_index("i").with_columns(pl.col("i").cast(pl.Int32))
    pool = pl.read_parquet(f"{d}/pool.parquet", columns=["entity_id"]).with_row_index("j").with_columns(pl.col("j").cast(pl.Int32))
    return (k.with_columns(pl.col("i").cast(pl.Int32), pl.col("j").cast(pl.Int32))
             .join(s1.rename({"entity_id": "s1_id"}), on="i").join(pool.rename({"entity_id": "m_id"}), on="j").select("s1_id", "m_id"))


def candidates(sc, new, blocked, tmin, split, c):
    """sc: A1 pairs (s1_id, m_id, p) of one country, old + new. Returns new pairs with p >= tmin, record argmax, record not
    blocked, with grp / sup / dtok computed the shipped way (other candidates of the S1 = its A1 pairs)."""
    sc = sc.with_columns((pl.col("p") == pl.col("p").max().over("m_id")).alias("am"))
    cand = (sc.join(new, on=["s1_id", "m_id"], how="semi").filter(pl.col("am") & (pl.col("p") >= tmin))
              .filter(~pl.col("m_id").is_in(blocked.implode())))
    info = {"new_scored": sc.join(new, on=["s1_id", "m_id"], how="semi").height, "cand": cand.height}
    if cand.height == 0:
        return cand.with_columns(pl.lit(None, pl.Utf8).alias("grp"), pl.lit(None, pl.Utf8).alias("sup"), pl.lit(False).alias("dtok")), info
    d = texts_dir(split, c)
    allp = sc.select("s1_id", "m_id").join(cand.select("s1_id").unique(), on="s1_id", how="semi")
    tg = cand.select("s1_id", "m_id")
    e = pi_eval.enrich(tg, allp, f"{d}/s1.parquet", f"{d}/pool.parquet")
    f = flags(tg, allp, f"{d}/s1.parquet", f"{d}/pool.parquet", VOCAB).select("s1_id", "m_id", "dtok")
    cand = cand.join(e, on=["s1_id", "m_id"], how="left").join(f, on=["s1_id", "m_id"], how="left").with_columns(pl.col("dtok").fill_null(False))
    cand = cand.with_columns(pl.when(pl.col("dtok") & ~pl.col("grp").fill_null("").str.starts_with("shift")).then(pl.lit("dtok"))
                             .otherwise(pl.col("grp")).alias("grp"))
    return cand, info


def accept(cand, T):
    """Shipped rules at base threshold max(T, group threshold); one S1 per record."""
    if cand.height == 0:
        return cand
    ok = pi_eval.decide(cand, {"t": T, "tg": {**BASE, "dtok": 1.01}}) & (pl.col("p") >= T)
    return cand.filter(ok).sort(["p", "s1_id"], descending=[True, False]).unique("m_id", keep="first", maintain_order=True)


def main():
    from noaddr_ceiling import base_decision
    from valid_macro import macro
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True, help="valid_scored_a1.parquet (s1_id, m_id, p[, y]) over old + new VALID pairs")
    ap.add_argument("--newpairs-dir", default=f"{ROOT}/work/errfix/out/a1")
    ap.add_argument("--outdir", default=f"{ROOT}/work/errfix/out/a1")
    a = ap.parse_args()
    v = base_decision().select("s1_id", "m_id", "p", "y", "D", "country")
    b = macro(v)
    print("base VALID " + " ".join(f"{k} {x:.5f}" for k, x in b.items()), flush=True)
    blocked = v.filter(pl.col("D"))["m_id"].unique()
    basepairs = v.filter(pl.col("p").is_not_null()).select("s1_id", "m_id")
    truth = v.filter(pl.col("y") == 1).select("s1_id", "m_id", pl.lit(1).alias("yt"))
    scols = [x for x in ["s1_id", "m_id", "p", "y"] if x in pl.read_parquet_schema(a.valid)]
    sc = pl.read_parquet(a.valid, columns=scols).rename({"y": "y_sc"} if "y" in scols else {})
    print(f"A1 VALID scored pairs {sc.height:,}, records {sc['m_id'].n_unique():,}", flush=True)
    cands = []
    for c in ["US", "India"]:
        new = load_new(a.newpairs_dir, "valid", c)
        n0 = new.height
        new = new.join(basepairs, on=["s1_id", "m_id"], how="anti")
        cs1 = pl.read_parquet(f"{texts_dir('valid', c)}/s1.parquet", columns=["entity_id"])["entity_id"]
        scc = sc.filter(pl.col("s1_id").is_in(cs1.implode())).select("s1_id", "m_id", "p")
        cand, info = candidates(scc, new, blocked, min(TS), "valid", c)
        ntrue = new.join(truth, on=["s1_id", "m_id"], how="semi").height
        nts = scc.join(new, on=["s1_id", "m_id"], how="semi").join(truth, on=["s1_id", "m_id"], how="semi").height
        print(f"{c}: newpairs {n0:,} (in base set {n0 - new.height}), true {ntrue}; scored by A1 {info['new_scored']:,} (true {nts}); "
              f"argmax & record free & p >= {min(TS)}: {info['cand']:,}", flush=True)
        cands.append(cand.with_columns(pl.lit(c).alias("country")))
    cand = pl.concat(cands, how="diagonal_relaxed").join(truth, on=["s1_id", "m_id"], how="left").with_columns(pl.col("yt").fill_null(0).alias("y"))
    if "y" in scols:
        cand = cand.join(sc.select("s1_id", "m_id", "y_sc"), on=["s1_id", "m_id"], how="left")
        mis = cand.filter(pl.col("y_sc").cast(pl.Int32) != pl.col("y")).height
        print(f"label check (A1 file y vs VALID truth) on candidates: {mis} mismatches", flush=True)
    rows, summary = [], {"base": b, "T": {}}
    for T in TS:
        acc = accept(cand, T)
        add = acc.select("s1_id", "m_id", pl.lit(True).alias("add"))
        x = v.join(add, on=["s1_id", "m_id"], how="left").with_columns((pl.col("D") | pl.col("add").fill_null(False)).alias("D")).drop("add")
        extra = acc.join(v.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti").select(
            "s1_id", "m_id", pl.lit(None, pl.Float64).alias("p"), pl.lit(0).cast(v["y"].dtype).alias("y"), pl.lit(True).alias("D"), "country")
        x = pl.concat([x, extra.select(x.columns)])
        m = macro(x)
        dl = {k: m[k] - b[k] for k in m}
        r = {"T": T, "d_all": dl["all"], "d_h0": dl["h0"], "d_h1": dl["h1"], "d_US": dl["US"], "d_India": dl["India"]}
        st = {}
        for c in ["US", "India"]:
            ac = acc.filter(pl.col("country") == c)
            n, tp = ac.height, int(ac["y"].sum()) if ac.height else 0
            r[f"add_{c}"], r[f"prec_{c}"], r[f"per1k_{c}"] = n, (tp / n if n else float("nan")), 1000 * n / NVALID[c]
            st[c] = {"adds": n, "tp": tp, "per1k": 1000 * n / NVALID[c],
                     "grp": ac.group_by("grp").len().sort("len", descending=True).head(6).rows() if n else []}
        r["pass"] = dl["h0"] >= MIN_GAIN and dl["h1"] >= MIN_GAIN
        rows.append(r)
        summary["T"][str(T)] = {"delta": dl, **st, "pass": r["pass"]}
        acc.write_parquet(f"{a.outdir}/gate_adds_valid_T{T}.parquet")
    o = pl.DataFrame(rows)
    pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20)
    print(o.with_columns(pl.col(pl.Float64).round(5)))
    for T in TS:
        print(f"T {T} adds by group:", {c: summary["T"][str(T)][c]["grp"] for c in ["US", "India"]})
    ok = o.filter(pl.col("pass"))
    chosen = float(ok.sort("d_all", descending=True)["T"][0]) if ok.height else None
    summary["chosen"] = chosen
    json.dump(summary, open(f"{a.outdir}/gate_summary.json", "w"), indent=1)
    if chosen is None:
        print(f"GATE FAIL: no T with delta >= +{MIN_GAIN} on both halves. Do not ship A1 adds.")
    else:
        s = summary["T"][str(chosen)]
        print(f"GATE PASS: T = {chosen}  VALID {s['delta']['all']:+.5f} (h0 {s['delta']['h0']:+.5f}, h1 {s['delta']['h1']:+.5f}); "
              f"adds US {s['US']['adds']} ({s['US']['per1k']:.2f}/1k S1), India {s['India']['adds']} ({s['India']['per1k']:.2f}/1k S1)")
        print(f"next: python postprocess/fixes/a1_apply.py --t {chosen} --base output_bucket/stack_e5hyb2 --out work/errfix/sub/stack_e5hyb2_a1")
    print(f"wrote {a.outdir}/gate_summary.json")


if __name__ == "__main__":
    main()
