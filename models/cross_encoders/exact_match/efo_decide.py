"""Expected-F0.5-optimal decision per S1 (EFO), as a drop-in replacement for the argmax + threshold step.
Input: a pairs parquet with s1_id, m_id, p, grp, sup, country (+ dtok optional). Keeps every team rule as a hard
constraint: record -> argmax S1 only; decoy-word pairs never; shift_pure / shift_ms_only only via the India restore
(p >= .98); shift_12 needs p >= .95 (corr .98); missing|corr needs .98. Within those, per S1 it chooses the top-k set
(by p) that maximises E[F0.5] under independent Bernoulli(p). Usage:
  python efo_decide.py <pairs.parquet> <out_kept.parquet> [floor=0.2]
Prints how many decisions differ from the plain threshold rule (t = .75)."""
import math, sys
import numpy as np, polars as pl
BASE = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98}; T = 0.75


def thr_expr():
    key = pl.when(pl.col("sup") == "corr").then(pl.col("grp") + "|corr").otherwise(pl.col("grp")); e = pl.lit(T)
    for g_, x in BASE.items():
        if "|" in g_: e = pl.when(key == g_).then(pl.lit(x)).otherwise(e)
    for g_, x in BASE.items():
        if "|" not in g_: e = pl.when((pl.col("grp") == g_) & ~key.is_in([k for k in BASE if "|" in k])).then(pl.lit(x)).otherwise(e)
    return e


def pb(ps):
    d = np.zeros(len(ps) + 1); d[0] = 1.0
    for p in ps:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= (1 - p)
    return d


def efo_select(ps, ok):
    idx = [i for i in range(len(ps)) if ok[i]]; fixed = [ps[i] for i in range(len(ps)) if not ok[i]]
    best_k, best_e = 0, -1.0
    for k in range(len(idx) + 1):
        dtp = pb([ps[i] for i in idx[:k]]); dfn = pb([ps[i] for i in idx[k:]] + fixed); e = 0.0
        for tp in range(len(dtp)):
            if dtp[tp] == 0: continue
            fp = k - tp
            for fn in range(len(dfn)):
                if dfn[fn] == 0: continue
                f = 1.0 if tp + fn + fp == 0 else (0.0 if tp == 0 else 1.25 * tp / (1.25 * tp + 0.25 * fn + fp))
                e += dtp[tp] * dfn[fn] * f
        if e > best_e + 1e-12: best_e, best_k = e, k
    return [idx[j] for j in range(best_k)]


def decide(d, floor=0.2):
    if "dtok" not in d.columns: d = d.with_columns(pl.lit(False).alias("dtok"))
    d = d.with_columns(thr_expr().alias("thr"), pl.col("dtok").fill_null(False), pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r"))
    d = d.with_columns(pl.when((pl.col("country") == "India") & pl.col("grp").is_in(["shift_pure", "shift_ms_only"])).then(pl.lit(0.98)).otherwise(pl.col("thr")).alias("thr"))
    el = d.filter((pl.col("r") == 1) & ~pl.col("dtok") & (pl.col("p") >= floor)).sort(["s1_id", "p"], descending=[False, True])
    s, m, p, th = el["s1_id"].to_list(), el["m_id"].to_list(), el["p"].to_list(), el["thr"].to_list()
    out_s, out_m = [], []; i = 0; n = len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]: j += 1
        ok = [(th[t] <= 1.0) and (p[t] >= th[t] if th[t] > 0.75 else True) for t in range(i, j)]
        for t in efo_select(p[i:j], ok): out_s.append(s[i + t]); out_m.append(m[i + t])
        i = j
    efo = pl.DataFrame({"s1_id": out_s, "m_id": out_m})
    base = d.filter((pl.col("r") == 1) & (pl.col("p") >= pl.col("thr")) & ~pl.col("dtok")).select("s1_id", "m_id")
    return efo, base


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    d = pl.read_parquet(src)
    efo, base = decide(d, floor)
    added = efo.join(base, on=["s1_id", "m_id"], how="anti"); dropped = base.join(efo, on=["s1_id", "m_id"], how="anti")
    print(f"threshold rule keeps {base.height}; EFO keeps {efo.height}; EFO adds {added.height}, drops {dropped.height}")
    efo.write_parquet(dst)
