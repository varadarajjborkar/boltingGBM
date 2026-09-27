"""Count-shape test per name class (label-free). For S1 holding exactly ONE kept pair of class X, the histogram of the S1's
total kept count k. If X records are real noisy copies, P(k) ~ k * Pref(k) (size-biased: bigger S1 have more copies to
corrupt). If X records are extra false pairs on top of the real ones, P(k) ~ Pref(k - 1) (shifted by one). Pref = the US
test file histogram (same generator distribution as India / train truth, same recall level). Least-squares false share f
over k = 1..8. Calibrators: US classes (file ~ truth, f should be ~0), French exact / typo (true), French listed swaps.
Usage: POLARS_MAX_THREADS=3 python postprocess/checks/count_shape.py stack_pkf2_a1_fr_fxs_us_sf"""
import re, sys, random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import numpy as np
import polars as pl
sys.path.insert(0, "work/errfix")
R = "."  # run from the repo root
F = sys.argv[1]
NUM = re.compile(r"\d+")
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
LEGAL = {"sci", "sa", "a s", "e l r u", "ets", "a s u", "5as", "sarl", "eurl", "sas"}
GENW = {"france", "services", "groupe", "developpement", "service", "centre", "holding", "associes", "st"}
def cls(a, b, aa, ab):
    same = bool(nk(aa)) and nk(aa) == nk(ab)
    tag = "same" if same else ("noaddr" if not ab.strip() else "diff")
    if a == b: return f"exact_{tag}"
    ta, tb = a.split(), b.split(); sa, sb = set(ta), set(tb)
    if sa == sb: return f"reorder_{tag}"
    if sa < sb:
        x = " ".join(sorted(sb - sa))
        return f"add_legal_{tag}" if x in LEGAL else (f"add_generic_{tag}" if x in GENW else f"add_other_{tag}")
    if sb < sa: return f"drop_{tag}"
    if len(ta) == len(tb):
        d = [i for i in range(len(ta)) if ta[i] != tb[i]]
        if all(SequenceMatcher(None, ta[i], tb[i]).ratio() >= 0.6 for i in d): return f"typo_{tag}"
        if len(d) == 1: return f"swap_first_{tag}" if d[0] == 0 else (f"swap_generic_{tag}" if tb[d[0]] in GENW else f"swap_descr_{tag}")
    if SequenceMatcher(None, a, b).ratio() >= 0.85: return f"fuzzy_{tag}"
    return f"lowov_{tag}"
sub = (pl.read_csv(f"{R}/output_bucket/{F}/matching_results.tsv", separator="\t", infer_schema=False, quote_char=None).fill_null("")
         .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
         .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"}))
fs = pl.read_parquet(f"{R}/work/errfix/out/fr_suffix_pairs.parquet").select("s1_id", "m_id", "kind")
swr = pl.read_parquet(f"{R}/work/errfix/out/fr_swr_remove.parquet").with_columns(pl.lit(1).alias("swr"))
random.seed(0)
out = {}
for c in ("US", "India", "France"):
    T = f"{R}/work_v4/p1/test/{c}"
    s1 = pl.read_parquet(f"{T}/s1.parquet", columns=["entity_id", "name_core", "addr_text"]).fill_null("")
    ids = s1["entity_id"].to_list()
    if c != "France": ids = random.sample(ids, 200000)
    k = sub.filter(pl.col("s1_id").is_in(ids))
    pool = pl.read_parquet(f"{T}/pool.parquet", columns=["entity_id", "name_core", "addr_text"]).filter(pl.col("entity_id").is_in(k["m_id"].unique().implode())).fill_null("")
    k = (k.join(s1.rename({"entity_id": "s1_id", "name_core": "na", "addr_text": "aa"}), on="s1_id")
          .join(pool.rename({"entity_id": "m_id", "name_core": "nb", "addr_text": "ab"}), on="m_id"))
    if c == "France":
        k = k.join(fs, on=["s1_id", "m_id"], how="left").join(swr, on=["s1_id", "m_id"], how="left")
    else:
        k = k.with_columns(pl.lit(None, pl.Utf8).alias("kind"), pl.lit(None, pl.Int32).alias("swr"))
    lab = []
    for na, nb, aa, ab, kind, sw in k.select("na", "nb", "aa", "ab", "kind", "swr").iter_rows():
        if kind is not None: lab.append("FILS_" + ("replace" if kind.startswith("drops") else "keepall"))
        elif sw: lab.append("SWR_listed")
        else: lab.append(cls(na, nb, aa, ab))
    k = k.with_columns(pl.Series("cls", lab))
    n_s1 = pl.DataFrame({"s1_id": ids}).join(k.group_by("s1_id").len("kk"), on="s1_id", how="left").with_columns(pl.col("kk").fill_null(0))
    h = n_s1["kk"].clip(0, 8).value_counts().sort("kk")
    ref = np.zeros(9); ref[h["kk"].to_numpy()] = h["count"].to_numpy(); ref /= ref.sum()
    per = k.group_by("s1_id", "cls").len("nx").join(n_s1, on="s1_id")
    out[c] = (ref, per, len(ids))
ref = out["US"][0]
A = np.arange(9) * ref; A /= A.sum()                       # size-biased (real copies)
B = np.r_[0, ref[:-1]]; B[-1] += ref[-1]; B /= B.sum()     # shifted by one (extra false pair)
print("ref US file hist:", " ".join(f"{x*100:.2f}" for x in ref))
print("A (real)        :", " ".join(f"{x*100:.2f}" for x in A))
print("B (extra false) :", " ".join(f"{x*100:.2f}" for x in B))
print(f"\n{'country':7s} {'class':22s} {'S1(1x)':>8s} {'per1k':>7s}  P(k=1..8) %                                    f_false  mean_k")
for c in ("US", "India", "France"):
    _, per, n = out[c]
    for cl, g in sorted(per.filter(pl.col("nx") == 1).group_by("cls"), key=lambda x: -x[1].height):
        if g.height < 300: continue
        o = np.zeros(9); hh = g["kk"].clip(0, 8).value_counts(); o[hh["kk"].to_numpy()] = hh["count"].to_numpy(); o /= o.sum()
        x = B[1:] - A[1:]; f = float(np.clip(np.dot(o[1:] - A[1:], x) / np.dot(x, x), -0.5, 1.5))
        print(f"{c:7s} {cl[0] if isinstance(cl, tuple) else cl:22s} {g.height:8,} {g.height / n * 1000:7.1f}  " + " ".join(f"{v*100:5.1f}" for v in o[1:]) + f"   {f:6.2f}  {float((np.arange(9) * o).sum()):.2f}")
