"""Robustness of the copy-count test to the baseline. For VALID (labels, team decision) and France (file), own-name kept
copies per S1 for: no-swap S1 with >= 1 kept record (baseline A), all no-swap S1 incl. zero kept (baseline B), and S1 whose
first swap is generic / other-single / other-twin / first. Deficits vs A and vs B; implied French true share per class =
France deficit / VALID deficit (VALID swaps are ~100% true)."""
import gzip, io, os, re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); E = f"{W}/e5hyb2"
NUM = re.compile(r"\d+")
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
GEN = {"US": {"center", "services", "service", "partners"}, "FR": {"france", "services", "groupe", "developpement", "service", "centre", "holding", "associes"}}
def swapinfo(na, nb, aa, ab):
    ta, tb = na.split(), nb.split()
    if len(ta) != len(tb) or len(ta) < 2: return None
    d = [i for i in range(len(ta)) if ta[i] != tb[i]]
    if len(d) != 1 or not nk(aa) or nk(aa) != nk(ab): return None
    sa, sb = street(aa), street(ab)
    if not sa or len(sa & sb) / len(sa | sb) < 0.5: return None
    if SequenceMatcher(None, ta[d[0]], tb[d[0]]).ratio() >= 0.6: return None
    return tb[d[0]], d[0]
def tabs(split, country=None):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "name_core", "addr_text", "country", "addr_state"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)]).fill_null("")
    if country: t = t.filter(pl.col("country") == country)
    NC = dict(zip(t["entity_id"].to_list(), t["name_core"].to_list())); AD = dict(zip(t["entity_id"].to_list(), t["addr_text"].to_list()))
    rk = Counter((n, nk(a)) for n, a, s in zip(t["name_core"].to_list(), t["addr_text"].to_list(), t["src"].to_list()) if s != 1 and n)
    return t, NC, AD, rk
def groups(all_s1, kept, NC, AD, rk, gen):
    by = defaultdict(list)
    for s, m in kept: by[s].append(m)
    g = defaultdict(list)
    for s in all_s1:
        ms = by.get(s, []); own = sum(1 for m in ms if NC.get(m) == NC.get(s))
        sw = [(m, swapinfo(NC.get(s, ""), NC.get(m, ""), AD.get(s, ""), AD.get(m, ""))) for m in ms]
        sw = [(m, x) for m, x in sw if x]
        if not sw:
            g["B_all_noswap"].append(own)
            if ms: g["A_kept_noswap"].append(own)
            continue
        m, (tgt, pos) = sw[0]
        cls = "first" if pos == 0 else ("generic" if tgt in gen else ("other_twin" if rk[(NC[m], nk(AD[m]))] >= 2 else "other_single"))
        g[cls].append(own)
    return {k: (sum(v) / len(v), len(v)) for k, v in g.items()}
BASE = {"shift_pure": 1.01, "shift_ms_only": 1.01, "shift_12": 0.95, "shift_12|corr": 0.98, "missing|corr": 0.98}; T = 0.75
def thr_expr():
    key = pl.when(pl.col("sup") == "corr").then(pl.col("grp") + "|corr").otherwise(pl.col("grp")); e = pl.lit(T)
    for g_, x in BASE.items():
        if "|" in g_: e = pl.when(key == g_).then(pl.lit(x)).otherwise(e)
    for g_, x in BASE.items():
        if "|" not in g_: e = pl.when((pl.col("grp") == g_) & ~key.is_in([k for k in BASE if "|" in k])).then(pl.lit(x)).otherwise(e)
    return e
flags = pl.read_parquet(f"{E}/rule_flags_pairs.parquet", columns=["s1_id", "m_id", "dtok", "split"])
v = pl.read_parquet(f"{E}/stage3_valid_oof_pairs.parquet").filter(pl.col("p").is_not_null()).join(flags.filter(pl.col("split") == "valid").drop("split"), on=["s1_id", "m_id"], how="left")
v = v.with_columns(pl.col("p").rank("ordinal", descending=True).over("m_id").alias("r"))
dv = v.filter((pl.col("r") == 1) & (pl.col("p") >= thr_expr()) & ~pl.col("dtok").fill_null(False))
t, NC, AD, rk = tabs("train")
s1 = t.filter(pl.col("src") == 1)
V = s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list()
GV = groups(V, list(zip(dv["s1_id"].to_list(), dv["m_id"].to_list())), NC, AD, rk, GEN["US"])
del t, NC, AD, rk
t, NC, AD, rk = tabs("test", "France")
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
kf = [(s, m) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()) if s in NC for m in ids.split(",") if m]
GF = groups(t.filter(pl.col("src") == 1)["entity_id"].to_list(), kf, NC, AD, rk, GEN["FR"])
print(f"{'class':14s} | {'VALID mean (n)':>18s} {'dA':>7s} {'dB':>7s} | {'FRANCE mean (n)':>18s} {'dA':>7s} {'dB':>7s} | {'true share via A':>16s} {'via B':>6s}")
for k in ("A_kept_noswap", "B_all_noswap", "generic", "other_single", "other_twin", "first"):
    vm, vn = GV.get(k, (0, 0)); fm, fn = GF.get(k, (0, 0))
    vA, vB = GV["A_kept_noswap"][0] - vm, GV["B_all_noswap"][0] - vm
    fA, fB = GF["A_kept_noswap"][0] - fm, GF["B_all_noswap"][0] - fm
    ts = "" if k.startswith(("A_", "B_")) else f"{fA / vA if vA else float('nan'):16.2f} {fB / vB if vB else float('nan'):6.2f}"
    print(f"{k:14s} | {vm:10.3f} ({vn:6d}) {vA:7.3f} {vB:7.3f} | {fm:10.3f} ({fn:6d}) {fA:7.3f} {fB:7.3f} | {ts}")
print("CALIB2_DONE")
