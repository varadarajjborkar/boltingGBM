"""Copy-count verdict for the team's French 'and fils / associes / compagnie' pocket (work/fr_suffix_pairs.parquet), per kind.
Own-name kept copies per S1 (stack_e5hyb2 file). Baselines: A = S1 with >= 1 kept record and no pair of any tested class;
B = all such S1 incl. zero kept. VALID calibration (team decision, labels ~100% true) with the SAME baselines for the
analogous classes: adds_word (record keeps all S1 words and adds some), other (multi-word difference), swap (one word
replaced). France reference classes computed alongside: adds_word (non-suffix), swap descriptor (non-suffix)."""
import gzip, io, os, re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); E = f"{W}/e5hyb2"
NUM = re.compile(r"\d+")
GEN = {"center", "services", "service", "partners", "france", "groupe", "developpement", "centre", "holding", "associes", "st", "sa"}
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))


def rel(na, nb, aa, ab):
    """class of a kept non-exact pair, or None if exact / not same-number."""
    if na == nb or not na or not nb: return None
    ta, tb = na.split(), nb.split(); sa, sb = set(ta), set(tb)
    if sa < sb: return "adds_word"
    if len(ta) == len(tb) and sum(x != y for x, y in zip(ta, tb)) == 1:
        i = [k for k in range(len(ta)) if ta[k] != tb[k]][0]
        if SequenceMatcher(None, ta[i], tb[i]).ratio() >= 0.6: return "typo"
        return "swap_generic" if tb[i] in GEN else "swap_descr"
    if sa == sb: return "reorder"
    if sb < sa: return "drops_word"
    return "other"


def tabs(split, country=None):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "name_core", "addr_text", "country", "addr_state"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)]).fill_null("")
    if country: t = t.filter(pl.col("country") == country)
    return t, dict(zip(t["entity_id"].to_list(), t["name_core"].to_list())), dict(zip(t["entity_id"].to_list(), t["addr_text"].to_list()))


TESTED = {"adds_word", "other", "swap_descr", "swap_generic", "drops_word"}
def groups(all_s1, kept, NC, AD, special=None):
    """special: dict (s1, m) -> label that overrides the class (suffix kinds)."""
    by = defaultdict(list)
    for s, m in kept: by[s].append(m)
    g = defaultdict(list)
    for s in all_s1:
        ms = by.get(s, []); own = sum(1 for m in ms if NC.get(m) == NC.get(s))
        labs = []
        for m in ms:
            if special and (s, m) in special: labs.append(special[(s, m)]); continue
            c = rel(NC.get(s, ""), NC.get(m, ""), AD.get(s, ""), AD.get(m, ""))
            if c in TESTED: labs.append(c)
        if not labs:
            g["B"].append(own)
            if ms: g["A"].append(own)
            continue
        lab = next((l for l in labs if l.startswith("suffix")), labs[0])
        g[lab].append(own)
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
t, NC, AD = tabs("train")
s1 = t.filter(pl.col("src") == 1)
V = s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list()
GV = groups(V, list(zip(dv["s1_id"].to_list(), dv["m_id"].to_list())), NC, AD)
del t, NC, AD
t, NC, AD = tabs("test", "France")
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
kf = [(s, m) for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()) if s in NC for m in ids.split(",") if m]
sfx = pl.read_parquet(f"{W}/fr_suffix_pairs.parquet")
special = {(s, m): "suffix|" + k.split()[0] for s, m, k in zip(sfx["s1_id"].to_list(), sfx["m_id"].to_list(), sfx["kind"].to_list())}
kfs = set(kf); print(f"suffix pairs present in the stack_e5hyb2 France rows: {sum(1 for p in special if p in kfs)} of {len(special)}")
GF = groups(t.filter(pl.col("src") == 1)["entity_id"].to_list(), kf, NC, AD, special)
dA = lambda G, k: G["A"][0] - G[k][0]; dB = lambda G, k: G["B"][0] - G[k][0]
print(f"baselines: VALID A {GV['A'][0]:.3f} B {GV['B'][0]:.3f} | France A {GF['A'][0]:.3f} B {GF['B'][0]:.3f}")
print(f"{'class':22s} {'VALID dA / dB (n)':>24s} | {'FRANCE dA / dB (n)':>24s}")
for k in ("adds_word", "other", "swap_descr", "swap_generic", "drops_word", "suffix|keeps", "suffix|drops", "suffix|replaces"):
    vv = f"{dA(GV, k):.3f} / {dB(GV, k):.3f} ({GV[k][1]})" if k in GV else "-"
    ff = f"{dA(GF, k):.3f} / {dB(GF, k):.3f} ({GF[k][1]})" if k in GF else "-"
    print(f"{k:22s} {vv:>24s} | {ff:>24s}")
def share(fk, vk):
    return f"{dA(GF, fk) / dA(GV, vk):.2f} (A) / {dB(GF, fk) / dB(GV, vk):.2f} (B)"
print("implied true share:")
print(f"   France adds_word (sanity, R2 says true) vs VALID adds_word: {share('adds_word', 'adds_word')}")
print(f"   France swap_descr (reference)          vs VALID swap_descr: {share('swap_descr', 'swap_descr')}")
print(f"   suffix|keeps  vs VALID adds_word : {share('suffix|keeps', 'adds_word')}")
print(f"   suffix|drops  vs VALID other     : {share('suffix|drops', 'other')}")
print(f"   suffix|drops  vs VALID swap_descr: {share('suffix|drops', 'swap_descr')}")
print("SUFFIX_DONE")
