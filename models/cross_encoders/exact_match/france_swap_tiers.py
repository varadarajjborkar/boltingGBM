"""Tier the French descriptor-swap removals by independent evidence (not by the copy-count estimate):
T1a: an S1 B with exactly the record's name_core sits at the same house numbers AND same street (unique) -> move r to B
     (VALID: the stack never keeps a record away from such a B; the exact key is the owner 99.99%).
T1b: the record's swapped name is carried by 2+ records at the same numbers (a separate business with its own records).
T2 : remaining non-generic single descriptor swaps (copy-count evidence only).
Writes the tiered list and the expected F change per tier under assumed false shares."""
import gzip, io, os, re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/france_swap"
NUM = re.compile(r"\d+")
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
def jac(a, b): return len(a & b) / max(len(a | b), 1)
GEN_FR = {"france", "services", "groupe", "developpement", "service", "centre", "holding", "associes", "st", "sa"}
t = pl.concat([pl.read_parquet(f"{W}/norm_v2/test_source{s}.parquet", columns=["entity_id", "name_core", "addr_text", "country"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])
t = t.filter(pl.col("country") == "France").fill_null("")
NC = dict(zip(t["entity_id"].to_list(), t["name_core"].to_list())); AD = dict(zip(t["entity_id"].to_list(), t["addr_text"].to_list()))
rk = Counter((n, nk(a)) for n, a, s in zip(t["name_core"].to_list(), t["addr_text"].to_list(), t["src"].to_list()) if s != 1 and n)
S1K = defaultdict(list)
for e, n, a, s in zip(t["entity_id"].to_list(), t["name_core"].to_list(), t["addr_text"].to_list(), t["src"].to_list()):
    if s == 1 and n: S1K[(n, nk(a))].append(e)
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
by = defaultdict(list)
for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()):
    if s in NC:
        for m in ids.split(","):
            if m: by[s].append(m)
rows = []
for s, ms in by.items():
    for m in ms:
        ta, tb = NC[s].split(), NC.get(m, "").split()
        if len(ta) != len(tb) or len(ta) < 2: continue
        d = [i for i in range(len(ta)) if ta[i] != tb[i]]
        if len(d) != 1 or not nk(AD[s]) or nk(AD[s]) != nk(AD[m]): continue
        if jac(street(AD[s]), street(AD[m])) < 0.5: continue
        if SequenceMatcher(None, ta[d[0]], tb[d[0]]).ratio() >= 0.6: continue
        tgt = tb[d[0]]
        if tgt in GEN_FR: continue
        B = [e for e in S1K.get((NC[m], nk(AD[m])), []) if e != s and jac(street(AD[e]), street(AD[m])) >= 0.5]
        tier = "T1a" if len(B) == 1 else ("T1b" if rk[(NC[m], nk(AD[m]))] >= 2 else "T2")
        rows.append((s, m, tgt, tier, B[0] if len(B) == 1 else "", len(ms)))
def F(tp, fn, fp): return 1.0 if tp + fn + fp == 0 else (0.0 if tp == 0 else 1.25 * tp / (1.25 * tp + 0.25 * fn + fp))
nF = 259452
print(f"non-generic French descriptor swaps kept: {len(rows)} ({1000 * len(rows) / nF:.1f} per 1k S1)")
for tier in ("T1a", "T1b", "T2"):
    R = [r for r in rows if r[3] == tier]
    line = f"  {tier}: {len(R)} pairs ({1000 * len(R) / nF:.1f}/1k)"
    for fs in (0.5, 0.7, 0.9):
        g = 0.0
        for s, m, tgt, tr, b, n in R:
            tp = max(n - 1, 1)
            g += fs * (F(tp, 0, 0) - F(tp, 0, 1)) - (1 - fs) * (F(tp + 1, 0, 0) - F(tp, 1, 0))
        line += f" | if {int(fs * 100)}% false: LB {0.149752 * g / nF:+.5f}"
    print(line)
pl.DataFrame({"s1_id": [r[0] for r in rows], "m_id": [r[1] for r in rows], "target_word": [r[2] for r in rows], "tier": [r[3] for r in rows],
              "move_to_s1": [r[4] for r in rows]}).write_parquet(f"{OUT}/fr_swap_tiers.parquet")
print("TIERS_DONE")
