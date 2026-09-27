"""Per target word copy-count test for French kept one-word swaps (non-first word or first word), plus the drop/move list.
true_share(word) ~= deficit(word) / deficit_true(class) with deficit = own-name mean(no-swap S1) - own-name mean(S1 whose
swap has that target word); deficit_true from VALID (single 0.371, twin 1.10, first 0.568). Builds exports-ready lists:
DROP = kept swap pairs whose word-level true share < .5 (move to S1 B when a unique S1 with the swapped name sits at the
same numbers + street), and reports expected France macro gain with the per-word true shares."""
import gzip, io, os, re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/france_swap"; os.makedirs(OUT, exist_ok=True)
NUM = re.compile(r"\d+")
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))
def street(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
GEN_FR = {"france", "services", "groupe", "developpement", "service", "centre", "holding", "associes"}
DT = {"single": 0.371, "twin": 1.10, "first": 0.568}


def swapinfo(na, nb, aa, ab):
    ta, tb = na.split(), nb.split()
    if len(ta) != len(tb) or len(ta) < 2: return None
    d = [i for i in range(len(ta)) if ta[i] != tb[i]]
    if len(d) != 1 or not nk(aa) or nk(aa) != nk(ab): return None
    sa, sb = street(aa), street(ab)
    if not sa or len(sa & sb) / len(sa | sb) < 0.5: return None
    if SequenceMatcher(None, ta[d[0]], tb[d[0]]).ratio() >= 0.6: return None
    return tb[d[0]], d[0]


t = pl.concat([pl.read_parquet(f"{W}/norm_v2/test_source{s}.parquet", columns=["entity_id", "name_core", "addr_text", "country"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])
t = t.filter(pl.col("country") == "France").fill_null("")
NC = dict(zip(t["entity_id"].to_list(), t["name_core"].to_list())); AD = dict(zip(t["entity_id"].to_list(), t["addr_text"].to_list()))
rk = Counter((n, nk(a)) for n, a, s in zip(t["name_core"].to_list(), t["addr_text"].to_list(), t["src"].to_list()) if s != 1 and n)
s1 = t.filter(pl.col("src") == 1); S1KEY = defaultdict(list)
for e, n, a in zip(s1["entity_id"].to_list(), s1["name_core"].to_list(), s1["addr_text"].to_list()):
    if n: S1KEY[(n, nk(a))].append((e, street(a)))
mr = pl.read_csv(io.BytesIO(gzip.open(f"{W}/e5hyb2_matching_results.tsv.gz").read()), separator="\t", quote_char=None, infer_schema=False).fill_null("")
by = defaultdict(list)
for s, ids in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()):
    if s in NC:
        for m in ids.split(","):
            if m: by[s].append(m)
own_noswap = []; per = defaultdict(list); pairs = []
for s, ms in by.items():
    own = sum(1 for m in ms if NC.get(m) == NC[s])
    sw = [(m, swapinfo(NC[s], NC.get(m, ""), AD[s], AD.get(m, ""))) for m in ms]
    sw = [(m, x) for m, x in sw if x]
    if not sw: own_noswap.append(own); continue
    for m, (tgt, pos) in sw:
        cls = "first" if pos == 0 else ("twin" if rk[(NC[m], nk(AD[m]))] >= 2 else "single")
        gen = tgt in GEN_FR
        pairs.append((s, m, tgt, cls, gen, len(ms)))
    m, (tgt, pos) = sw[0]
    cls = "first" if pos == 0 else ("twin" if rk[(NC[m], nk(AD[m]))] >= 2 else "single")
    per[(tgt if tgt not in GEN_FR else "<generic>", cls)].append(own)
base = sum(own_noswap) / len(own_noswap)
print(f"France no-swap S1 own-name mean {base:.3f} (n={len(own_noswap)})")
share = {}
rows = sorted(per.items(), key=lambda kv: -len(kv[1]))
print(f"{'word':16s} {'class':7s} {'S1':>6s} {'own mean':>8s} {'deficit':>8s} {'true share':>10s}")
for (w, cls), L in rows:
    dfc = base - sum(L) / len(L); ts = max(0.0, min(1.0, dfc / DT[cls]))
    share[(w, cls)] = (ts, len(L))
    if len(L) >= 150: print(f"{w:16s} {cls:7s} {len(L):6d} {sum(L) / len(L):8.3f} {dfc:8.3f} {ts:10.2f}")
# pooled estimate for small words per class
pool = defaultdict(list)
for (w, cls), L in per.items():
    if len(L) < 150 and w != "<generic>": pool[cls].extend(L)
for cls, L in pool.items():
    dfc = base - sum(L) / len(L); ts = max(0.0, min(1.0, dfc / DT[cls])); share[("<small>", cls)] = (ts, len(L))
    print(f"{'<small words>':16s} {cls:7s} {len(L):6d} {sum(L) / len(L):8.3f} {dfc:8.3f} {ts:10.2f}")
# drop / move lists and expected gain (F0.5 per S1 with the S1's current kept count)
drop, move = [], []; gain = 0.0
def F(tp, fn, fp): return 1.0 if tp + fn + fp == 0 else (0.0 if tp == 0 else 1.25 * tp / (1.25 * tp + 0.25 * fn + fp))
for s, m, tgt, cls, gen, nkept in pairs:
    if gen: continue
    key = (tgt, cls) if (tgt, cls) in share and share[(tgt, cls)][1] >= 150 else ("<small>", cls)
    ts = share.get(key, (1.0, 0))[0]
    if ts >= 0.5: continue
    # expected gain of removing this pair from S1 s: with prob (1-ts) it is an FP, with prob ts a TP
    tp = max(nkept - 1, 1)
    gain += (1 - ts) * (F(tp, 0, 0) - F(tp, 0, 1)) - ts * (F(tp + 1, 0, 0) - F(tp, 1, 0))
    B = [e for e, sb in S1KEY.get((NC[m], nk(AD[m])), []) if e != s and street(AD[m]) and len(street(AD[m]) & sb) / max(len(street(AD[m]) | sb), 1) >= 0.5]
    (move if len(B) == 1 else drop).append((s, m, tgt, cls, ts, B[0] if len(B) == 1 else ""))
nF = 259452
print(f"pairs to remove: {len(drop) + len(move)} ({1000 * (len(drop) + len(move)) / nF:.1f} per 1k French S1); of which re-assignable to a unique same-address S1 with the swapped name: {len(move)}")
print(f"expected France macro change from removal only: {gain / nF:+.4f} -> LB {0.149752 * gain / nF:+.5f}")
pl.DataFrame({"s1_id": [x[0] for x in drop + move], "m_id": [x[1] for x in drop + move], "target_word": [x[2] for x in drop + move], "class": [x[3] for x in drop + move],
              "est_true_share": [x[4] for x in drop + move], "move_to_s1": [x[5] for x in drop + move]}).write_parquet(f"{OUT}/fr_swap_remove.parquet")
print("PERWORD_DONE")
