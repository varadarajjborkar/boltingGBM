"""Re-check the US/India exact adds with an IDF-weighted street match (generic words like st/ave/road/city names weigh ~0).
Weighted Jaccard over address words (len > 2, digits removed), weights log(N/df) from all test S1 addresses of the country.
Reports how many adds keep a weighted Jaccard >= .5 and the same check on VALID (the exact rule applied to all VALID records
with the unique key, including those the stack assigned) to calibrate precision by weighted-Jaccard bucket."""
import math, os, re
from collections import Counter, defaultdict
import polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
NUM = re.compile(r"\d+")
def words(s): return set(w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2)
def nk(s): return tuple(sorted(int(x) for x in NUM.findall(s or "") if len(x) <= 7))


def idf_table(addrs):
    df = Counter(); n = 0
    for a in addrs:
        n += 1; df.update(words(a))
    return {w: math.log(n / c) for w, c in df.items()}, math.log(n)


def wj(a, b, idf, dflt):
    A, B = words(a), words(b)
    if not A or not B: return 0.0
    num = sum(idf.get(w, dflt) for w in A & B); den = sum(idf.get(w, dflt) for w in A | B)
    return num / den if den else 0.0


def bucket(x): return "wj>=.5" if x >= 0.5 else ("wj.3-.5" if x >= 0.3 else "wj<.3")


# ---- VALID calibration: records with the unique key (same core + numbers, one VALID S1), all of them
t = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=["entity_id", "name_core", "addr_text", "country", "addr_state"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)]).fill_null("")
s1 = t.filter(pl.col("src") == 1)
V = s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
owner = {m: s for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list()) for m in ids.split(",") if m}
AD = dict(zip(t["entity_id"].to_list(), t["addr_text"].to_list()))
for c in ("US", "India"):
    Vc = V.filter(pl.col("country") == c)
    idf, dflt = idf_table(s1.filter(pl.col("country") == c)["addr_text"].to_list())
    g = defaultdict(list)
    for e, n, a in zip(Vc["entity_id"].to_list(), Vc["name_core"].to_list(), Vc["addr_text"].to_list()):
        if n and nk(a): g[(n, nk(a))].append(e)
    r = t.filter((pl.col("src") != 1) & (pl.col("country") == c) & pl.col("addr_state").is_in(["ny"] if c == "US" else ["ap", "ts"]))
    st = Counter()
    for e, n, a in zip(r["entity_id"].to_list(), r["name_core"].to_list(), r["addr_text"].to_list()):
        L = g.get((n, nk(a))) if n and nk(a) else None
        if not L or len(L) != 1: continue
        b = bucket(wj(a, AD[L[0]], idf, dflt)); st[b] += 1; st[(b, "ok")] += owner.get(e) == L[0]
    print(f"VALID {c}: unique-key records by weighted street match: " + "  ".join(f"{b} {st[b]} (prec {st[(b, 'ok')] / max(st[b], 1):.3f})" for b in ("wj>=.5", "wj.3-.5", "wj<.3")))
del t, AD
# ---- TEST adds
t = pl.concat([pl.read_parquet(f"{W}/norm_v2/test_source{s}.parquet", columns=["entity_id", "addr_text", "country"]) for s in (1, 2, 3)]).fill_null("")
AD = dict(zip(t["entity_id"].to_list(), t["addr_text"].to_list()))
adds = pl.read_parquet(f"{W}/usin_fix/usin_exact_adds.parquet")
keep = []
for c in ("US", "India"):
    idf, dflt = idf_table(t.filter(pl.col("entity_id").str.starts_with("S1") & (pl.col("country") == c))["addr_text"].to_list())
    a = adds.filter(pl.col("country") == c); st = Counter()
    for s, m in zip(a["s1_id"].to_list(), a["m_id"].to_list()):
        x = wj(AD[s], AD[m], idf, dflt); b = bucket(x); st[b] += 1
        if x >= 0.5: keep.append((s, m, c, x))
    print(f"TEST {c} adds by weighted street match: " + "  ".join(f"{b} {st[b]}" for b in ("wj>=.5", "wj.3-.5", "wj<.3")))
pl.DataFrame({"s1_id": [k[0] for k in keep], "m_id": [k[1] for k in keep], "country": [k[2] for k in keep], "street_wj": [k[3] for k in keep]}).write_parquet(f"{W}/usin_fix/usin_exact_adds_samestreet.parquet")
print(f"kept {len(keep)}; USIN2_DONE")
