"""Label-free pair features for 3 pair-level hypotheses. Output research/feats_valid.parquet aligned with base_d order."""
import sys, re; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from collections import Counter
d = load_base().select("s1_id", "m_id")
cols = ["entity_id", "business_name", "business_address", "country", "name_core", "addr_empty"]
S1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=cols)
# ---- H3 tables (label-free, from the full S1 table): token doc-freq and exact-name collision counts per country
tokdf = Counter()
for s in S1["name_core"].fill_null(""):
    tokdf.update(set(s.split()))
N1 = S1.height
namecnt = S1.group_by("country", "name_core").len().rename({"len": "s1_namecnt"})
s1 = S1.join(d.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id").join(namecnt, on=["country", "name_core"], how="left")
rec = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=cols) for s in (2, 3)]).join(d.select(pl.col("m_id").alias("entity_id")).unique(), on="entity_id")
rec = rec.join(namecnt, on=["country", "name_core"], how="left")
s1 = s1.rename({c: "a_" + c for c in cols[1:] + ["s1_namecnt"]}).rename({"entity_id": "s1_id"})
rec = rec.rename({c: "b_" + c for c in cols[1:] + ["s1_namecnt"]}).rename({"entity_id": "m_id"})
x = d.join(s1, on="s1_id", how="left", maintain_order="left").join(rec, on="m_id", how="left", maintain_order="left")
log(f"joined {x.height}")
# ---------------- H1: compound house-number agreement
NUMRE = re.compile(r"[A-Za-z]?\d[\dA-Za-z]*(?:\s*[-/]\s*[\dA-Za-z]+)*")
PIN = re.compile(r"^\d{6}$|^\d{5}$")
def ntoks(a):
    out = []
    for m in NUMRE.findall(a or ""):
        t = re.sub(r"\s+", "", m).upper()
        if not any(ch.isdigit() for ch in t): continue
        if re.fullmatch(r"\d+(ST|ND|RD|TH)", t): continue   # ordinals like 2Nd Floor
        out.append(t)
    return out
def comps(t): return [c for c in re.split(r"[-/]", t) if c != ""]
def rel(a, b):
    """3 exact, 2 truncation/prefix (component-level or digit prefix), 1 near (same components but last differs, or one-digit substitution), 0 none"""
    if a == b: return 3
    ca, cb = comps(a), comps(b)
    if len(ca) >= 2 and len(cb) >= 2:
        k = 0
        while k < min(len(ca), len(cb)) and ca[k] == cb[k]: k += 1
        if k == min(len(ca), len(cb)): return 2
        if k >= len(ca) - 1 and len(ca) == len(cb): return 1
        return 0
    if len(ca) == 1 and len(cb) == 1:
        a1, b1 = ca[0], cb[0]
        if len(a1) >= 2 and len(b1) >= 1 and (a1.startswith(b1) or b1.startswith(a1)): return 2
        if len(a1) == len(b1) and sum(p != q for p, q in zip(a1, b1)) == 1: return 1
    return 0
def h1(a, b):
    ta, tb = ntoks(a), ntoks(b)
    ta = [t for t in ta if not PIN.match(t)]; tb = [t for t in tb if not PIN.match(t)]
    if not ta or not tb: return (-1, -1, -1, -1, -1, -1, len(ta), len(tb))
    sb = set(tb)
    n_ex = sum(t in sb for t in ta)
    best = [max(rel(t, u) for u in tb) for t in ta]
    n_near = sum(b_ in (1, 2) for b_ in best); n_none = sum(b_ == 0 for b_ in best)
    lmax = max([len(t) for t in ta if t in sb], default=0)
    prim = best[0]
    multi = [t for t in ta if len(comps(t)) >= 3]
    mexact = (-1 if not multi else float(any(t in sb for t in multi)))
    return (n_ex / len(ta), n_near / len(ta), n_none / len(ta), lmax, prim, mexact, len(ta), len(tb))
A = x["a_business_address"].fill_null("").to_list(); B = x["b_business_address"].fill_null("").to_list()
H1 = np.array([h1(a, b) for a, b in zip(A, B)], dtype=np.float64)
log("H1 done")
# ---------------- H2: name edit signature
LEGAL = {"inc", "llc", "ltd", "co", "corp", "corporation", "company", "private", "pvt", "limited", "llp", "lp", "pc", "pllc", "plc", "group", "services",
         "service", "center", "centre", "enterprises", "enterprise", "associates", "partners", "holdings", "international", "india", "the", "and", "of", "l", "c", "p"}
TITLE = {"mr", "mrs", "ms", "dr", "smt", "shri", "sri", "the"}
def toks(s): return re.findall(r"[a-z0-9]+", (s or "").lower())
def lev2(a, b):
    if abs(len(a) - len(b)) > 2: return False
    m, n = len(a), len(b); prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n] <= 2
LEET = re.compile(r"[A-Za-z][0-9][A-Za-z]|[A-Za-z]'?[0-9]\b|\b[0-9][A-Za-z]{2,}")
def h2(a, b):
    a = a or ""; b = b or ""
    raw_eq = float(a == b); ci_eq = float(a.lower() == b.lower()); ws_eq = float(" ".join(a.lower().split()) == " ".join(b.lower().split()))
    ta, tb = toks(a), toks(b); an_eq = float(ta == tb)
    sa, sb = set(ta), set(tb)
    set_eq = float(sa == sb and ta != tb) if sa else 0.0
    extra = [t for t in tb if t not in sa]; miss = [t for t in ta if t not in sb]
    extra_legal = sum(t in LEGAL for t in extra); extra_other = len(extra) - extra_legal
    miss_legal = sum(t in LEGAL for t in miss); miss_other = len(miss) - miss_legal
    typo = sum(any(lev2(t, u) for u in miss) for t in extra if len(t) >= 3 and t not in LEGAL)
    title = float(len(tb) > 0 and tb[0] in TITLE and (not ta or ta[0] != tb[0]))
    letters = [c for c in b if c.isalpha()]
    case = 0.0 if not letters else (1.0 if all(c.isupper() for c in letters) else (2.0 if all(c.islower() for c in letters) else 0.0))
    leet = float(bool(LEET.search(b)) and not LEET.search(a))
    phone = float(bool(re.search(r"\d{7,}", b)))
    brack = float(bool(re.search(r"[\(\)\[\]]", b)) and not re.search(r"[\(\)\[\]]", a))
    dup = float(len(tb) != len(sb) and len(ta) == len(sa))
    return (raw_eq, ci_eq, ws_eq, an_eq, set_eq, extra_legal, extra_other, miss_legal, miss_other, typo, title, case, leet, phone, brack, dup)
NA = x["a_business_name"].fill_null("").to_list(); NB = x["b_business_name"].fill_null("").to_list()
H2 = np.array([h2(a, b) for a, b in zip(NA, NB)], dtype=np.float64)
log("H2 done")
# ---------------- H3: distinctive-word agreement + name collision counts (full S1 table, label-free)
def idf(t): return float(np.log(N1 / (1 + tokdf.get(t, 0))))
def h3(a, b, ca, cb):
    ta, tb = (a or "").split(), (b or "").split()
    if not ta or not tb: return (-1, -1, -1, -1, -1, ca or 0, cb or 0)
    sa, sb = set(ta), set(tb)
    rare = min(sorted(sa), key=lambda t: tokdf.get(t, 0))
    rare_in = 1.0 if rare in sb else (0.5 if len(rare) >= 4 and any(lev2(rare, u) for u in sb) else 0.0)
    ia = sum(idf(t) for t in sa); ib = sum(idf(t) for t in sb)
    cov_a = sum(idf(t) for t in sa if t in sb) / max(ia, 1e-9); cov_b = sum(idf(t) for t in sb if t in sa) / max(ib, 1e-9)
    xmax = max([idf(t) for t in sb if t not in sa], default=0.0)
    return (rare_in, idf(rare), cov_a, cov_b, xmax, ca or 0, cb or 0)
CA = x["a_name_core"].fill_null("").to_list(); CB = x["b_name_core"].fill_null("").to_list()
KA = x["a_s1_namecnt"].to_list(); KB = x["b_s1_namecnt"].to_list()
H3 = np.array([h3(a, b, ka, kb) for a, b, ka, kb in zip(CA, CB, KA, KB)], dtype=np.float64)
log("H3 done")
n1 = ["hn_frac_exact", "hn_frac_near", "hn_frac_none", "hn_lmax_exact", "hn_prim_rel", "hn_multi_exact", "hn_n_s1", "hn_n_rec"]
n2 = ["ne_raw_eq", "ne_ci_eq", "ne_ws_eq", "ne_an_eq", "ne_set_eq", "ne_extra_legal", "ne_extra_other", "ne_miss_legal", "ne_miss_other", "ne_typo",
      "ne_title", "ne_case", "ne_leet", "ne_phone", "ne_brack", "ne_dup"]
n3 = ["dw_rare_in", "dw_rare_idf", "dw_cov_s1", "dw_cov_rec", "dw_extra_maxidf", "dw_s1_namecnt", "dw_rec_namecnt"]
F = pl.DataFrame({**{n: H1[:, i] for i, n in enumerate(n1)}, **{n: H2[:, i] for i, n in enumerate(n2)}, **{n: H3[:, i] for i, n in enumerate(n3)}})
F.write_parquet(f"{R}/feats_valid.parquet"); log(f"saved {F.shape}")
