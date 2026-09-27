"""Bayesian tie-breaker for address-less records (team R1). Generator prior J(n2, n3) = P(business has n2 copies in S2 and
n3 in S3), from train truth. For a tied S1 j with observed address-bearing copies (a2, a3) and an address-less record r in
source s: LR_j = J(a + e_s) / J(a) (one more copy, in r's source). P(owner = j) = LR_j / sum_k LR_k (ties of k S1).
Evaluated: TRAIN with TRUE address-bearing sibling counts (ceiling), and VALID with PREDICTED counts (stack_s3a2 kept records,
split by source) for ties whose S1 are all VALID. Reports top-1 accuracy vs chance, and precision / coverage when the top
posterior >= .72 (break-even) or >= .80, and the implied VALID gain."""
import os
from collections import Counter, defaultdict
import numpy as np, polars as pl
W = os.environ.get("ER_WORK_DIR", "work")
t = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=["entity_id", "name_core", "country", "addr_empty", "addr_state"]).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)]).fill_null("")
SRC = dict(zip(t["entity_id"].to_list(), t["src"].to_list())); AE = dict(zip(t["entity_id"].to_list(), t["addr_empty"].to_list()))
s1 = t.filter(pl.col("src") == 1)
NC = dict(zip(s1["entity_id"].to_list(), s1["name_core"].to_list())); CT = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
ties = defaultdict(list)
for e, n, c in zip(s1["entity_id"].to_list(), s1["name_core"].to_list(), s1["country"].to_list()):
    if n: ties[(c, n)].append(e)
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
cl = {s: [m for m in ids.split(",") if m] for s, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].fill_null("").to_list())}
owner = {m: s for s, ms in cl.items() for m in ms}
J = Counter()
for s, ms in cl.items():
    J[(min(sum(1 for m in ms if SRC[m] == 2), 8), min(sum(1 for m in ms if SRC[m] == 3), 8))] += 1
N = sum(J.values()); Jp = lambda a2, a3: (J[(min(a2, 8), min(a3, 8))] + 0.5) / N
def post(comp, counts, src):
    lr = []
    for j in comp:
        a2, a3 = counts(j)
        lr.append(Jp(a2 + (src == 2), a3 + (src == 3)) / Jp(a2, a3))
    lr = np.array(lr); return lr / lr.sum()
def evaluate(label, counts, restrict=None):
    st = Counter()
    for m, s in owner.items():
        if not AE.get(m): continue
        comp = ties.get((CT[s], NC[s]), [])
        if len(comp) < 2: continue
        if restrict is not None and not all(j in restrict for j in comp): continue
        p = post(comp, lambda j: counts(j, m), SRC[m]); i = int(np.argmax(p)); top = p[i]; ok = comp[i] == s
        st["n"] += 1; st["chance"] += 1 / len(comp); st["top1"] += ok
        for thr in (0.72, 0.8, 0.9):
            if top >= thr: st[(thr, "n")] += 1; st[(thr, "ok")] += ok
    n = st["n"]
    print(f"{label}: address-less records in ties {n}; top-1 {st['top1'] / n:.3f} vs chance {st['chance'] / n:.3f}")
    for thr in (0.72, 0.8, 0.9):
        k = st[(thr, "n")]
        print(f"   posterior >= {thr}: covers {k / n:.3f} of ties ({k}), precision {st[(thr, 'ok')] / max(k, 1):.3f}")
    return st
def true_counts(j, m):
    ms = [x for x in cl.get(j, []) if x != m and not AE.get(x)]
    return sum(1 for x in ms if SRC[x] == 2), sum(1 for x in ms if SRC[x] == 3)
evaluate("TRAIN, true address-bearing sibling counts (ceiling)", true_counts)
V = set(s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"])))["entity_id"].to_list())
v = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "p")
d = v.sort("p", descending=True).unique("m_id", keep="first").filter(pl.col("p") >= 0.75)
pk = defaultdict(list)
for s, m in zip(d["s1_id"].to_list(), d["m_id"].to_list()): pk[s].append(m)
def pred_counts(j, m):
    ms = [x for x in pk.get(j, []) if x != m and not AE.get(x)]
    return sum(1 for x in ms if SRC[x] == 2), sum(1 for x in ms if SRC[x] == 3)
evaluate("VALID, predicted counts (both tied S1 in VALID)", pred_counts, V)
print("BAYES_DONE")
