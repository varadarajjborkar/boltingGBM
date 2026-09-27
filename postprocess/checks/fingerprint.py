"""Generator-fingerprint tool: compare unlabelled TEST candidate pairs with labelled VALID pairs, per S1, cell by cell.

Purpose
    Find pair types (cells) that are denser on test than on VALID, e.g. look-alike distractors, and measure how much
    denser (rho = inflation of negatives). VALID states are NY (US) and AP+TS (India); the test side is restricted to
    S1 of the SAME states, because address styles differ a lot between states (an all-state comparison is confounded).

Subcommands
    build    descriptors for VALID scored pairs (labels, v2b p, v4_rule-style decision) and for test candidates of the
             VALID states (v2b p on test, v4_rule predictions). Writes <out>/desc_valid.parquet, desc_test.parquet.
    cells    per-1000-S1 table by any descriptor keys: VALID count, positive rate, test count, rho, predictions.
    weights  rho by (country, group, score band) and per-pair weights for postprocess/checks/dense_valid.py.
    allstate k(country, group) and k_pos: corrects the VALID-state weights to ALL test states. Positives on all-state
             test come from TRAIN labels (20% S1 sample, all states) x VALID recall; predictions from a 20% all-state
             test sample with the v2b scores (p >= 0.75). Needs `build` first. Writes <out>/allstate_k.parquet.

Descriptors (all computed from normalised strings, identical on both sides)
    num_rel    house numbers as MULTISETS (duplicates kept, unlike addr_nums): eq, sub (one side has extra numbers
               only), missing (a side has none), shift_pure (set-level record-minus-S1 difference in {3,4,5,7,9,11,13,21},
               the v4_rule), shift_12 (set-level shift hit, closest difference +1/+2), shift_ms_only (shift hit only
               visible with duplicates kept, e.g. '58 14 14' -> '58 14 19'), neg_small (-25..-1), typo, near_other, far
    name_rel   eq, perm, space, add_1/add_2+, drop_1/drop_2+, typo1, swap1, fuzzy_hi, first_same, fuzzy_mid, diff
    legal_rel  both_empty, eq, one_empty, diff
    street_rel eq / hi / mid / lo / empty (Jaccard of non-numeric address tokens without state codes)
    state_rel  eq / diff / empty

Usage
    python postprocess/checks/fingerprint.py build   [--out work/advisor/fingerprint]
    python postprocess/checks/fingerprint.py cells   [--keys num_rel,name_rel] [--min 2] [--out ...]
    python postprocess/checks/fingerprint.py weights [--out ...]
    python postprocess/checks/fingerprint.py allstate [--out ...]
Limits: about 25 s and 2 GB for build (3 threads); cells/weights take seconds.
"""
import argparse
import re
from collections import Counter

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = "."
VALID_TAB = f"{ROOT}/work/advisor/valid_pairs_model_v2b.parquet"
TEST_P = f"{ROOT}/work_fr/p1/test/{{c}}/stage2_scored_model_v2b.parquet"       # v4 model scores on test
TEST_CANDS = f"{ROOT}/work/p1/test/{{c}}/stage2_scored_model_v2s.parquet"       # stage-2 candidate set on test
TEST_PRED = f"{ROOT}/output_bucket/v4_rule/matching_results.tsv"
N_VALID = {"US": 102314, "India": 73301}
STATES = {"US": ["ny"], "India": ["ap", "ts"]}
NUM = re.compile(r"\d+")
OFF = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
PURE = {3, 4, 5, 7, 9, 11, 13, 21}
COLS = ["entity_id", "name_core", "name_legal", "addr_text", "addr_state"]


# ---------------------------------------------------------------- descriptors
def _nums(s):
    return [int(x) for x in NUM.findall(s or "") if len(x) <= 7]


def _is_drop(a, b):
    a, b = str(a), str(b)
    if abs(len(a) - len(b)) != 1:
        return False
    L, S = (a, b) if len(a) > len(b) else (b, a)
    return any(L[:k] + L[k + 1:] == S for k in range(len(L)))


def _is_sub_nonlast(a, b):
    a, b = str(a), str(b)
    if len(a) != len(b):
        return False
    dif = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return len(dif) == 1 and dif[0] < len(a) - 1


def num_rel(sa, sb):
    A, B = _nums(sa), _nums(sb)
    if not A or not B:
        return "missing"
    ca, cb = Counter(A), Counter(B)
    if ca == cb:
        return "eq"
    SA, SB = set(A) - set(B), set(B) - set(A)
    if any((b - a) in OFF for a in SA for b in SB):
        d = min(((abs(b - a), b - a) for a in SA for b in SB))[1]
        return "shift_pure" if d in PURE else "shift_12"
    MA, MB = list((ca - cb).elements()), list((cb - ca).elements())
    if any((b - a) in OFF for a in MA for b in MB):
        return "shift_ms_only"
    if not MA or not MB:
        return "sub"
    d = min(((abs(b - a), b - a) for a in MA for b in MB))[1]
    if -25 <= d <= -1:
        return "neg_small"
    if any(_is_drop(a, b) or _is_sub_nonlast(a, b) for a in MA for b in MB):
        return "typo"
    return "near_other" if abs(d) <= 100 else "far"


def name_rel(a, b):
    a, b = a or "", b or ""
    if a == b:
        return "eq"
    ta, tb = a.split(), b.split()
    if sorted(ta) == sorted(tb):
        return "perm"
    if a.replace(" ", "") == b.replace(" ", ""):
        return "space"
    A, B = set(ta), set(tb)
    if A and A < B:
        return "add_" + ("1" if len(B - A) == 1 else "2+")
    if B and B < A:
        return "drop_" + ("1" if len(A - B) == 1 else "2+")
    if len(A - B) == 1 and len(B - A) == 1:
        x, = A - B
        y, = B - A
        return "typo1" if Levenshtein.normalized_similarity(x, y) >= 0.5 else "swap1"
    ts = fuzz.token_set_ratio(a, b)
    if ts >= 85:
        return "fuzzy_hi"
    if ta and tb and ta[0] == tb[0]:
        return "first_same"
    return "fuzzy_mid" if ts >= 60 else "diff"


def legal_rel(a, b):
    a, b = a or "", b or ""
    if not a and not b:
        return "both_empty"
    if a == b:
        return "eq"
    return "one_empty" if (not a or not b) else "diff"


def street_rel(sa, sb, st):
    A = {t for t in (sa or "").split() if not t.isdigit() and t not in st and len(t) > 1}
    B = {t for t in (sb or "").split() if not t.isdigit() and t not in st and len(t) > 1}
    if not A or not B:
        return "empty"
    j = len(A & B) / len(A | B)
    return "eq" if j == 1 else ("hi" if j >= 0.6 else ("mid" if j >= 0.3 else "lo"))


def describe(d, states):
    """d has an, bn (names), al, bl (legal), aa, ba (addresses), as_, bs_ (states)."""
    st = set(states)
    aa, ba = d["aa"].to_list(), d["ba"].to_list()
    return d.with_columns(
        pl.Series("num_rel", [num_rel(a, b) for a, b in zip(aa, ba)]),
        pl.Series("name_rel", [name_rel(a, b) for a, b in zip(d["an"].to_list(), d["bn"].to_list())]),
        pl.Series("legal_rel", [legal_rel(a, b) for a, b in zip(d["al"].to_list(), d["bl"].to_list())]),
        pl.Series("street_rel", [street_rel(a, b, st) for a, b in zip(aa, ba)]),
        pl.when((pl.col("as_") == "") | (pl.col("bs_") == "")).then(pl.lit("empty"))
          .when(pl.col("as_") == pl.col("bs_")).then(pl.lit("eq")).otherwise(pl.lit("diff")).alias("state_rel"),
    ).drop("an", "bn", "al", "bl", "aa", "ba", "as_", "bs_")


def _ren(df, p):
    return df.rename({"name_core": f"{p}n", "name_legal": f"{p}l", "addr_text": f"{p}a", "addr_state": f"{p}s_"})


# ---------------------------------------------------------------- build
def build(out):
    import os
    os.makedirs(out, exist_ok=True)
    t = pl.read_parquet(VALID_TAB, columns=["s1_id", "m_id", "y", "p", "pred", "country", "src"]).filter(pl.col("p").is_not_null())
    res = []
    for c in ["US", "India"]:
        tc = t.filter(pl.col("country") == c)
        s1 = pl.scan_parquet(f"{ROOT}/work/p1/train/{c}/s1.parquet").select(COLS).join(
            tc.select(pl.col("s1_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect()
        pool = pl.scan_parquet(f"{ROOT}/work/p1/train/{c}/pool.parquet").select(COLS).join(
            tc.select(pl.col("m_id").unique().alias("entity_id")).lazy(), on="entity_id", how="semi").collect()
        states = [x for x in pl.concat([s1["addr_state"], pool["addr_state"]]).unique().to_list() if x]
        d = tc.join(_ren(s1, "a").rename({"entity_id": "s1_id"}), on="s1_id").join(_ren(pool, "b").rename({"entity_id": "m_id"}), on="m_id")
        res.append(describe(d, states))
    v = pl.concat(res).with_columns(
        (pl.col("pred") & (pl.col("num_rel") != "shift_pure")).alias("pred_rule"))       # v4_rule on VALID
    v.write_parquet(f"{out}/desc_valid.parquet")
    print("VALID pairs", v.height)

    s1all = pl.scan_parquet(f"{ROOT}/work/norm/test_source1.parquet").select(COLS + ["country"])
    ids = s1all.filter(pl.col("addr_state").is_in(sum(STATES.values(), []))).select(pl.col("entity_id").alias("source1_entity_id")).collect()
    pred = (pl.scan_csv(TEST_PRED, separator="\t").join(ids.lazy(), on="source1_entity_id", how="semi").collect()
              .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m_id"})
              .filter(pl.col("m_id").is_not_null() & (pl.col("m_id") != "")).with_columns(pl.lit(True).alias("pred_rule")))
    res = []
    for c in ["US", "India"]:
        s1 = s1all.filter(pl.col("addr_state").is_in(STATES[c])).collect()
        pr = pl.read_parquet(TEST_CANDS.format(c=c), columns=["s1_id", "m_id"]).join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
        pc = pred.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
        pr = pr.join(pc, on=["s1_id", "m_id"], how="full", coalesce=True).with_columns(pl.col("pred_rule").fill_null(False))
        pr = pr.join(pl.read_parquet(TEST_P.format(c=c)).join(pr.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="semi"),
                     on=["s1_id", "m_id"], how="left")
        mids = pr.select(pl.col("m_id").unique().alias("entity_id")).lazy()
        pool = pl.concat([pl.scan_parquet(f"{ROOT}/work/norm/test_source{k}.parquet").select(COLS).join(mids, on="entity_id", how="semi")
                          .with_columns(pl.lit(f"S{k}").alias("src")).collect() for k in (2, 3)])
        states = [x for x in pl.concat([s1["addr_state"], pool["addr_state"]]).unique().to_list() if x]
        d = pr.join(_ren(s1.drop("country"), "a").rename({"entity_id": "s1_id"}), on="s1_id").join(_ren(pool, "b").rename({"entity_id": "m_id"}), on="m_id")
        res.append(describe(d, states).with_columns(pl.lit(c).alias("country"), pl.lit(s1.height).alias("n_s1")))
        print("test", c, "S1", s1.height, "pairs", d.height, "predicted", int(d["pred_rule"].sum()))
    pl.concat(res).write_parquet(f"{out}/desc_test.parquet")


# ---------------------------------------------------------------- cells
def cells(out, keys, minn):
    V = pl.read_parquet(f"{out}/desc_valid.parquet")
    T = pl.read_parquet(f"{out}/desc_test.parquet")
    for c in ["US", "India"]:
        nv = N_VALID[c]
        v = V.filter(pl.col("country") == c).group_by(keys).agg(
            (pl.len() * 1000 / nv).alias("v_n"), pl.col("y").mean().alias("v_pos_rate"),
            ((pl.col("y") == 1).sum() * 1000 / nv).alias("_vpos"), ((pl.col("y") == 0).sum() * 1000 / nv).alias("_vneg"),
            ((pl.col("pred_rule")) & (pl.col("y") == 1)).sum().mul(1000 / nv).alias("v_tp"),
            ((pl.col("pred_rule")) & (pl.col("y") == 0)).sum().mul(1000 / nv).alias("v_fp"))
        t = T.filter(pl.col("country") == c)
        n = t["n_s1"][0]
        t = t.group_by(keys).agg((pl.len() * 1000 / n).alias("t_n"), (pl.col("pred_rule").sum() * 1000 / n).alias("t_pred"))
        x = (t.join(v, on=keys, how="full", coalesce=True).fill_null(0)
               .with_columns(((pl.col("t_n") - pl.col("_vpos")) / pl.col("_vneg")).alias("rho"),
                             (pl.col("t_pred") - pl.col("v_tp")).alias("t_excess_pred"))
               .filter((pl.col("t_n") >= minn) | (pl.col("v_n") >= minn)).drop("_vpos", "_vneg"))
        print(f"=== {c} (test S1 of {STATES[c]}: {n}; VALID S1 {nv}) per 1000 S1, sorted by excess test predictions")
        print(x.with_columns(pl.col(pl.Float64).round(3)).sort("t_excess_pred", descending=True))


# ---------------------------------------------------------------- weights
GROUP = pl.when(pl.col("num_rel").is_in(["shift_pure", "shift_12", "shift_ms_only"])).then(pl.col("num_rel")).otherwise(pl.lit("other")).alias("g")
BAND = pl.col("p").cut([0.05, 0.3, 0.75, 0.95], labels=["a", "b", "c", "d", "e"]).cast(pl.Utf8).alias("band")


def weights(out):
    V = pl.read_parquet(f"{out}/desc_valid.parquet")
    T = pl.read_parquet(f"{out}/desc_test.parquet")
    rows = []
    for c in ["US", "India"]:
        nv = N_VALID[c]
        v = V.filter(pl.col("country") == c).with_columns(GROUP, BAND).group_by("g", "band").agg(
            ((pl.col("y") == 1).sum() * 1000 / nv).alias("v_pos"), ((pl.col("y") == 0).sum() * 1000 / nv).alias("v_neg"))
        t = T.filter((pl.col("country") == c) & pl.col("p").is_not_null())
        n = t["n_s1"][0]
        t = t.with_columns(GROUP, BAND).group_by("g", "band").agg((pl.len() * 1000 / n).alias("t_n"))
        rows.append(v.join(t, on=["g", "band"]).with_columns(
            ((pl.col("t_n") - pl.col("v_pos")) / pl.col("v_neg")).clip(1.0, 5.0).alias("w"), pl.lit(c).alias("country")))
    W = pl.concat(rows).select("country", "g", "band", "w", "v_neg", "t_n")
    # band e (p > 0.95): too few negatives to estimate; reuse band d of the same group
    W = pl.concat([W.filter(pl.col("band") != "e"), W.filter(pl.col("band") == "d").with_columns(pl.lit("e").alias("band"))])
    W.select("country", "g", "band", "w").write_parquet(f"{out}/dense_w_table.parquet")
    pw = V.with_columns(GROUP, BAND).join(W.select("country", "g", "band", "w"), on=["country", "g", "band"], how="left").select(
        "s1_id", "m_id", "country", "g", pl.col("w").fill_null(1.85))
    pw.write_parquet(f"{out}/dense_pair_weights.parquet")
    print(W.filter(pl.col("band") != "e").sort("country", "g", "band").with_columns(pl.col(pl.Float64).round(3)))
    print(f"wrote {out}/dense_w_table.parquet and {out}/dense_pair_weights.parquet ({pw.height} pairs)")


# ---------------------------------------------------------------- all-state correction
def allstate(out):
    W = f"{ROOT}/work"
    groups = ["shift_pure", "shift_12", "shift_ms_only"]
    # train true pairs, 20% S1 sample, all states
    s1 = (pl.scan_parquet(f"{W}/norm/train_source1.parquet").filter(pl.col("country").is_in(["US", "India"]) & (pl.col("entity_id").hash(3) % 5 == 0))
            .select("entity_id", "country", "addr_text").collect())
    gt = (pl.scan_parquet(f"{W}/parquet/train_ground_truth.parquet").rename({"source1_entity_id": "s1_id"})
            .join(s1.lazy().select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi").collect()
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"matched_entity_ids": "m_id"})
            .filter(pl.col("m_id").is_not_null() & (pl.col("m_id") != "")))
    ids = gt.select(pl.col("m_id").unique().alias("entity_id")).lazy()
    pool = pl.concat([pl.scan_parquet(f"{W}/norm/train_source{k}.parquet").select("entity_id", "addr_text").join(ids, on="entity_id", how="semi").collect() for k in (2, 3)])
    d = gt.join(s1.rename({"entity_id": "s1_id", "addr_text": "aa"}), on="s1_id").join(pool.rename({"entity_id": "m_id", "addr_text": "ba"}), on="m_id")
    d = d.with_columns(pl.Series("num_rel", [num_rel(a, b) for a, b in zip(d["aa"].to_list(), d["ba"].to_list())]))
    n_tr = s1.group_by("country").len()
    # test: 20% all-state S1 sample, stage-2 candidates with v2b scores
    t1 = (pl.scan_parquet(f"{W}/norm/test_source1.parquet").filter(pl.col("country").is_in(["US", "India"]) & (pl.col("entity_id").hash(3) % 5 == 0))
            .select("entity_id", "country", "addr_text").collect())
    rows = []
    V = pl.read_parquet(f"{out}/desc_valid.parquet")
    TS = pl.read_parquet(f"{out}/desc_test.parquet")
    for c in ["US", "India"]:
        tc = t1.filter(pl.col("country") == c)
        pr = pl.read_parquet(TEST_P.format(c=c)).filter(pl.col("p") >= 0.75).join(tc.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
        mids = pr.select(pl.col("m_id").unique().alias("entity_id")).lazy()
        tp = pl.concat([pl.scan_parquet(f"{W}/norm/test_source{k}.parquet").select("entity_id", "addr_text").join(mids, on="entity_id", how="semi").collect() for k in (2, 3)])
        x = pr.join(tc.rename({"entity_id": "s1_id", "addr_text": "aa"}), on="s1_id").join(tp.rename({"entity_id": "m_id", "addr_text": "ba"}), on="m_id")
        x = x.with_columns(pl.Series("num_rel", [num_rel(a, b) for a, b in zip(x["aa"].to_list(), x["ba"].to_list())]))
        ntr = n_tr.filter(pl.col("country") == c)["len"][0]
        for g in groups:
            v = V.filter((pl.col("country") == c) & (pl.col("num_rel") == g))
            pos = v.filter(pl.col("y") == 1)
            rec = pos.filter(pl.col("p") >= 0.75).height / max(pos.height, 1)
            v_tp = v.filter((pl.col("p") >= 0.75) & (pl.col("y") == 1)).height * 1000 / N_VALID[c]
            v_fp = max(v.filter((pl.col("p") >= 0.75) & (pl.col("y") == 0)).height * 1000 / N_VALID[c], 0.01)
            ts = TS.filter((pl.col("country") == c) & (pl.col("num_rel") == g))
            ts_pred = ts.filter(pl.col("p") >= 0.75).height * 1000 / TS.filter(pl.col("country") == c)["n_s1"][0]
            ta_pred = x.filter(pl.col("num_rel") == g).height * 1000 / tc.height
            true_all = d.filter((pl.col("country") == c) & (pl.col("num_rel") == g)).height * 1000 / ntr
            k = (max(ta_pred - rec * true_all, 0.01) / v_fp) / (max(ts_pred - v_tp, 0.01) / v_fp)
            k_pos = (rec * true_all) / max(v_tp, 0.01)
            rows.append((c, g, round(k, 3), round(k_pos, 3), round(v_tp, 2), round(v_fp, 2), round(ts_pred, 2), round(ta_pred, 2), round(rec * true_all, 2)))
    df = pl.DataFrame(rows, schema=["country", "g", "k", "k_pos", "v_tp", "v_fp", "test_vstates_pred", "test_all_pred", "test_all_tp_est"], orient="row")
    print(df)
    df.select("country", "g", "k", "k_pos").write_parquet(f"{out}/allstate_k.parquet")
    print(f"wrote {out}/allstate_k.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "cells", "weights", "allstate"])
    ap.add_argument("--out", default=f"{ROOT}/work/advisor/fingerprint")
    ap.add_argument("--keys", default="num_rel")
    ap.add_argument("--min", type=float, default=2.0)
    a = ap.parse_args()
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
    if a.cmd == "build":
        build(a.out)
    elif a.cmd == "cells":
        cells(a.out, a.keys.split(","), a.min)
    elif a.cmd == "weights":
        weights(a.out)
    else:
        allstate(a.out)
