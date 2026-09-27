"""Word-by-word translation of native-script India record names into S1 vocabulary (v10, from the engineer's L1 rule).

Table: learned from India train true pairs (romanised record word -> S1 word, aligned by position after stripping
legal fillers such as elelpi / pra li / limtid that records append). Cross-fitted so no train pair is described with a
table learned from itself: pairs of VALID-state S1 use a table learned without the VALID states; other train pairs use
the table learned from the other S1 half (s1_id hash(13) % 2, the stage-1 fold key); test uses all train pairs.
"""
import os
from collections import Counter, defaultdict

import polars as pl

from p1_block import STATE_MERGE
from utils import WORK

# ER_TR_VALID_XFIT=1: VALID S1 use tables cross-fitted by S1 half over all train states (v0 / v1). The default 'nv' table
# excludes the VALID states, so the AP/TS Telugu script is barely covered on VALID while train and test tables cover it.
VALID_XFIT = os.environ.get("ER_TR_VALID_XFIT") == "1"


def _strip(tb, drop):
    s = " ".join(tb)
    for t in drop:
        if s.endswith(" " + t):
            s = s[: -len(t) - 1]
    return s.split()


def _learn(tp):
    extra = Counter()
    for a, b in zip(tp["an"].to_list(), tp["bn"].to_list()):
        ta, tb = (a or "").split(), (b or "").split()
        if len(tb) == len(ta) + 1:
            extra[tb[-1]] += 1
        elif len(tb) == len(ta) + 2:
            extra[" ".join(tb[-2:])] += 1
    drop = sorted({t for t, n in extra.most_common(40) if n >= 200}, key=len, reverse=True)
    cnt = defaultdict(Counter)
    for a, b in zip(tp["an"].to_list(), tp["bn"].to_list()):
        ta, tb = (a or "").split(), _strip((b or "").split(), drop)
        if len(ta) == len(tb) and ta:
            for x, y in zip(tb, ta):
                cnt[x][y] += 1
    dic = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= 2 and n / sum(c.values()) >= 0.5:
            dic[x] = y
    return dic, drop


def true_pairs():
    """India train true pairs with a native-script record: s1_id, an (S1 name_core), bn (record name_core), st, half."""
    d = WORK / "p1" / "train" / "India"
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id", "name_core", "addr_state"])
    pool = pl.scan_parquet(d / "pool.parquet").filter(pl.col("name_non_latin")).select("entity_id", "name_core").collect()
    gt = (pl.read_parquet(WORK / "parquet" / "train_ground_truth.parquet").rename({"source1_entity_id": "s1_id"})
            .filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").rename({"matched_entity_ids": "m_id"}))
    return (gt.join(pool.rename({"entity_id": "m_id", "name_core": "bn"}), on="m_id")
              .join(s1.rename({"entity_id": "s1_id", "name_core": "an"}), on="s1_id")
              .with_columns(pl.col("addr_state").replace(STATE_MERGE).alias("st"), (pl.col("s1_id").hash(13) % 2).alias("half")))


def tables(split, valid_states):
    """{key: (dic, drop)}: test -> {'all'}; train -> {'nv', 'h0', 'h1'} (h0 = learned from half 1, used for half 0)."""
    tp = true_pairs()
    if split == "test":
        return {"all": _learn(tp)}
    nv = tp.filter(~pl.col("st").is_in(valid_states))
    out = {"nv": _learn(nv), "h0": _learn(nv.filter(pl.col("half") == 1)), "h1": _learn(nv.filter(pl.col("half") == 0))}
    if VALID_XFIT:   # VALID pairs get tables from the other S1 half of ALL states (their script covered, as on test)
        out.update({"v0": _learn(tp.filter(pl.col("half") == 1)), "v1": _learn(tp.filter(pl.col("half") == 0))})
    return out


def table_key(split, s1_ids, s1_states, valid_states):
    """Polars expression inputs -> key per S1 row (see tables)."""
    if split == "test":
        return pl.lit("all")
    if VALID_XFIT:
        return (pl.when(s1_states.is_in(valid_states) & (s1_ids.hash(13) % 2 == 0)).then(pl.lit("v0"))
                  .when(s1_states.is_in(valid_states)).then(pl.lit("v1"))
                  .when(s1_ids.hash(13) % 2 == 0).then(pl.lit("h0")).otherwise(pl.lit("h1")))
    return (pl.when(s1_states.is_in(valid_states)).then(pl.lit("nv"))
              .when(s1_ids.hash(13) % 2 == 0).then(pl.lit("h0")).otherwise(pl.lit("h1")))


def translate(name, table):
    dic, drop = table
    return " ".join(dic.get(t, t) for t in _strip((name or "").split(), drop))


def valid_states_india():
    """Merged India VALID states from the model folder's roles.parquet (ER_MODEL_DIR)."""
    from p1_train import M
    r = pl.read_parquet(M / "roles.parquet")
    return list(r.filter((pl.col("country") == "India") & (pl.col("role") == "valid"))["st"])
