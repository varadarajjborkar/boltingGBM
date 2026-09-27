import os
"""Tagged cross-encoder pack (teammate T4): a per-pair tag prepended to the record side,
e.g. "num up 9; adds holding || <name> | <address>".
num: same (a shared number) / up d / down d (record number = S1 number +- d, d <= 50) / other / missing (record has
no number) / s1 none (S1 has no number). adds: record name_core tokens absent from the S1 name_core (max 3) or none.
Reads $ER_WORK_DIR/ce_pack/{train,valid,test}_pairs.parquet, writes the same files plus a `tag` column to
$ER_WORK_DIR/ce_pack_tag/ (records unchanged: reuse ce_pack's)."""
from pathlib import Path

import polars as pl

W = Path(os.environ.get("ER_WORK_DIR", "work"))
O = W / "ce_pack_tag"
O.mkdir(exist_ok=True)
COLS = ["entity_id", "name_core", "addr_nums"]


def table(split, kind):
    return pl.concat([pl.read_parquet(p, columns=COLS) for p in sorted((W / "p1" / split).glob(f"*/{kind}.parquet"))])


def num_tag(a, b):
    A = [int(x) for x in (a or "").split() if x.isdigit()]
    B = [int(x) for x in (b or "").split() if x.isdigit()]
    if not A:
        return "num s1 none"
    if not B:
        return "num missing"
    if set(A) & set(B):
        return "num same"
    a0 = max(A, key=lambda x: (len(str(x)), x))           # the S1's main (longest) number
    d = min((x - a0 for x in B), key=abs)
    if abs(d) > 50:
        return "num other"
    return f"num up {d}" if d > 0 else f"num down {-d}"


def adds_tag(a, b):
    s = set((a or "").split())
    add = [t for t in dict.fromkeys((b or "").split()) if t not in s][:3]
    return "adds " + (" ".join(add) if add else "none")


for split, files in [("train", ["train_pairs", "valid_pairs"]), ("test", ["test_pairs"])]:
    s1 = table(split, "s1").rename({"entity_id": "s1_id", "name_core": "a_name", "addr_nums": "a_nums"})
    pool = table(split, "pool").rename({"entity_id": "m_id", "name_core": "b_name", "addr_nums": "b_nums"})
    for f in files:
        d = pl.read_parquet(W / "ce_pack" / f"{f}.parquet")
        x = d.join(s1, on="s1_id", how="left").join(pool, on="m_id", how="left")
        miss = x["a_name"].is_null().sum() + x["b_name"].is_null().sum()
        tag = [f"{num_tag(an, bn)}; {adds_tag(a, b)}" for a, an, b, bn in
               zip(x["a_name"].to_list(), x["a_nums"].to_list(), x["b_name"].to_list(), x["b_nums"].to_list())]
        out = d.with_columns(pl.Series("tag", tag))
        out.write_parquet(O / f"{f}.parquet", compression="zstd")
        top = out.group_by(pl.col("tag").str.split(";").list.first()).len().sort("len", descending=True).head(6)
        print(f"{f}: {out.height} pairs, {miss} missing lookups; num tags {dict(zip(top[:, 0].to_list(), top[:, 1].to_list()))}")
        print("  e.g.", out["tag"].head(3).to_list())
