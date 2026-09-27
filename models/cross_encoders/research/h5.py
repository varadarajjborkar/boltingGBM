"""H5: co-location (address-collision) counts from the full S1 table (label-free), then residual tests for H5 and ALL."""
import sys, re; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from collections import Counter
d = load_base()
S1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "business_address", "country", "addr_text", "addr_empty"])
addrcnt = S1.filter(~pl.col("addr_empty")).group_by("country", "addr_text").len().rename({"len": "s1_addrcnt"})
NUMRE = re.compile(r"[A-Za-z]?\d[\dA-Za-z]*(?:\s*[-/]\s*[\dA-Za-z]+)*")
def key_toks(a):
    out = set()
    for m in NUMRE.findall(a or ""):
        t = re.sub(r"\s+", "", m).upper()
        if len([c for c in re.split(r"[-/]", t) if c]) >= 3 or len(t) >= 6: out.add(t)
    return out
kc = Counter()
for c, a in zip(S1["country"].to_list(), S1["business_address"].fill_null("").to_list()):
    for t in key_toks(a): kc[(c, t)] += 1
log(f"key token table {len(kc)}")
s1 = S1.join(addrcnt, on=["country", "addr_text"], how="left").select(pl.col("entity_id").alias("s1_id"), pl.col("s1_addrcnt").fill_null(0).cast(pl.Float64))
rec = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=["entity_id", "business_address", "country", "addr_text", "addr_empty"]) for s in (2, 3)])
rec = rec.join(d.select(pl.col("m_id").alias("entity_id")).unique(), on="entity_id").join(addrcnt, on=["country", "addr_text"], how="left")
rk = [min([kc.get((c, t), 0) for t in key_toks(a)], default=-1) for c, a in zip(rec["country"].to_list(), rec["business_address"].fill_null("").to_list())]
rec = rec.select(pl.col("entity_id").alias("m_id"), pl.col("s1_addrcnt").fill_null(0).cast(pl.Float64).alias("rec_addrcnt"), pl.Series("rec_keytok_cnt", rk, dtype=pl.Float64))
x = d.select("s1_id", "m_id").join(s1, on="s1_id", how="left", maintain_order="left").join(rec, on="m_id", how="left", maintain_order="left").fill_null(-1)
x.select("s1_addrcnt", "rec_addrcnt", "rec_keytok_cnt").write_parquet(f"{R}/feats_h5_valid.parquet")
H5 = {c: x[c].to_numpy() for c in ("s1_addrcnt", "rec_addrcnt", "rec_keytok_cnt")}
print(x.with_columns(d["y"]).group_by(pl.col("s1_addrcnt").clip(0, 5)).agg(pl.len(), pl.col("y").mean()).sort("s1_addrcnt"))
run_test(d, H5, "H5 co-location counts")
F = pl.read_parquet(f"{R}/feats_valid.parquet")
ALL = {**{c: F[c].to_numpy() for c in F.columns}, **H5}
run_test(d, ALL, "ALL H1+H2+H3+H5")
H2 = {c: F[c].to_numpy() for c in F.columns if c.startswith("ne_")}
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
for sd in (8, 9):
    A = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(np.column_stack(base_X(d)), yy, grp, seed=sd))))
    B = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(np.column_stack(base_X(d) + list(H2.values())), yy, grp, seed=sd))))
    log(f"H2 seed {sd}: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
print("H5_DONE", flush=True)
