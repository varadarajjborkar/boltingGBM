import os
"""Zero-shot MiniLM-L6 similarity for the stage-3 uncertain pairs (same pair lists as the char net: cn_valid / cn_test).
Each record's name and address is embedded once (all-MiniLM-L6-v2, normalised); per pair: cosine of names, cosine of
addresses (NaN if either address is empty). Writes stage-3 extra files (s1_id, m_id, p) under work_v4/minilm/:
mn_valid / ma_valid.parquet and mn_test_<c> / ma_test_<c>.parquet.   Usage (spot 6): python zeroshot.py"""
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)
A = Path(os.environ.get("ER_ROOT", "."))
P, O = A / "work" / "parquet", A / "work_v4" / "minilm"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
model.max_seq_length = 64


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def enc(strings, tag):
    out, t0, B = [], time.time(), 200_000
    for s in range(0, len(strings), B):
        out.append(model.encode(strings[s:s + B], batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False).astype(np.float16))
        log(f"{tag}: {min(s + B, len(strings))}/{len(strings)} ({(min(s + B, len(strings))) / (time.time() - t0):.0f}/s)")
    return np.vstack(out)


def run(split, pairs, tag):
    ids = pl.concat([pairs["s1_id"], pairs["m_id"]]).unique().to_frame("entity_id")
    t = pl.concat([pl.read_parquet(P / f"{split}_source{i}.parquet", columns=["entity_id", "business_name", "business_address"])
                   for i in (1, 2, 3)]).join(ids, on="entity_id", how="semi")
    log(f"{tag}: {pairs.height} pairs, {t.height} records")
    addr = t["business_address"].fill_null("").str.replace_all(r"(?i)<?null>?", "").str.strip_chars(" ,")
    En = enc(t["business_name"].fill_null("").to_list(), f"{tag} names")
    Ea = enc(addr.to_list(), f"{tag} addresses")
    empty = (addr.str.len_chars() == 0).to_numpy()
    idx = {e: i for i, e in enumerate(t["entity_id"].to_list())}
    a = np.array([idx[s] for s in pairs["s1_id"].to_list()])
    b = np.array([idx[s] for s in pairs["m_id"].to_list()])
    cn = np.einsum("ij,ij->i", En[a].astype(np.float32), En[b].astype(np.float32))
    ca = np.einsum("ij,ij->i", Ea[a].astype(np.float32), Ea[b].astype(np.float32))
    ca[empty[a] | empty[b]] = np.nan
    return pairs.with_columns(pl.Series("mn", cn), pl.Series("ma", ca))


v = run("train", pl.read_parquet(O / "cn_valid.parquet").select("s1_id", "m_id"), "VALID")
for k in ("mn", "ma"):
    v.select("s1_id", "m_id", pl.col(k).alias("p")).write_parquet(O / f"{k}_valid.parquet")
log("VALID written")
t = run("test", pl.read_parquet(O / "cn_test.parquet").select("s1_id", "m_id", "country"), "test")
for c in ["France", "India", "US"]:
    for k in ("mn", "ma"):
        t.filter(pl.col("country") == c).select("s1_id", "m_id", pl.col(k).alias("p")).write_parquet(O / f"{k}_test_{c}.parquet")
log("test written; done")
