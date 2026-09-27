"""Stage 1: normalise names and addresses of every source file (train + test).

Output: work/norm/<split>_<source>.parquet, one row per record.
Checkpointed per file: an existing output is skipped unless --force. Kill-safe (atomic writes).
Usage: python build_normalized.py [--force] [--only train_source1,...] [--workers 7]
"""
import argparse
import multiprocessing as mp

import polars as pl

from normalize import normalize_address, normalize_name, phonetic_key
from utils import WORK, announce_pid, done, log, resource_guard, rss_gb, timed, write_parquet_atomic

FILES = ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]


def _names_chunk(names):
    out = []
    for n in names:
        r = normalize_name(n)
        out.append((n, r["core"], r["legal"], r["alts"], r["is_domain"], r["has_alt"], r["non_latin"],
                    phonetic_key(r["core"])))
    return out


def _addr_chunk(pairs):
    out = []
    for a, c in pairs:
        r = normalize_address(a, c)
        out.append((a, c, r["text"], r["state"], r["nums"], r["empty"], r["native_state"]))
    return out


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def normalize_file(name: str, workers: int):
    df = pl.read_parquet(WORK / "parquet" / f"{name}.parquet")
    names = df["business_name"].unique().to_list()
    pairs = df.select("business_address", "country").unique().rows()
    # workers are recycled so per-process string caches cannot grow without bound (protects RAM)
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=20) as pool:
        nres = [r for part in pool.imap(_names_chunk, _chunks(names, 20000)) for r in part]
        ares = [r for part in pool.imap(_addr_chunk, _chunks(pairs, 20000)) for r in part]
    ndf = pl.DataFrame(nres, schema=["business_name", "name_core", "name_legal", "name_alts", "name_is_domain",
                                     "name_has_alt", "name_non_latin", "name_phon"], orient="row")
    adf = pl.DataFrame(ares, schema=["business_address", "country", "addr_text", "addr_state", "addr_nums",
                                     "addr_empty", "addr_native_state"], orient="row")
    out = (df.join(ndf, on="business_name", how="left")
             .join(adf, on=["business_address", "country"], how="left"))
    assert out.height == df.height, "join changed row count"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    announce_pid("build_normalized")
    files = a.only.split(",") if a.only else FILES
    for f in files:
        dst = WORK / "norm" / f"{f}.parquet"
        if done(dst, a.force):
            log(f"skip {f} (exists)")
            continue
        resource_guard(f"normalize {f}")
        with timed(f"normalize {f}"):
            write_parquet_atomic(normalize_file(f, a.workers), dst)
        log(f"main process RSS {rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
