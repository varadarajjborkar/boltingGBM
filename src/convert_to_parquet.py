"""Convert the raw challenge TSVs to Parquet for fast loading.

Quoting is disabled on purpose: names/addresses may contain stray quote characters,
and a quote-aware parser would silently merge lines. Row counts are checked against
a raw line count so no record is lost.
"""
import sys
from pathlib import Path

import polars as pl

from utils import DATA as _DATA, WORK as _WORK
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else _DATA
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else _WORK / "parquet"
OUT.mkdir(parents=True, exist_ok=True)

for split in ("train", "test"):
    for f in sorted((DATA / split).glob("*.tsv")):
        df = pl.read_csv(
            f, separator="\t", quote_char=None, infer_schema=False, encoding="utf8",
        ).fill_null("")
        with open(f, "rb") as fh:
            n_lines = sum(1 for _ in fh) - 1  # minus header
        assert df.height == n_lines, f"{f.name}: parsed {df.height} rows but file has {n_lines} lines"
        df.write_parquet(OUT / f"{f.stem}.parquet")
        print(f"{f.name}: {df.height:,} rows, cols={df.columns}")
