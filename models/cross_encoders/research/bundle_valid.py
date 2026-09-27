"""Bundle the research feature groups (H1-H3, H5) into one VALID table for the stage-3 stacker."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
d = load_base().select("s1_id", "m_id")
a = pl.read_parquet(f"{R}/feats_valid.parquet"); b = pl.read_parquet(f"{R}/feats_h5_valid.parquet")
pl.concat([d, a, b], how="horizontal").write_parquet(f"{R}/research_valid.parquet"); print("VALID bundle", d.height, a.width + b.width)
