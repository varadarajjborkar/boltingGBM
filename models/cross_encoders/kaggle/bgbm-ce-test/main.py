# Shared setup (v2): code from the bgbm-code dataset, data from bgbm-fetch, block outputs found by search.
import glob, os, shutil, subprocess, sys, time
def find(pattern, prefer=None):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    if prefer:
        hits = [h for h in hits if prefer in h] or hits
    return hits
REQ = find("requirements.txt", prefer="bgbm-code")[0]
CODE = os.path.dirname(os.path.dirname(os.path.dirname(REQ)))
SRC = f"{CODE}/src"
DATA = os.path.dirname(os.path.dirname(find("work/norm_v2/train_source1.parquet")[0]))
W = "/tmp/work"
os.makedirs(f"{W}/p1", exist_ok=True)
for d in ["norm_v2", "parquet"]:
    if not os.path.exists(f"{W}/{d}"):
        os.symlink(f"{DATA}/{d}", f"{W}/{d}")
def link_split(split, countries):
    for c in countries:
        src = os.path.dirname(find(f"p1/{split}/{c}/cands.parquet")[0])
        dst = f"{W}/p1/{split}/{c}"
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not os.path.exists(f"{dst}/{f}"):
                os.symlink(f"{src}/{f}", f"{dst}/{f}")
# ---- bgbm-ce-test: score pairs with the fine-tuned cross-encoder from bgbm-ce.
# Jobs: k1 test uncertain pairs (bgbm-merge) and, if an exports dataset is attached, the team's VALID/test pair lists.
import json, math
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
MD = os.path.dirname(find("ce_model/ce.pt")[0])
cfg = json.load(open(f"{MD}/cfg.json")); MAXLEN = cfg["maxlen"]
dev, ngpu = "cuda", torch.cuda.device_count()
log(f"GPUs {ngpu} | model {cfg['model']} from {MD}")
class CE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(cfg["model"])
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, ids, am):
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        return self.head((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)
base = CE(); base.load_state_dict(torch.load(f"{MD}/ce.pt", map_location="cpu")); base.to(dev).eval()
net = torch.nn.DataParallel(base) if ngpu > 1 else base
tok = AutoTokenizer.from_pretrained(MD)
_T = {}
def texts(split):
    if split not in _T:
        t = pl.concat([pl.read_parquet(f"{DATA}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "business_address"])
                       for s in (1, 2, 3)])
        _T[split] = t.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                             .str.to_lowercase().str.replace_all(r"\s+", " ").alias("txt"))
    return _T[split]
@torch.inference_mode()
def score(pairs, split, bs=1024):
    T = texts(split)
    d = (pairs.select("s1_id", "m_id").unique().join(T.rename({"entity_id": "s1_id", "txt": "ta"}), on="s1_id")
              .join(T.rename({"entity_id": "m_id", "txt": "tb"}), on="m_id"))
    a, b = d["ta"].to_numpy(), d["tb"].to_numpy()
    res = np.empty(len(a), dtype=np.float32)
    order = np.argsort(np.fromiter((len(x) + len(z) for x, z in zip(a, b)), dtype=np.int64, count=len(a)))
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        e = tok(list(a[idx]), list(b[idx]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        res[idx] = torch.sigmoid(net(e["input_ids"].to(dev), e["attention_mask"].to(dev))).cpu().numpy()
    log(f"  scored {len(a)} pairs in {(time.time() - t0) / 60:.1f} min")
    return d.select("s1_id", "m_id").with_columns(pl.Series("p", res))
OUT = "/kaggle/working"
for c in ["France", "India", "US"]:
    hits = find(f"p1/test/{c}/stage2_scored_model_k1.parquet")
    if hits:
        sc = pl.read_parquet(hits[0]).filter((pl.col("p") >= 0.01) & (pl.col("p") <= 0.999))
        log(f"k1 test {c}: {sc.height} uncertain pairs")
        score(sc, "test").write_parquet(f"{OUT}/ce_test_{c}.parquet")
for f in find("exports/pairs_*.parquet"):          # team pair lists: pairs_valid.parquet, pairs_test_<Country>.parquet
    name = os.path.basename(f).replace("pairs_", "ce_team_")
    split = "train" if "valid" in name else "test"
    pr = pl.read_parquet(f)
    log(f"team {os.path.basename(f)}: {pr.height} pairs ({split})")
    score(pr, split).write_parquet(f"{OUT}/{name}")
print("CE_TEST_DONE", sorted(os.listdir(OUT)), flush=True)
