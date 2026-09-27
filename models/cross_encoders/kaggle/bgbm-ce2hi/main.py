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
# ---- bgbm-ce2hi: score the team's HIGH-score pairs (stage-2 p outside the uncertain band, never seen by any cross-encoder)
# with the tagged e5 v2 model (decoy veto candidates). VALID for the veto test, test for the file.
import json, math
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
W = DATA
MD = os.path.dirname(find("ce2_model/ce.pt")[0]); cfg = json.load(open(f"{MD}/cfg.json")); MODEL, MAXLEN = cfg["model"], cfg["maxlen"]
dev, ngpu = "cuda", torch.cuda.device_count()
def raw_texts(split):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet",
                                   columns=["entity_id", "business_name", "business_address", "name_core", "addr_nums"])
                   for s in (1, 2, 3)])
    if t["addr_nums"].dtype == pl.Utf8:
        t = t.with_columns(pl.col("addr_nums").fill_null("").str.extract_all(r"[0-9]+").alias("addr_nums"))
    else:
        t = t.with_columns(pl.col("addr_nums").cast(pl.List(pl.Utf8)))
    return t.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                    .str.to_lowercase().str.replace_all(r"\s+", " ").alias("txt"),
                    pl.col("name_core").fill_null("").alias("nc"), pl.col("addr_nums").alias("nums"))
def _rel(a, b):
    a = [x for x in (a or []) if x]; b = [x for x in (b or []) if x]
    if not a and not b: return "num none"
    if not a: return "num s1 none"
    if not b: return "num missing"
    if a[0] in b: return "num same"
    try:
        d = int(str(b[0]).lstrip("0") or 0) - int(str(a[0]).lstrip("0") or 0)
    except ValueError:
        return "num differs"
    if d == 0: return "num same"
    return f"num up {d}" if d > 0 else f"num down {-d}"
def tags(nca, ncb, na, nb):
    out = []
    for x, y, u, v in zip(nca, ncb, na, nb):
        sa = set(x.split()); add = [w for w in y.split() if w not in sa][:3]
        out.append(_rel(u, v) + ("; adds " + " ".join(add) if add else "") + " || ")
    return out
def attach(df, T=None):
    T = T
    d = (df.join(T.rename({"entity_id": "s1_id", "txt": "ta", "nc": "nca", "nums": "na"}), on="s1_id")
           .join(T.rename({"entity_id": "m_id", "txt": "tb", "nc": "ncb", "nums": "nb"}), on="m_id"))
    tg = tags(d["nca"].to_list(), d["ncb"].to_list(), d["na"].to_list(), d["nb"].to_list())
    d = d.with_columns(pl.Series("tg", tg))
    return d.with_columns((pl.col("tg") + pl.col("tb")).alias("tb")).drop("nca", "ncb", "na", "nb", "tg")

class CE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, ids, am):
        with torch.autocast("cuda", dtype=torch.float16):   # inside forward: DataParallel replicas run in threads
            h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        return self.head((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

base = CE(); base.load_state_dict(torch.load(f"{MD}/ce.pt", map_location="cpu")); base.to(dev).eval()
net = torch.nn.DataParallel(base) if ngpu > 1 else base
tok = AutoTokenizer.from_pretrained(MD)
@torch.inference_mode()
def score(a, b, bs=1024):
    res = np.empty(len(a), dtype=np.float32)
    order = np.argsort(np.fromiter((len(x) + len(z) for x, z in zip(a, b)), dtype=np.int64, count=len(a)))
    for s_ in range(0, len(order), bs):
        idx = order[s_:s_ + bs]
        e = tok(list(a[idx]), list(b[idx]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        res[idx] = torch.sigmoid(net(e["input_ids"].to(dev), e["attention_mask"].to(dev))).cpu().numpy()
    return res
OUT = "/kaggle/working"
def run(allp, done, split, name):
    pr = pl.read_parquet(allp).select("s1_id", "m_id").join(pl.read_parquet(done).select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti")
    T = raw_texts(split); d = attach(pr, T); t1 = time.time()
    d.select("s1_id", "m_id").with_columns(pl.Series("p", score(d["ta"].to_numpy(), d["tb"].to_numpy()))).write_parquet(f"{OUT}/{name}")
    log(f"{name}: {d.height} pairs in {(time.time() - t1) / 60:.1f} min")
fx = lambda n: find(n, prefer="bgbm-exports")[0]
run(fx("best_valid_scored.parquet"), fx("pairs_valid.parquet"), "train", "ce2hi_valid.parquet")
for c in ["US", "India", "France"]:
    run(fx(f"best_test_scored_{c}.parquet"), fx(f"pairs_test_{c}.parquet"), "test", f"ce2hi_test_{c}.parquet")
print("CE2HI_DONE", flush=True)
