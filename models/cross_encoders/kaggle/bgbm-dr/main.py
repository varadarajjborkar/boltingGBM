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
# ---- bgbm-dr (VALID stage): DENSE RETRIEVAL for pairs the team's blocking never compared.
# Pretrained multilingual-e5-small (a retrieval model) embeds VALID S1 and every train S2/S3 record of the VALID states
# (plus stateless records of the country); top-K S1 per record by cosine -> pairs NOT in the team's candidate set ->
# judged by the fine-tuned tagged cross-encoder (ce2) -> added after one-S1-per-record. Measured on the team's VALID.
K = 5
emb_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
emb = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to(dev).eval().half()
@torch.inference_mode()
def embed(texts, prefix, bs=2048):
    out = np.empty((len(texts), 384), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    for s_ in range(0, len(order), bs):
        idx = order[s_:s_ + bs]
        e = emb_tok([prefix + texts[i] for i in idx], truncation=True, max_length=64, padding=True, return_tensors="pt").to(dev)
        h = emb(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).half()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1); v = torch.nn.functional.normalize(v.float(), dim=-1)
        out[idx] = v.half().cpu().numpy()
    return out
@torch.inference_mode()
def topk(q, d, k=K, bs=4096):
    D = torch.from_numpy(d).to(dev); I = np.empty((len(q), k), dtype=np.int64); S = np.empty((len(q), k), dtype=np.float32)
    for s_ in range(0, len(q), bs):
        Q = torch.from_numpy(q[s_:s_ + bs]).to(dev); sc = Q @ D.T
        v, i = sc.topk(min(k, D.shape[0]), dim=1); I[s_:s_ + bs] = i.cpu().numpy(); S[s_:s_ + bs] = v.float().cpu().numpy()
    return I, S
def txt(df):
    return (df["business_name"].fill_null("") + " | " + df["business_address"].fill_null("")).str.to_lowercase().to_list()
cols = ["entity_id", "business_name", "business_address", "country", "addr_state"]
s1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=cols).with_columns(pl.col("addr_state").fill_null("").replace("ts", "ap"))
s1v = s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) | ((pl.col("country") == "India") & (pl.col("addr_state") == "ap")))
rec = pl.concat([pl.read_parquet(f"{W}/norm_v2/train_source{s}.parquet", columns=cols) for s in (2, 3)]).with_columns(pl.col("addr_state").fill_null("").replace("ts", "ap"))
groups = [("US", "ny"), ("India", "ap")]
pairs = []
t0 = time.time()
for c, st in groups:
    a = s1v.filter((pl.col("country") == c) & (pl.col("addr_state") == st))
    r = rec.filter((pl.col("country") == c) & pl.col("addr_state").is_in([st, ""]))
    ea = embed(txt(a), "passage: "); er = embed(txt(r), "query: ")
    I, S = topk(er, ea)
    aid = np.array(a["entity_id"].to_list()); rid = np.array(r["entity_id"].to_list())
    pairs.append(pl.DataFrame({"s1_id": aid[I.ravel()], "m_id": np.repeat(rid, I.shape[1]), "cos": S.ravel()}))
    log(f"{c}/{st}: {a.height} S1, {r.height} records -> {I.size} retrieved pairs ({(time.time() - t0) / 60:.1f} min)")
dr = pl.concat(pairs)
team = pl.read_parquet(find("best_valid_scored.parquet", prefer="bgbm-exports")[0]).select("s1_id", "m_id", "y", "p")
new = dr.join(team.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti")
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id"))).join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
missing = gt.join(team.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti")
new = new.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1_id", "m_id"], how="left").with_columns(pl.col("y").fill_null(0))
log(f"new pairs (not in the team set): {new.height}; true among them {new['y'].sum()} of {missing.height} VALID true pairs the team never scored")
T = raw_texts("train"); d = attach(new.select("s1_id", "m_id"), T)
d = d.select("s1_id", "m_id").with_columns(pl.Series("ce", score(d["ta"].to_numpy(), d["tb"].to_numpy())))
new = new.join(d, on=["s1_id", "m_id"], how="left").with_columns(pl.col("ce").fill_null(0.0))
new.write_parquet(f"{OUT}/dr_valid_new_pairs.parquet")
for t_ in (0.9, 0.95, 0.98, 0.99, 0.995):
    x = new.filter(pl.col("ce") >= t_); log(f"  ce >= {t_}: {x.height} pairs, true {x['y'].sum()}, precision {x['y'].mean() if x.height else 0:.3f}")
sys.path[:0] = [SRC, f"{CODE}/tools"]; os.environ.update({"ER_WORK_DIR": "/tmp/work", "ER_NORM_SUBDIR": "norm_v2"})
link_split("train", ["US", "India"])
from quick_residual import macro, universe
s1u, truth = universe()
base = macro(team.select("s1_id", "m_id", "p"), s1u, truth)
log(f"team stack_s3a2 VALID {json.dumps({k: round(float(v), 5) for k, v in base.items()})}")
res = {"base": base}
for t_ in (0.95, 0.98, 0.99, 0.995):
    add = new.filter(pl.col("ce") >= t_).select("s1_id", "m_id", pl.col("ce").alias("p"))
    r = macro(pl.concat([team.select("s1_id", "m_id", "p"), add]), s1u, truth); res[f"dr_{t_}"] = r
    log(f"dense retrieval + ce >= {t_}: {r['macro'] - base['macro']:+.5f} | halves {r['h0'] - base['h0']:+.5f} {r['h1'] - base['h1']:+.5f} | US {r['US'] - base['US']:+.5f} India {r['India'] - base['India']:+.5f}")
json.dump(res, open(f"{OUT}/dr_result.json", "w"), default=float, indent=1)
print("DR_DONE", flush=True)
