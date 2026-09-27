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
# ---- bgbm-drtest: DENSE RETRIEVAL on TEST, exactly the VALID-approved recipe (VALID: +0.00036 both halves, 0.74 judged
# pairs per S1, precision 0.923): records the team left unassigned (stack p < 0.725 everywhere) -> e5-small retrieval of the
# top-2 S1 of the same country+state (stateless records: whole country) -> keep pairs NOT in the team's candidate set that
# SHARE A HOUSE NUMBER -> tagged cross-encoder (ce2) judge -> accept ce >= 0.998.
import re
K = 2; THR = 0.998
NUM = re.compile(r"\d+")
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
        v = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1).clamp(min=1)).float(), dim=-1)
        out[idx] = v.half().cpu().numpy()
    return out
@torch.inference_mode()
def topk(q, d, k=K, bs=4096):
    D = torch.from_numpy(d).to(dev); I = np.empty((len(q), min(k, D.shape[0])), dtype=np.int64); S = np.empty(I.shape, dtype=np.float32)
    for s_ in range(0, len(q), bs):
        Q = torch.from_numpy(q[s_:s_ + bs]).to(dev); v, i = (Q @ D.T).topk(I.shape[1], dim=1)
        I[s_:s_ + bs] = i.cpu().numpy(); S[s_:s_ + bs] = v.float().cpu().numpy()
    return I, S
cols = ["entity_id", "business_name", "business_address", "country", "addr_state", "addr_text"]
s1 = pl.read_parquet(f"{W}/norm_v2/test_source1.parquet", columns=cols).with_columns(pl.col("addr_state").fill_null("").replace("ts", "ap"))
rec = pl.concat([pl.read_parquet(f"{W}/norm_v2/test_source{s}.parquet", columns=cols) for s in (2, 3)]).with_columns(pl.col("addr_state").fill_null("").replace("ts", "ap"))
fx = lambda n: find(n, prefer="bgbm-exports")[0]
team = pl.concat([pl.read_parquet(fx(f"best_test_scored_{c}.parquet")).select("s1_id", "m_id", "p") for c in ("US", "India", "France")])
assigned = team.filter(pl.col("p") >= 0.725).select("m_id").unique()
rec = rec.join(assigned, left_on="entity_id", right_on="m_id", how="anti")
log(f"test S1 {s1.height}; unassigned records {rec.height}")
txt = lambda df: (df["business_name"].fill_null("") + " | " + df["business_address"].fill_null("")).str.to_lowercase().to_list()
parts = []; t0 = time.time()
for c in ["US", "India", "France"]:
    sc_ = s1.filter(pl.col("country") == c); rc = rec.filter(pl.col("country") == c)
    for st in sorted(sc_["addr_state"].unique().to_list()):
        a = sc_.filter(pl.col("addr_state") == st)
        r = rc.filter(pl.col("addr_state") == st) if st else rc.filter(pl.col("addr_state") == "")
        if a.height == 0 or r.height == 0: continue
        I, S = topk(embed(txt(r), "query: "), embed(txt(a), "passage: "))
        aid = np.array(a["entity_id"].to_list()); rid = np.array(r["entity_id"].to_list())
        parts.append(pl.DataFrame({"s1_id": aid[I.ravel()], "m_id": np.repeat(rid, I.shape[1]), "cos": S.ravel()}))
    # stateless records of the country against the whole country
    r = rc.filter(pl.col("addr_state") == "")
    if r.height:
        I, S = topk(embed(txt(r), "query: "), embed(txt(sc_), "passage: "))
        aid = np.array(sc_["entity_id"].to_list()); rid = np.array(r["entity_id"].to_list())
        parts.append(pl.DataFrame({"s1_id": aid[I.ravel()], "m_id": np.repeat(rid, I.shape[1]), "cos": S.ravel()}))
    log(f"{c}: retrieval done ({(time.time() - t0) / 60:.1f} min)")
dr = pl.concat(parts).unique(["s1_id", "m_id"])
new = dr.join(team.select("s1_id", "m_id"), on=["s1_id", "m_id"], how="anti")
AT = dict(zip(s1["entity_id"].to_list(), s1["addr_text"].fill_null("").to_list())); AT.update(zip(rec["entity_id"].to_list(), rec["addr_text"].fill_null("").to_list()))
share = [bool(set(NUM.findall(AT.get(a, ""))) & set(NUM.findall(AT.get(b, "")))) for a, b in zip(new["s1_id"].to_list(), new["m_id"].to_list())]
new = new.filter(pl.Series(share))
log(f"new pairs sharing a house number: {new.height} ({new.height / s1.height:.3f} per S1)")
T = raw_texts("test"); d = attach(new.select("s1_id", "m_id"), T)
d = d.select("s1_id", "m_id").with_columns(pl.Series("ce", score(d["ta"].to_numpy(), d["tb"].to_numpy())))
new = new.join(d, on=["s1_id", "m_id"], how="left").with_columns(pl.col("ce").fill_null(0.0))
cty = s1.select(pl.col("entity_id").alias("s1_id"), "country")
new = new.join(cty, on="s1_id", how="left")
new.write_parquet(f"{OUT}/dr_test_judged.parquet")
acc = new.filter(pl.col("ce") >= THR)
acc.write_parquet(f"{OUT}/dr_test_accepted.parquet")
for c in ["US", "India", "France"]:
    n_s1 = s1.filter(pl.col("country") == c).height
    j = new.filter(pl.col("country") == c); a_ = acc.filter(pl.col("country") == c)
    log(f"{c}: judged {j.height} ({j.height / max(n_s1, 1):.3f} per S1), accepted {a_.height} ({1000 * a_.height / max(n_s1, 1):.1f} per 1000 S1)")
print("DRTEST_DONE", flush=True)
