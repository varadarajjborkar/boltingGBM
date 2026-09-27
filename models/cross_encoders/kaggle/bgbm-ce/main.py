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
# ---- bgbm-ce: cross-encoder (multilingual-e5-small, MIT, 118M params) on the stage-2 training states (s2fit, US+India),
# scores VALID uncertain pairs, cross-fitted residual gain over logit(p) (both S1-hash halves). Saves the model for test.
import json, math
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
link_split("train", ["US", "India"])
mf = find("model_k1/feature_cols.json")[0]; shutil.copytree(os.path.dirname(mf), f"{W}/p1/model_k1", dirs_exist_ok=True)
os.environ.update({"ER_WORK_DIR": W, "ER_NORM_SUBDIR": "norm_v2", "ER_THREADS": "4"})
sys.path[:0] = [SRC, f"{CODE}/tools"]
import numpy as np, polars as pl, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
MODEL = os.environ.get("CE_MODEL", "intfloat/multilingual-e5-small")
MAXLEN, BS, EPOCHS, LR = 128, 256, 2, 5e-5
MAX_TRAIN = int(os.environ.get("CE_MAX_TRAIN", "1500000"))
OUT, M = "/kaggle/working", f"{W}/p1/model_k1"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
dev, ngpu = "cuda", torch.cuda.device_count()
log(f"GPUs {ngpu}: {[torch.cuda.get_device_name(i) for i in range(ngpu)]} | model {MODEL}")

def raw_texts(split):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "business_address"])
                   for s in (1, 2, 3)])
    return t.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                    .str.to_lowercase().str.replace_all(r"\s+", " ").alias("txt"))
T = raw_texts("train")
def attach(df):
    return (df.join(T.rename({"entity_id": "s1_id", "txt": "ta"}), on="s1_id")
              .join(T.rename({"entity_id": "m_id", "txt": "tb"}), on="m_id"))

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

tok = AutoTokenizer.from_pretrained(MODEL)
def enc(a, b):
    e = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return e["input_ids"].to(dev, non_blocking=True), e["attention_mask"].to(dev, non_blocking=True)

tr = pl.read_parquet(f"{M}/s2fit_p1.parquet").filter(pl.col("p1") >= 0.01)
tr = attach(tr).sample(min(MAX_TRAIN, tr.height), seed=7, shuffle=True)
log(f"train pairs {tr.height} (positive {tr['y'].mean():.3f}; p1<0.99 share {(tr['p1'] < 0.99).mean():.3f})")
ta, tb = tr["ta"].to_numpy(), tr["tb"].to_numpy()
y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
torch.manual_seed(7); np.random.seed(7)
base = CE().to(dev)
net = torch.nn.DataParallel(base) if ngpu > 1 else base
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(y) / BS)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler("cuda")
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); perm = np.random.permutation(len(y)); tot = 0.0
    for k, s in enumerate(range(0, len(y), BS)):
        idx = perm[s:s + BS]
        ids, am = enc(ta[idx], tb[idx])
        loss = F.binary_cross_entropy_with_logits(net(ids, am), y[idx].to(dev))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(idx)
        if k % 500 == 0:
            log(f"epoch {ep + 1} step {k}/{math.ceil(len(y) / BS)} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)")
    log(f"epoch {ep + 1}/{EPOCHS} mean loss {tot / len(y):.4f} ({(time.time() - t0) / 60:.1f} min)")
os.makedirs(f"{OUT}/ce_model", exist_ok=True)
torch.save(base.state_dict(), f"{OUT}/ce_model/ce.pt"); tok.save_pretrained(f"{OUT}/ce_model")
json.dump({"model": MODEL, "maxlen": MAXLEN}, open(f"{OUT}/ce_model/cfg.json", "w"))
del tr, ta, tb

@torch.inference_mode()
def score(a, b, bs=1024):
    net.eval(); res = np.empty(len(a), dtype=np.float32)
    order = np.argsort(np.fromiter((len(x) + len(z) for x, z in zip(a, b)), dtype=np.int64, count=len(a)))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        res[idx] = torch.sigmoid(net(*enc(a[idx], b[idx]))).cpu().numpy()
    return res

sc = pl.read_parquet(f"{M}/valid_scored.parquet").select("s1_id", "m_id", "y", "p")
va = attach(sc.filter((pl.col("p") >= 0.01) & (pl.col("p") <= 0.999)))
t1 = time.time(); ce = score(va["ta"].to_numpy(), va["tb"].to_numpy())
log(f"VALID uncertain pairs scored: {va.height} in {(time.time() - t1) / 60:.1f} min")
cev = va.select("s1_id", "m_id").with_columns(pl.Series("p", ce))
cev.write_parquet(f"{OUT}/ce_valid.parquet")
from sklearn.metrics import roc_auc_score, average_precision_score
yv = va["y"].to_numpy()
log(f"uncertain band: AUC stage-2 p {roc_auc_score(yv, va['p'].to_numpy()):.4f} vs CE {roc_auc_score(yv, ce):.4f} | "
    f"AP p {average_precision_score(yv, va['p'].to_numpy()):.4f} vs CE {average_precision_score(yv, ce):.4f}")
from quick_residual import macro, universe, oof
s1v, truth = universe()
d = sc.join(cev.rename({"p": "ce"}), on=["s1_id", "m_id"], how="left")
p = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
flag = d["ce"].is_not_null().cast(pl.Float64).to_numpy(); cef = d["ce"].fill_null(-1.0).to_numpy()
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
res = {}
for name, X in (("A logit p", np.c_[lp]), ("B + CE", np.c_[lp, cef, flag])):
    res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(X, yy, grp))), s1v, truth)
    log(f"{name:10s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
A, B = res["A logit p"], res["B + CE"]
msg = (f"CE residual delta {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | "
       f"US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
log(msg)
json.dump({"A": A, "B": B, "msg": msg}, open(f"{OUT}/ce_result.json", "w"), default=float, indent=1)
print("CE_DONE", flush=True)
