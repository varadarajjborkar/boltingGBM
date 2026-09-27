"""CE v4 on the Lightning L4: multilingual-e5-base cross-encoder with tags (num relation, extra numbers, added and dropped
name words), trained on stage-2-state pairs (s2fit, p1 >= 0.01) + hard pairs of the FIT states (fit_oof p1 in [0.02, 0.98]),
never on VALID (NY, AP/TS). Scores the team's exact pairs and measures the gain over the team's stack_s3a2 VALID p
(2 folds by S1 hash, both halves), like tools/quick_residual.py."""
import json, math, os, sys, time
import numpy as np, polars as pl, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

W = os.environ.get("ER_WORK_DIR", "work"); OUT = f"{W}/out_ce4"; os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get("CE_MODEL", "microsoft/mdeberta-v3-base"); MAXLEN, BS, LR, MAX_TRAIN = 144, 32, 2e-5, 2_600_000
ACC = 4
dev = "cuda"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def raw_texts(split):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet",
                                   columns=["entity_id", "business_name", "business_address", "name_core", "addr_nums"]) for s in (1, 2, 3)])
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
        sa = set(x.split()); sb = set(y.split())
        add = [w for w in y.split() if w not in sa][:3]; drop = [w for w in x.split() if w not in sb][:2]
        ua = {str(n).lstrip("0") for n in (u or []) if n}
        extra = [str(n).lstrip("0") for n in (v or []) if n and str(n).lstrip("0") not in ua][:2]
        t = _rel(u, v) + ("; num extra " + " ".join(extra) if extra and ua else "")
        t += ("; adds " + " ".join(add) if add else "") + ("; drops " + " ".join(drop) if drop else "")
        out.append(t + " || ")
    return out

def attach(df, T):
    d = (df.join(T.rename({"entity_id": "s1_id", "txt": "ta", "nc": "nca", "nums": "na"}), on="s1_id")
           .join(T.rename({"entity_id": "m_id", "txt": "tb", "nc": "ncb", "nums": "nb"}), on="m_id"))
    d = d.with_columns(pl.Series("tg", tags(d["nca"].to_list(), d["ncb"].to_list(), d["na"].to_list(), d["nb"].to_list())))
    return d.with_columns((pl.col("tg") + pl.col("tb")).alias("tb")).drop("nca", "ncb", "na", "nb", "tg")

class CE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, ids, am):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        return self.head((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

try:
    tok = AutoTokenizer.from_pretrained(MODEL); AutoModel.from_pretrained(MODEL)
except Exception as e:
    log(f"{MODEL} failed to load ({e!r}); falling back to intfloat/multilingual-e5-base")
    MODEL = "intfloat/multilingual-e5-base"; LR = 3e-5; tok = AutoTokenizer.from_pretrained(MODEL)
log(f"model {MODEL}")
def enc(a, b):
    e = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return e["input_ids"].to(dev, non_blocking=True), e["attention_mask"].to(dev, non_blocking=True)

TT = raw_texts("train")
s2 = pl.read_parquet(f"{W}/s2fit_p1.parquet").filter(pl.col("p1") >= 0.01)
fo = pl.read_parquet(f"{W}/fit_oof_p1.parquet").sample(800_000, seed=11)   # FIT blocking candidates (single stage-1 model: no OOF p1)
log(f"s2fit pairs {s2.height} (pos {s2['y'].mean():.3f}); FIT candidate pairs {fo.height} (pos {fo['y'].mean():.3f})")
tr = pl.concat([s2.select("s1_id", "m_id", "y"), fo.select("s1_id", "m_id", "y")]).unique(["s1_id", "m_id"])
tr = attach(tr.sample(min(MAX_TRAIN, tr.height), seed=11, shuffle=True), TT)
log(f"train pairs {tr.height} (positive {tr['y'].mean():.3f})")
ta, tb = tr["ta"].to_numpy(), tr["tb"].to_numpy(); y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
del tr, s2, fo
torch.manual_seed(11); np.random.seed(11)
net = CE().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
steps = math.ceil(len(y) / BS); osteps = math.ceil(steps / ACC); sched = get_linear_schedule_with_warmup(opt, int(0.05 * osteps), osteps)
t0 = time.time(); net.train(); perm = np.random.permutation(len(y)); tot = 0.0
opt.zero_grad(set_to_none=True)
for k, s in enumerate(range(0, len(y), BS)):
    idx = perm[s:s + BS]
    loss = F.binary_cross_entropy_with_logits(net(*enc(ta[idx], tb[idx])), y[idx].to(dev))
    (loss / ACC).backward(); tot += loss.item() * len(idx)
    if (k + 1) % ACC == 0 or k == steps - 1:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if k % 4000 == 0:
        el = time.time() - t0; log(f"step {k}/{steps} loss {loss.item():.4f} ({el / 60:.1f} min, eta {(steps - k - 1) * el / max(k + 1, 1) / 60:.0f} min)")
log(f"epoch mean loss {tot / len(y):.4f} ({(time.time() - t0) / 60:.1f} min)")
torch.save(net.state_dict(), f"{OUT}/ce4.pt"); del ta, tb

@torch.inference_mode()
def score(a, b, bs=1024):
    net.eval(); res = np.empty(len(a), dtype=np.float32)
    order = np.argsort(np.fromiter((len(x) + len(z) for x, z in zip(a, b)), dtype=np.int64, count=len(a)))
    for s_ in range(0, len(order), bs):
        idx = order[s_:s_ + bs]; res[idx] = torch.sigmoid(net(*enc(a[idx], b[idx]))).float().cpu().numpy()
    return res

pv = pl.read_parquet(f"{W}/pairs_valid.parquet"); va = attach(pv.select("s1_id", "m_id"), TT)
cev = va.select("s1_id", "m_id").with_columns(pl.Series("p", score(va["ta"].to_numpy(), va["tb"].to_numpy())))
cev.write_parquet(f"{OUT}/ce4_team_valid.parquet"); log(f"team VALID pairs scored {cev.height}")

# residual over the team's stack_s3a2 VALID p (same universe, halves and threshold sweep as quick_residual)
import lightgbm as lgb
s1 = pl.read_parquet(f"{W}/norm_v2/train_source1.parquet", columns=["entity_id", "country", "addr_state"])
s1v = s1.filter(((pl.col("country") == "US") & (pl.col("addr_state") == "ny")) |
                ((pl.col("country") == "India") & pl.col("addr_state").is_in(["ap", "ts"]))).select("entity_id", "country")
gt = pl.read_parquet(f"{W}/parquet/train_ground_truth.parquet")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
gt = gt.join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
ntrue = gt.group_by("s1_id").len().rename({"len": "nt"}); gtp = gt.with_columns(pl.lit(1).alias("t"))
def macro(sc):
    sc1 = sc.sort("p", descending=True).unique("m_id", keep="first")
    best = None
    for t in np.arange(0.4, 0.96, 0.025).round(3):
        pr = sc1.filter(pl.col("p") >= t).join(gtp, on=["s1_id", "m_id"], how="left").with_columns(pl.col("t").fill_null(0))
        agg = pr.group_by("s1_id").agg(pl.len().alias("np"), pl.col("t").sum().alias("tp"))
        u = (s1v.rename({"entity_id": "s1_id"}).join(agg, on="s1_id", how="left").join(ntrue, on="s1_id", how="left").fill_null(0)
               .with_columns(pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
                             .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))).alias("f"),
                             (pl.col("s1_id").hash(13) % 2).alias("h")))
        r = (float(t), u["f"].mean(), u.filter(pl.col("h") == 0)["f"].mean(), u.filter(pl.col("h") == 1)["f"].mean(),
             u.filter(pl.col("country") == "US")["f"].mean(), u.filter(pl.col("country") == "India")["f"].mean())
        if best is None or r[1] > best[1]: best = r
    return dict(zip(["t", "macro", "h0", "h1", "US", "India"], best))
def oof(X, yv, grp):
    p = np.zeros(len(yv)); prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=16, seed=7)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[grp != k], yv[grp != k]), 300); p[grp == k] = m.predict(X[grp == k])
    return p
scb = pl.read_parquet(f"{W}/best_valid_scored.parquet").select("s1_id", "m_id", "y", "p")
log(f"SANITY raw stack_s3a2 macro (must be ~0.9873): {macro(scb.select('s1_id', 'm_id', 'p'))}")
d = scb.join(cev.rename({"p": "ce"}), on=["s1_id", "m_id"], how="left", maintain_order="left")
pp = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
fl = d["ce"].is_not_null().cast(pl.Float64).to_numpy(); ce = d["ce"].fill_null(-1.0).to_numpy()
yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
res = {}
for name, X in (("A stack_s3a2 p", np.c_[lp]), ("B + ce4", np.c_[lp, ce, fl])):
    res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(X, yy, grp))))
    log(f"{name:16s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
A, B = res["A stack_s3a2 p"], res["B + ce4"]
log(f"ce4 over stack_s3a2: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
json.dump(res, open(f"{OUT}/ce4_result.json", "w"), default=float, indent=1)

TE = raw_texts("test")
for c in ["France", "India", "US"]:
    te = attach(pl.read_parquet(f"{W}/pairs_test_{c}.parquet").select("s1_id", "m_id"), TE)
    t1 = time.time(); ps = score(te["ta"].to_numpy(), te["tb"].to_numpy())
    te.select("s1_id", "m_id").with_columns(pl.Series("p", ps)).write_parquet(f"{OUT}/ce4_team_test_{c}.parquet")
    log(f"team test {c}: {te.height} pairs in {(time.time() - t1) / 60:.1f} min")
print("CE4_DONE", flush=True)
