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
# ---- bgbm-tta: TEST-TIME ADAPTATION of the e5 v1 cross-encoder with REAL test decoys.
# Negatives = test look-alikes the team's own rules identify (pure up-shift of the house-number multiset, record name ~ S1
# name; India only when stack p < 0.98, as the team's India fix says high-p India shifts are real) + decoy-vocabulary adds.
# Positives = test pairs with stack p >= 0.999, equal house-number multiset, similar names. Mixed 1:1 with training pairs.
# No VALID labels in training. Output: scores for the team's VALID and test pairs + residual on VALID (must stay neutral).
import json, math, re
from collections import Counter
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars==1.44.2"], check=True)
link_split("train", ["US", "India"])
mf = find("model_k1/feature_cols.json")[0]; shutil.copytree(os.path.dirname(mf), f"{W}/p1/model_k1", dirs_exist_ok=True)
os.environ.update({"ER_WORK_DIR": W, "ER_NORM_SUBDIR": "norm_v2", "ER_THREADS": "4"})
sys.path[:0] = [SRC, f"{CODE}/tools"]
import numpy as np, polars as pl, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
VOC = set(['advisors', 'agro', 'associates', 'bakery', 'biotech', 'builders', 'capital', 'care', 'cargo', 'central', 'coastal', 'consultants', 'consulting', 'data', 'deli', 'dental', 'design', 'developers', 'digital', 'downtown', 'east', 'eastgate', 'electric', 'energy', 'exports', 'farms', 'finance', 'financial', 'foods', 'greater', 'grill', 'group', 'harbor', 'health', 'healthcare', 'highland', 'holdings', 'hospitality', 'india', 'industries', 'infotech', 'infra', 'infratech', 'interiors', 'international', 'investments', 'lakeside', 'liquor', 'logistics', 'marketing', 'media', 'medical', 'metro', 'midtown', 'motors', 'north', 'northside', 'overseas', 'producer', 'products', 'projects', 'properties', 'realty', 'research', 'restaurant', 'retail', 'riverside', 'salon', 'sciences', 'security', 'software', 'solutions', 'south', 'southside', 'spa', 'summit', 'supply', 'systems', 'tax', 'textiles', 'therapy', 'trading', 'transport', 'uptown', 'valley', 'ventures', 'west', 'westgate', 'works'])
NUM = re.compile(r"\d+"); PURE = {3, 4, 5, 7, 9, 11, 13, 21}
def nums(s): return [int(x) for x in NUM.findall(s or "") if len(x) <= 7]
def rel(sa, sb):
    A, B = nums(sa), nums(sb)
    if not A or not B: return "missing"
    ca, cb = Counter(A), Counter(B)
    if ca == cb: return "eq"
    SA, SB = set(A) - set(B), set(B) - set(A)
    if any((b - a) in PURE for a in SA for b in SB): return "shift_pure"
    return "other"
def jac(a, b):
    A, B = set((a or "").split()), set((b or "").split()); return len(A & B) / max(len(A | B), 1)
MD = os.path.dirname(find("ce_model/ce.pt", prefer="bgbm-ce")[0]); cfg = json.load(open(f"{MD}/cfg.json")); MODEL, MAXLEN = cfg["model"], cfg["maxlen"]
dev, ngpu = "cuda", torch.cuda.device_count()
def texts(split):
    t = pl.concat([pl.read_parquet(f"{W}/norm_v2/{split}_source{s}.parquet", columns=["entity_id", "business_name", "business_address", "name_core", "addr_text"]) for s in (1, 2, 3)])
    return t.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).str.to_lowercase().str.replace_all(r"\s+", " ").alias("txt"),
                    pl.col("name_core").fill_null("").alias("nc"), pl.col("addr_text").fill_null("").alias("at"))
TE = texts("test")
fx = lambda n: find(n, prefer="bgbm-exports")[0]
tp = pl.concat([pl.read_parquet(fx(f"best_test_scored_{c}.parquet")).with_columns(pl.lit(c).alias("cty")) for c in ("US", "India", "France")])
tp = (tp.join(TE.select(pl.col("entity_id").alias("s1_id"), pl.col("nc").alias("na"), pl.col("at").alias("aa")), on="s1_id")
        .join(TE.select(pl.col("entity_id").alias("m_id"), pl.col("nc").alias("nb"), pl.col("at").alias("ab")), on="m_id"))
R = [rel(a, b) for a, b in zip(tp["aa"].to_list(), tp["ab"].to_list())]
J = [jac(a, b) for a, b in zip(tp["na"].to_list(), tp["nb"].to_list())]
V = [bool((set(b.split()) - set(a.split())) & VOC) for a, b in zip(tp["na"].to_list(), tp["nb"].to_list())]
tp = tp.with_columns(pl.Series("rel", R), pl.Series("jac", J), pl.Series("voc", V))
neg = tp.filter(((pl.col("rel") == "shift_pure") & (pl.col("jac") >= 0.6) & ((pl.col("cty") != "India") | (pl.col("p") < 0.98)))
                | (pl.col("voc") & (pl.col("cty") != "France")))
pos = tp.filter((pl.col("p") >= 0.999) & (pl.col("rel") == "eq") & (pl.col("jac") >= 0.5))
log(f"test-derived negatives {neg.height} (" + ", ".join(f"{c} {neg.filter(pl.col('cty') == c).height}" for c in ("US", "India", "France")) + f"); positives available {pos.height}")
pos = pos.sample(min(pos.height, 2 * neg.height), seed=5)
tt = pl.concat([neg.select("s1_id", "m_id").with_columns(pl.lit(0).alias("y")), pos.select("s1_id", "m_id").with_columns(pl.lit(1).alias("y"))])
TR = texts("train")
s2 = pl.read_parquet(f"{W}/p1/model_k1/s2fit_p1.parquet").filter(pl.col("p1") >= 0.01).sample(tt.height, seed=5).select("s1_id", "m_id", "y")
def attach(df, T):
    return (df.join(T.select(pl.col("entity_id").alias("s1_id"), pl.col("txt").alias("ta")), on="s1_id")
              .join(T.select(pl.col("entity_id").alias("m_id"), pl.col("txt").alias("tb")), on="m_id"))
tr = pl.concat([attach(tt, TE), attach(s2, TR)]).sample(fraction=1.0, shuffle=True, seed=5)
log(f"TTA training pairs {tr.height} (test-derived {tt.height}, training {s2.height}; positive {tr['y'].mean():.3f})")
class CE(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.enc = AutoModel.from_pretrained(MODEL); self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, ids, am):
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float(); return self.head((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)
base = CE(); base.load_state_dict(torch.load(f"{MD}/ce.pt", map_location="cpu")); base.to(dev)
net = torch.nn.DataParallel(base) if ngpu > 1 else base
tok = AutoTokenizer.from_pretrained(MD)
def enc(a, b):
    e = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return e["input_ids"].to(dev), e["attention_mask"].to(dev)
ta, tb = tr["ta"].to_numpy(), tr["tb"].to_numpy(); y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
BS = 256; opt = torch.optim.AdamW(net.parameters(), lr=2e-5, weight_decay=0.01); steps = math.ceil(len(y) / BS)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps); scaler = torch.amp.GradScaler("cuda"); t0 = time.time(); net.train()
for k, s_ in enumerate(range(0, len(y), BS)):
    loss = F.binary_cross_entropy_with_logits(net(*enc(ta[s_:s_ + BS], tb[s_:s_ + BS])), y[s_:s_ + BS].to(dev))
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    if k % 500 == 0: log(f"step {k}/{steps} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)")
@torch.inference_mode()
def score(a, b, bs=1024):
    net.eval(); res = np.empty(len(a), dtype=np.float32)
    order = np.argsort(np.fromiter((len(x) + len(z) for x, z in zip(a, b)), dtype=np.int64, count=len(a)))
    for s_ in range(0, len(order), bs):
        idx = order[s_:s_ + bs]; res[idx] = torch.sigmoid(net(*enc(a[idx], b[idx]))).cpu().numpy()
    return res
OUT = "/kaggle/working"
va = attach(pl.read_parquet(fx("pairs_valid.parquet")).select("s1_id", "m_id"), TR)
cev = va.select("s1_id", "m_id").with_columns(pl.Series("p", score(va["ta"].to_numpy(), va["tb"].to_numpy())))
cev.write_parquet(f"{OUT}/cetta_team_valid.parquet")
from quick_residual import macro, universe, oof
s1v, truth = universe()
sc = pl.read_parquet(fx("best_valid_scored.parquet")).select("s1_id", "m_id", "y", "p")
v1 = pl.read_parquet(find("ce_team_valid.parquet", prefer="bgbm-ce-team")[0]).rename({"p": "c1"})
d = sc.join(v1, on=["s1_id", "m_id"], how="left", maintain_order="left").join(cev.rename({"p": "ct"}), on=["s1_id", "m_id"], how="left", maintain_order="left")
pp = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp)); fl = d["c1"].is_not_null().cast(pl.Float64).to_numpy()
c1 = d["c1"].fill_null(-1.0).to_numpy(); ct = d["ct"].fill_null(-1.0).to_numpy(); yy, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
res = {}
for name, X in (("A stack+e5v1", np.c_[lp, c1, fl]), ("B stack+tta", np.c_[lp, ct, fl]), ("C stack+e5v1+tta", np.c_[lp, c1, ct, fl])):
    res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof(X, yy, grp))), s1v, truth)
    log(f"{name:18s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
A = res["A stack+e5v1"]
for k in ("B stack+tta", "C stack+e5v1+tta"):
    B = res[k]; log(f"{k} vs A: {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
json.dump(res, open(f"{OUT}/cetta_result.json", "w"), default=float, indent=1)
for c in ["US", "India", "France"]:
    te = attach(pl.read_parquet(fx(f"pairs_test_{c}.parquet")).select("s1_id", "m_id"), TE)
    te.select("s1_id", "m_id").with_columns(pl.Series("p", score(te["ta"].to_numpy(), te["tb"].to_numpy()))).write_parquet(f"{OUT}/cetta_team_test_{c}.parquet")
    log(f"team test {c}: {te.height} pairs scored")
print("TTA_DONE", flush=True)
