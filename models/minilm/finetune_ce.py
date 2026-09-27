"""Fine-tune a MiniLM-L6 cross-encoder on entity pairs, then score VALID and test uncertain pairs (runs on a CUDA GPU,
e.g. an RTX 4060 laptop under WSL2).
Input folder (ce_pack): train_pairs, valid_pairs, test_pairs, train_records, test_records (.parquet).
Output (same folder): ce_valid.parquet (s1_id, m_id, p), ce_test_<country>.parquet, ce_model/ (weights), ce_log.txt.
Setup:  python3 -m venv ce && . ce/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cu124
        pip install transformers polars pyarrow
Usage:  python finetune_ce.py --pack ./ce_pack [--epochs 1 --bs 128 --max-len 96 --lr 5e-5]"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="./ce_pack")
ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--max-len", type=int, default=96)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--limit", type=int, default=0, help="train on the first N pairs only (smoke test / CPU pilot)")
ap.add_argument("--device", default="cuda", help="cuda | cpu (CPU pilot: fp32, no mixed precision)")
ap.add_argument("--no-test", action="store_true", help="score VALID only")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--mask-addr", type=float, default=0.0, help="blank the record-side address on this fraction of training pairs (name-only regime)")
ap.add_argument("--tag", default="", help="suffix for output files (ce_valid<tag>.parquet, ce_test<tag>_<c>.parquet, ce_model<tag>/)")
a = ap.parse_args()
D = Path(a.pack)
LOG = open(D / "ce_log.txt", "a")


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


CUDA = a.device == "cuda"
if CUDA:
    assert torch.cuda.is_available(), "no CUDA device"
elif a.threads:
    torch.set_num_threads(a.threads)
dev = torch.device(a.device)
log(f"device {torch.cuda.get_device_name(0) if CUDA else 'cpu'}; model {a.model}")
AC = (lambda: torch.autocast("cuda", dtype=torch.float16)) if CUDA else (lambda: torch.autocast("cpu", enabled=False))
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, ignore_mismatched_sizes=True).to(dev)


def texts(pairs, rec):
    m = dict(zip(rec["entity_id"].to_list(), rec["text"].to_list()))
    tb = [m[x] for x in pairs["m_id"].to_list()]
    if "tag" in pairs.columns:   # tagged pack (build_ce_tags.py): "num up 9; adds holding || <name> | <address>"
        tb = [t + " || " + x for t, x in zip(pairs["tag"].to_list(), tb)]
    return [m[x] for x in pairs["s1_id"].to_list()], tb


def batches(ta, tb, bs, order):
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        yield idx, {k: v.to(dev, non_blocking=True) for k, v in enc.items()}


@torch.no_grad()
def predict(ta, tb, bs=1024 if CUDA else 256):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)])     # length-sorted batches: less padding
    out = np.zeros(len(ta), np.float32)
    for idx, enc in batches(ta, tb, bs, order):
        with AC():
            out[idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
    return 1 / (1 + np.exp(-out))


tr = pl.read_parquet(D / "train_pairs.parquet")
rtr = pl.read_parquet(D / "train_records.parquet")
if a.limit:
    tr = pl.concat([tr.filter(pl.col("holdout")).head(max(2000, a.limit // 40)), tr.filter(~pl.col("holdout")).head(a.limit)])
ho, tr = tr.filter(pl.col("holdout")), tr.filter(~pl.col("holdout"))
ta, tb = texts(tr, rtr)
ha, hb = texts(ho, rtr)
if a.mask_addr > 0:   # masked training: the record keeps only its name, as address-less records do at inference
    rng = np.random.default_rng(13)
    for i in np.flatnonzero(rng.random(len(tb)) < a.mask_addr):
        tb[i] = tb[i].rsplit(" | ", 1)[0] + " | "
    log(f"masked record address on {a.mask_addr:.0%} of training pairs")
y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
hy = ho["y"].to_numpy()
steps = int(math.ceil(len(ta) / a.bs) * a.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler("cuda", enabled=CUDA)
lossf = torch.nn.BCEWithLogitsLoss()
log(f"train {len(ta)} pairs (pos {y.mean():.3f}), holdout {len(ha)}, steps {steps}, bs {a.bs}, max_len {a.max_len}")


def holdout_report(tag):
    p = predict(ha, hb)
    ll = -np.mean(hy * np.log(np.clip(p, 1e-7, 1)) + (1 - hy) * np.log(np.clip(1 - p, 1e-7, 1)))
    acc = np.mean((p >= 0.5) == hy)
    log(f"{tag}: holdout logloss {ll:.4f} acc {acc:.4f}")
    model.train()


holdout_report("before training")
step, t0 = 0, time.time()
model.train()
while step < steps:
    order = np.random.default_rng(step).permutation(len(ta))
    for idx, enc in batches(ta, tb, a.bs, order):
        with AC():
            loss = lossf(model(**enc).logits.float().squeeze(-1), y[idx].to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        step += 1
        if step % (500 if CUDA else 100) == 0:
            r = step * a.bs / (time.time() - t0)
            log(f"step {step}/{steps} loss {loss.item():.4f} ({r:.0f} pairs/s, ETA {(steps - step) * a.bs / r / 60:.1f} min)")
        if step % 5000 == 0:
            holdout_report(f"step {step}")
        if step >= steps:
            break
holdout_report("after training")
model.save_pretrained(D / f"ce_model{a.tag}"); tok.save_pretrained(D / f"ce_model{a.tag}")

va = pl.read_parquet(D / "valid_pairs.parquet")
va.with_columns(pl.Series("p", predict(*texts(va, rtr)))).select("s1_id", "m_id", "p").write_parquet(D / f"ce_valid{a.tag}.parquet")
log(f"VALID scored ({va.height})")
if a.no_test:
    raise SystemExit(0)
te = pl.read_parquet(D / "test_pairs.parquet")
rte = pl.read_parquet(D / "test_records.parquet")
te = te.with_columns(pl.Series("p", predict(*texts(te, rte))))
for c in te["country"].unique().to_list():
    te.filter(pl.col("country") == c).select("s1_id", "m_id", "p").write_parquet(D / f"ce_test{a.tag}_{c}.parquet")
log(f"test scored ({te.height}); done")
