"""Tiny character network (trained from scratch, CPU, minutes): learns the generator's letter-level noise.

Encoder: char embedding (32) -> 1-D convolutions (kernels 2-5, 96 filters each) -> max-pool, shared by name and address.
Pair head: for name and for address, [u, v, |u - v|, u * v] -> MLP -> logit. About 1M parameters.
Training data and scoring set are the same as tools/ce_train.py (hard pairs of the world's stage-2 states; uncertain
VALID pairs), output <model>/charnet/cn_valid.parquet (s1_id, m_id, y, p, cn) for tools/quick_residual.py.
Usage (from src):
  python ../../../tools/charnet_train.py --world <world dir> --model <world>/models/<run> [--max-train 300000 --epochs 3]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

P1_MIN, P_LO, P_HI = 0.01, 0.01, 0.999
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 "
CID = {c: i + 2 for i, c in enumerate(CHARS)}          # 0 = pad, 1 = other
LN, LA = 48, 80


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def enc(strings, L):
    out = np.zeros((len(strings), L), np.int64)
    for k, s in enumerate(strings):
        ids = [CID.get(ch, 1) for ch in (s or "")[:L]]
        out[k, :len(ids)] = ids
    return torch.from_numpy(out)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS) + 2, 32, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(32, 96, k, padding=k // 2) for k in (2, 3, 4, 5)])

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        return torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)   # 384


class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.head = nn.Sequential(nn.Linear(384 * 8, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, an, bn, aa, ba):
        u, v, p, q = self.enc(an), self.enc(bn), self.enc(aa), self.enc(ba)
        return self.head(torch.cat([u, v, (u - v).abs(), u * v, p, q, (p - q).abs(), p * q], dim=1)).squeeze(-1)


def table(W):
    parts = {}
    for c in ["US", "India"]:
        for f, pre in (("s1", "a"), ("pool", "b")):
            df = pl.read_parquet(W / "p1" / "train" / c / f"{f}.parquet", columns=["entity_id", "name_core", "addr_text"])
            parts.setdefault(pre, []).append(df)
    return {k: pl.concat(v) for k, v in parts.items()}


def attach(pairs, T):
    return (pairs.join(T["a"].rename({"entity_id": "s1_id", "name_core": "an", "addr_text": "aa"}), on="s1_id")
                 .join(T["b"].rename({"entity_id": "m_id", "name_core": "bn", "addr_text": "ba"}), on="m_id"))


def tensors(df):
    return (enc(df["an"].to_list(), LN), enc(df["bn"].to_list(), LN), enc(df["aa"].to_list(), LA), enc(df["ba"].to_list(), LA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-train", type=int, default=300000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(7)
    W, M = Path(a.world), Path(a.model)
    out = M / "charnet"
    out.mkdir(exist_ok=True)
    T = table(W)
    tr = attach(pl.read_parquet(M / "s2fit_p1.parquet").filter(pl.col("p1") >= P1_MIN), T)
    tr = tr.sample(min(a.max_train, tr.height), seed=7, shuffle=True)
    va = attach(pl.read_parquet(M / "valid_scored.parquet").filter((pl.col("p") >= P_LO) & (pl.col("p") <= P_HI)), T)
    log(f"train {tr.height} (positive {tr['y'].mean():.3f}); score {va.height} uncertain VALID pairs")
    Xtr, ytr = tensors(tr), torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    net = PairNet()
    log(f"parameters {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    t0, bs = time.time(), 512
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for s in range(0, len(ytr), bs):
            idx = perm[s:s + bs]
            loss = lossf(net(*(x[idx] for x in Xtr)), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{a.epochs} loss {tot / len(ytr):.4f} ({(time.time() - t0) / 60:.1f} min)")
    net.eval()
    Xva, sc = tensors(va), []
    with torch.inference_mode():
        for s in range(0, va.height, 4096):
            sc.append(torch.sigmoid(net(*(x[s:s + 4096] for x in Xva))).numpy())
    va = va.select("s1_id", "m_id", "y", "p").with_columns(pl.Series("cn", np.concatenate(sc)))
    va.write_parquet(out / "cn_valid.parquet")
    from sklearn.metrics import roc_auc_score
    log(f"uncertain VALID pairs: AUC stage-2 p {roc_auc_score(va['y'], va['p']):.4f} | AUC charnet {roc_auc_score(va['y'], va['cn']):.4f} "
        f"| total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
