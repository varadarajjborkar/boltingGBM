"""Stage 3: stack neural pair scores on top of stage 2, fitted on VALID with cross-fitting (docs/NIGHT_PLAN.md).

  charnet-train  train the char network (tools/charnet_train.PairNet) on the model's stage-2 training states: pairs of
                 s2fit_p1.parquet with p1 >= 0.01, labels from the ground truth (no VALID labels involved)
  charnet-score  score the uncertain pairs (0.01 <= stage-2 p <= 0.999) of VALID and of test
  fit            stage-3 LightGBM on VALID pairs: inputs logit(p) + cn + cn_scored (+ any --extra score files).
                 Estimate: 2 folds by S1 hash (each half predicted by a model fitted on the other half), macro F0.5
                 over every VALID S1 at the best threshold, against the same procedure on logit(p) alone.
                 Then refit on all VALID, apply to every scored test pair and write
                 work/p1/test/<c>/stage2_scored_<model>_s3.parquet, so p1_score.py --stage write can make a submission
                 with ER_MODEL_DIR=<...>/<model>_s3.
Usage (from src, ER_NORM_SUBDIR=norm_v2):
  python ../../../tools/stage3.py charnet-train --model ../../../work/p1/model_n2
  python ../../../tools/stage3.py charnet-score --model ../../../work/p1/model_n2
  python ../../../tools/stage3.py fit --model ../../../work/p1/model_n2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from charnet_train import PairNet, tensors  # noqa: E402
from quick_residual import macro, universe  # noqa: E402
from utils import WORK  # noqa: E402

P_LO, P_HI = 0.01, 0.999
COUNTRIES_TEST = ["France", "India", "US"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def texts(split, countries):
    a, b = [], []
    for c in countries:
        d = WORK / "p1" / split / c
        a.append(pl.read_parquet(d / "s1.parquet", columns=["entity_id", "name_core", "addr_text"]))
        b.append(pl.read_parquet(d / "pool.parquet", columns=["entity_id", "name_core", "addr_text"]))
    return (pl.concat(a).rename({"entity_id": "s1_id", "name_core": "an", "addr_text": "aa"}),
            pl.concat(b).rename({"entity_id": "m_id", "name_core": "bn", "addr_text": "ba"}))


def attach(pairs, T):
    return pairs.join(T[0], on="s1_id").join(T[1], on="m_id")


def score(net, df, bs=8192):
    X, out = tensors(df), []
    with torch.inference_mode():
        for s in range(0, df.height, bs):
            out.append(torch.sigmoid(net(*(x[s:s + bs] for x in X))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def charnet_train(M, max_train, epochs, threads):
    torch.set_num_threads(threads); torch.manual_seed(7)
    tr = pl.read_parquet(M / "s2fit_p1.parquet").filter(pl.col("p1") >= 0.01)
    tr = attach(tr, texts("train", ["US", "India"])).sample(min(max_train, tr.height), seed=7, shuffle=True)
    log(f"char net training on {tr.height} stage-2-state pairs (positive {tr['y'].mean():.3f})")
    X, y = tensors(tr), torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    net = PairNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm, tot = torch.randperm(len(y)), 0.0
        for s in range(0, len(y), 512):
            idx = perm[s:s + 512]
            loss = lossf(net(*(x[idx] for x in X)), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{epochs} loss {tot / len(y):.4f} ({(time.time() - t0) / 60:.1f} min)")
    (M / "stage3").mkdir(exist_ok=True)
    torch.save(net.state_dict(), M / "stage3" / "charnet.pt")


def charnet_score(M, threads):
    torch.set_num_threads(threads)
    net = PairNet(); net.load_state_dict(torch.load(M / "stage3" / "charnet.pt")); net.eval()
    va = pl.read_parquet(M / "valid_scored.parquet").filter((pl.col("p") >= P_LO) & (pl.col("p") <= P_HI)).select("s1_id", "m_id")
    va = attach(va, texts("train", ["US", "India"]))
    va.select("s1_id", "m_id").with_columns(pl.Series("cn", score(net, va))).write_parquet(M / "stage3" / "cn_valid.parquet")
    log(f"VALID: {va.height} uncertain pairs scored")
    if not (WORK / "p1" / "test").exists():          # dev-loop worlds have no test split
        return
    parts = []
    for c in COUNTRIES_TEST:
        f = WORK / "p1" / "test" / c / f"stage2_scored_{M.name}.parquet"
        te = pl.read_parquet(f).filter((pl.col("p") >= P_LO) & (pl.col("p") <= P_HI)).select("s1_id", "m_id")
        te = attach(te, texts("test", [c]))
        parts.append(te.select("s1_id", "m_id").with_columns(pl.Series("cn", score(net, te)), pl.lit(c).alias("country")))
        log(f"test {c}: {te.height} uncertain pairs scored")
    pl.concat(parts).write_parquet(M / "stage3" / "cn_test.parquet")


def design(sc, cn, extras):
    d = sc.join(cn.select("s1_id", "m_id", "cn"), on=["s1_id", "m_id"], how="left")
    cols = ["lp", "cn", "cn_scored"]
    for name, e in extras.items():
        d = d.join(e.select("s1_id", "m_id", pl.col("p").alias(name)), on=["s1_id", "m_id"], how="left")
        cols.append(name)
    p = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6)
    d = d.with_columns(pl.Series("lp", np.log(p / (1 - p))), pl.col("cn").is_not_null().cast(pl.Float64).alias("cn_scored"),
                       pl.col("cn").fill_null(-1.0))
    return d, cols


def fit(M, extra_specs):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8, seed=7)
    s1v, truth = universe()
    sc = pl.read_parquet(M / "valid_scored.parquet").select("s1_id", "m_id", "y", "p")
    extras_valid = {n: pl.read_parquet(Path(v)) for n, (v, _) in extra_specs.items()}
    d, cols = design(sc, pl.read_parquet(M / "stage3" / "cn_valid.parquet"), extras_valid)
    y, grp = d["y"].to_numpy(), (d["s1_id"].hash(13) % 2).to_numpy()
    res = {}
    for name, cs in (("stage2 only (logit p)", ["lp"]), ("stage3", cols)):
        X = d.select(cs).to_numpy().astype(np.float64)
        oof = np.zeros(len(y))
        for k in (0, 1):
            m = lgb.train(params, lgb.Dataset(X[grp != k], y[grp != k]), 300)
            oof[grp == k] = m.predict(X[grp == k])
        res[name] = macro(d.select("s1_id", "m_id").with_columns(pl.Series("p", oof)), s1v, truth)
        log(f"{name:22s} {json.dumps({k: round(float(v), 5) for k, v in res[name].items()})}")
    A, B = res["stage2 only (logit p)"], res["stage3"]
    log(f"stage-3 gain {B['macro'] - A['macro']:+.5f} | halves {B['h0'] - A['h0']:+.5f} {B['h1'] - A['h1']:+.5f} | "
        f"US {B['US'] - A['US']:+.5f} India {B['India'] - A['India']:+.5f}")
    final = lgb.train(params, lgb.Dataset(d.select(cols).to_numpy().astype(np.float64), y), 300)
    (M / "stage3").mkdir(exist_ok=True)
    (M / "stage3" / "report.json").write_text(json.dumps({"stage2_only": A, "stage3": B, "threshold": B["t"], "cols": cols},
                                                          default=float, indent=1))
    if not (WORK / "p1" / "test").exists():
        return
    cn_test = pl.read_parquet(M / "stage3" / "cn_test.parquet")
    for c in COUNTRIES_TEST:
        te = pl.read_parquet(WORK / "p1" / "test" / c / f"stage2_scored_{M.name}.parquet").select("s1_id", "m_id", "p")
        ex = {n: pl.concat([pl.read_parquet(Path(t.format(c=cc))) for cc in [c]]) for n, (_, t) in extra_specs.items()}
        dt, _ = design(te, cn_test.filter(pl.col("country") == c), ex)
        p3 = final.predict(dt.select(cols).to_numpy().astype(np.float64))
        dt.select("s1_id", "m_id").with_columns(pl.Series("p", p3)).write_parquet(
            WORK / "p1" / "test" / c / f"stage2_scored_{M.name}_s3.parquet")
        log(f"test {c}: stage-3 scores written ({dt.height} pairs)")
    (M / "stage3" / "report.json").write_text(json.dumps({"stage2_only": A, "stage3": B, "threshold": B["t"], "cols": cols},
                                                          default=float, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["charnet-train", "charnet-score", "fit"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--extra", default="", help="name=valid_scored_path@test_scores_path_with_{c};...")
    a = ap.parse_args()
    M = Path(a.model).resolve()
    extras = {}
    for x in filter(None, a.extra.split(";")):
        n, rest = x.split("=", 1)
        v, t = rest.split("@", 1)
        extras[n] = (v, t)
    if a.mode == "charnet-train":
        charnet_train(M, a.max_train, a.epochs, a.threads)
    elif a.mode == "charnet-score":
        charnet_score(M, a.threads)
    else:
        fit(M, extras)


if __name__ == "__main__":
    main()
