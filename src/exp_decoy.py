"""Synthetic-decoy stage 2 (director spec docs/director/DECOY_SPEC_2026-09-26.md), runs on a worker with ER_MODEL_DIR = the
stage-1 model to keep (e.g. model_v9).

gen  : per country, for the stage-2 training states (role s2fit -> feats_fitD) and VALID (-> feats_validD): decoy entities
       copy 1-3 true S2/S3 records of a true S1 entity and apply the same change (SHIFT12, SHIFTK, MSHIDE, SUB, TOKEN, +token
       combo) to the raw name/address, then the standard normalizer. Candidates: the original record's candidate rows (option B).
       Features for every pair of the role's S1 are rebuilt with the decoys in the country context.
s2   : stage-1 p1 for feats_fitD, stage-2 + sibling features (decoy texts included), stage 2 retrained -> <out>/stage2.txt
eval : VALID clean (feats_valid) and injected (feats_validD): old stage 2 (ER_MODEL_DIR) vs new, macro F0.5 at t 0.70 with one
       S1 per record, by country and S1-hash half; scored tables are written for the rule check.
Usage: python exp_decoy.py --stage gen|s2|eval [--out DIR] [--rate-us 45 --rate-in 25] [--seed 1]
"""
import argparse
import json
import re

import numpy as np
import polars as pl

from utils import WORK, log, timed, write_parquet_atomic

K_ALL = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
MIX = {"US": [("SHIFT12", .45), ("SHIFTK", .40), ("TOKEN", .15)],
       "India": [("SHIFT12", .40), ("SHIFTK", .20), ("MSHIDE", .15), ("SUB", .10), ("TOKEN", .15)]}
NUM = re.compile(r"\d+")
VOCAB = json.loads((WORK.parent / "work" / "director" / "decoy_vocab_validated.json").read_text())


def _shift(addr, targets, k, rng, multipart=False):
    """Shift one digit run of addr by k: prefer a run whose value is in targets (numbers shared with S1)."""
    ms = list(NUM.finditer(addr or ""))
    if not ms:
        return None
    if multipart:
        ms2 = [m for m in ms if (m.start() > 0 and addr[m.start() - 1] in "-/")]
        if not ms2:
            return None
        m = ms2[-1]
    else:
        pick = [m for m in ms if int(m.group()) in targets]
        if not pick:                  # no number shared with S1: shifting a ZIP or unit would make a 'far' decoy
            return None
        m = pick[0]
    v = int(m.group()) + k
    return addr[:m.start()] + str(v) + addr[m.end():]


def _sub(addr, rng):
    ms = list(NUM.finditer(addr or ""))
    if len(ms) >= 2 and rng.random() < 0.5:           # drop one number component (and one adjacent separator)
        m = ms[rng.integers(len(ms))]
        s, e = m.start(), m.end()
        if e < len(addr) and addr[e] in "-/ ,":
            e += 1
        elif s > 0 and addr[s - 1] in "-/ ,":
            s -= 1
        return addr[:s] + addr[e:]
    return f"{int(rng.integers(1, 400))}, {addr}"      # add one number component


def make_decoys(c, s1, raw, gt, s1_ids, rate, rng, prefix):
    """Decoy raw rows (entity_id, business_name, business_address, country, src, orig_id, dtype) for S1 entities in s1_ids."""
    recs = gt.filter(pl.col("s1_id").is_in(s1_ids)).join(raw, left_on="m_id", right_on="entity_id")
    by = recs.group_by("s1_id").agg("m_id", "business_name", "business_address").sort("s1_id")
    n = min(by.height, int(round(rate / 1000 * len(s1_ids))))
    by = by.sample(n, seed=int(rng.integers(1 << 30)))
    s1n = dict(zip(s1["entity_id"].to_list(), s1["addr_nums"].to_list()))
    types, probs = zip(*MIX[c])
    out, k_id = [], 0
    for sid, mids, names, addrs in by.iter_rows():
        cnt = min(len(mids), int(rng.choice([1, 2, 3], p=[.15, .45, .40])))
        idx = rng.choice(len(mids), cnt, replace=False)
        t = str(rng.choice(types, p=np.array(probs) / sum(probs)))
        k = int(rng.choice([1, 2])) if t == "SHIFT12" else int(rng.choice(K_ALL[2:])) if t in ("SHIFTK", "MSHIDE") else 0
        tok = (t == "TOKEN") or (t.startswith("SHIFT") and rng.random() < 0.15)
        word = str(rng.choice(VOCAB)) if tok else ""
        targets = {int(x) for x in (s1n.get(sid) or "").split() if x.isdigit() and len(x) <= 7}
        subkey = int(rng.integers(1 << 30))
        for i in idx:
            name, addr = names[i], addrs[i]
            if t in ("SHIFT12", "SHIFTK"):
                addr = _shift(addr, targets, k, rng)
            elif t == "MSHIDE":
                addr = _shift(addr, targets, k, rng, multipart=True) or _shift(addr, targets, k, rng)
            elif t == "SUB":
                addr = _sub(addr, np.random.default_rng(subkey))    # same change on every copy
            if addr is None:
                continue
            if word:
                name = f"{name} {word.title()}"
            k_id += 1
            src = mids[i][:2]
            out.append((f"{src}-D{prefix}{k_id:07d}", name, addr, c, src, mids[i], t + ("+tok" if word and t != "TOKEN" else "")))
    return pl.DataFrame(out, schema=["entity_id", "business_name", "business_address", "country", "src", "orig_id", "dtype"], orient="row")


def normalize_rows(d, pool_schema):
    from normalize import normalize_address, normalize_name, phonetic_key
    rows = []
    for eid, nm, ad, c, src in d.select("entity_id", "business_name", "business_address", "country", "src").iter_rows():
        n, a = normalize_name(nm), normalize_address(ad, c)
        rows.append({"entity_id": eid, "country": c, "name_core": n["core"], "name_legal": n["legal"], "name_alts": n["alts"],
                     "name_phon": phonetic_key(n["core"]), "name_is_domain": n["is_domain"], "name_has_alt": n["has_alt"],
                     "name_non_latin": n["non_latin"], "addr_text": a["text"], "addr_state": a["state"], "addr_nums": a["nums"],
                     "addr_empty": a["empty"], "addr_native_state": a["native_state"], "src": src})
    df = pl.DataFrame(rows)
    return df.select([pl.col(k).cast(v) for k, v in pool_schema.items() if k in df.columns])


def recompute_cos(s1, pool2, dc):
    """Blocking cosines (as p1_block / add_fuzzy) for the decoy candidate rows, keeping their search bits."""
    from p1_block import K_STATELESS, BlockCfgV2, cos_all, fit_vectorizers, pairs_df, texts, vecs_for
    cfg = BlockCfgV2(verbose=False, k_stateless=K_STATELESS)
    ta, tb = texts(s1), texts(pool2)
    V = fit_vectorizers(ta, tb, cfg.max_df)
    pos = np.full(int(s1["i"].max()) + 1, -1, np.int64)
    pos[s1["i"].to_numpy()] = np.arange(s1.height)
    ii, jj = dc["i"].to_numpy(), dc["j"].to_numpy()
    ui, inv_i = np.unique(ii, return_inverse=True)
    uj, inv_j = np.unique(jj, return_inverse=True)
    XA, XB = vecs_for(V, ta, pos[ui], cfg.w_name), vecs_for(V, tb, uj, cfg.w_name)
    return pairs_df(ii, jj, dc["via"].to_numpy(), cos_all(XA, XB, inv_i, inv_j))


def stage_gen(a):
    from p1_features import country_context, iter_feature_chunks, load_country
    from p1_train import COUNTRIES, labels, role_mask
    rng = np.random.default_rng(a.seed)
    gt = labels().select("s1_id", "m_id")
    raw = pl.concat([pl.read_parquet(WORK / "parquet" / f"train_source{k}.parquet", columns=["entity_id", "business_name", "business_address"])
                     for k in (2, 3)])
    lab = labels()
    for c in COUNTRIES:
        s1, pool, cands = load_country("train", c)
        fz = WORK / "p1" / "train" / c / "cands_fz.parquet"
        if fz.exists():
            cands = pl.concat([cands, pl.read_parquet(fz).select(cands.columns)])
        rate = a.rate_us if c == "US" else a.rate_in
        dec_all = []
        for role, dname, pre in (("s2fit", "fitD", "F"), ("valid", "validD", "V")):
            ids = s1.filter(role_mask(c, role))["entity_id"].to_list()
            d = make_decoys(c, s1, raw, gt, set(ids), rate, rng, pre)
            log(f"{c}/{role}: {len(ids)} S1, {d.height} decoy records; types {dict(d.group_by('dtype').len().rows())}")
            dec_all.append(d)
        dec = pl.concat(dec_all)
        write_parquet_atomic(dec, WORK / "p1" / "train" / c / "decoys.parquet")
        drows = normalize_rows(dec, pool.drop("j").schema)
        j0 = pool.height
        pool2 = pl.concat([pool.drop("j"), drows]).with_row_index("j").with_columns(pl.col("j").cast(pl.Int32))
        jmap = pl.DataFrame({"orig_id": dec["orig_id"], "jd": np.arange(j0, j0 + dec.height, dtype=np.int32)})
        jmap = jmap.join(pool.select(pl.col("entity_id").alias("orig_id"), pl.col("j").alias("jo")), on="orig_id")
        dc = cands.join(jmap.select(pl.col("jo").alias("j"), "jd"), on="j").with_columns(pl.col("jd").alias("j")).drop("jd").select(cands.columns)
        dc = recompute_cos(s1, pool2, dc).select(cands.columns)      # the decoy's own blocking cosines, not its source's
        log(f"{c}: {dc.height} decoy candidate pairs (cosines recomputed)")
        cands2 = pl.concat([cands, dc])
        with timed(f"{c}: country context with decoys"):
            ctx, idf, freq = country_context(s1, pool2, cands2)
        for role, dname in (("s2fit", "fitD"), ("valid", "validD")):
            fdir = WORK / "p1" / "train" / c / f"feats_{dname}"
            fdir.mkdir(parents=True, exist_ok=True)
            n = 0
            for b, nb, f in iter_feature_chunks(s1, pool2, ctx, idf, freq, s1_mask=role_mask(c, role), labels=lab):
                write_parquet_atomic(f, fdir / f"part_{b:03d}.parquet")
                n += f.height
            log(f"  {c}/{dname}: {n} pairs ({nb} parts)")
        write_parquet_atomic(pool2.select(pl.col("entity_id").alias("m_id"), "name_core", "addr_text"), WORK / "p1" / "train" / c / "texts_decoy.parquet")


def s2_table(role, cols):
    """p1 (stage 1 of ER_MODEL_DIR) + stage-2 and sibling features for every pair of a feature folder."""
    from p1_train import P1_MIN, load_role_numpy, predict_p1, COUNTRIES
    from stack import sibling_features, stage2_features
    X, meta = load_role_numpy(role, cols)
    meta = meta.with_row_index("row").with_columns(pl.Series("p1", predict_p1(X).astype(np.float32)))
    texts = pl.concat([pl.read_parquet(WORK / "p1" / "train" / c / "texts_decoy.parquet") for c in COUNTRIES])
    s2 = stage2_features(meta.select("row", "s1_id", "m_id", "p1")).filter(pl.col("p1") >= P1_MIN)
    s2 = s2.join(sibling_features(meta.select("s1_id", "m_id", "p1"), texts, P1_MIN), on=["s1_id", "m_id"], how="left").sort("row")
    return X, meta, s2


def stage_s2(a):
    import lightgbm as lgb
    from p1_train import M, PARAMS
    cols = json.loads((M / "feature_cols.json").read_text())
    allc = json.loads((M / "stage2_cols.json").read_text())
    s2cols = allc[len(cols):]
    X, meta, s2 = s2_table("fitD", cols)
    rows = s2["row"].to_numpy()
    X2 = np.hstack([X[rows], s2.select([pl.col(c).cast(pl.Float32) for c in s2cols]).to_numpy()])
    y = meta["y"].to_numpy()[rows]
    es = meta["s1_grp"].to_numpy()[rows] % 7 == 3
    dec = meta["m_id"].str.contains("-D")[rows].to_numpy()
    log(f"stage-2 rows {len(y)}, positive {y.mean():.4f}, decoy rows {dec.sum()} (p1 >= filter)")
    del X
    with timed("stage 2 with decoys"):
        m = lgb.train(PARAMS, lgb.Dataset(X2[~es], y[~es], feature_name=allc), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X2[es], y[es], feature_name=allc)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    out = WORK / "p1" / a.out
    out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out / "stage2.txt"))
    log(f"new stage 2 best iter {m.best_iteration} -> {out / 'stage2.txt'}")


def stage_eval(a):
    import lightgbm as lgb
    from decide import by_threshold, one_s1_per_record, to_sets
    from metric import entity_f05
    from p1_train import COUNTRIES, M, labels, role_mask
    cols = json.loads((M / "feature_cols.json").read_text())
    allc = json.loads((M / "stage2_cols.json").read_text())
    s2cols = allc[len(cols):]
    s1v = pl.concat([pl.read_parquet(WORK / "p1" / "train" / c / "s1.parquet", columns=["entity_id", "addr_state", "country"])
                     .filter(role_mask(c, "valid")) for c in COUNTRIES])
    truth = to_sets(labels().join(s1v.select(pl.col("entity_id").alias("s1_id")), on="s1_id"))
    ids, ctry = s1v["entity_id"].to_list(), np.array(s1v["country"].to_list())
    half = (s1v["entity_id"].hash(13) % 2).to_numpy()
    old, new = lgb.Booster(model_file=str(M / "stage2.txt")), lgb.Booster(model_file=str(WORK / "p1" / a.out / "stage2.txt"))
    res = {}
    for role in ("valid", "validD"):
        X, meta, s2 = s2_table(role, cols)
        rows = s2["row"].to_numpy()
        X2 = np.hstack([X[rows], s2.select([pl.col(c).cast(pl.Float32) for c in s2cols]).to_numpy()])
        del X
        base = meta[rows].select("s1_id", "m_id", "y")
        for nm, bst in (("old", old), ("new", new)):
            sc = base.with_columns(pl.Series("p", bst.predict(X2)))
            write_parquet_atomic(sc, WORK / "p1" / a.out / f"scored_{role}_{nm}.parquet")
            P = to_sets(by_threshold(one_s1_per_record(sc), 0.70))
            f = np.array([entity_f05(P.get(s, set()), truth.get(s, set())) for s in ids])
            res[f"{role}/{nm}"] = {"macro": round(f.mean(), 5), "US": round(f[ctry == "US"].mean(), 5), "India": round(f[ctry == "India"].mean(), 5),
                                   "h0": round(f[half == 0].mean(), 5), "h1": round(f[half == 1].mean(), 5)}
            log(f"{role}/{nm}: {res[f'{role}/{nm}']}")
    (WORK / "p1" / a.out / "eval.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "s2", "eval"])
    ap.add_argument("--out", default="model_v9_decoy")
    ap.add_argument("--rate-us", type=float, default=45)
    ap.add_argument("--rate-in", type=float, default=25)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"gen": stage_gen, "s2": stage_s2, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
