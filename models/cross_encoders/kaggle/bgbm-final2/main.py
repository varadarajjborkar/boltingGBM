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
req = [l.strip() for l in open(REQ) if l.strip() and not l.startswith("#") and not l.startswith(("lleaves", "llvmlite", "optuna"))]
t = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + req, check=True)
print(f"code {CODE} | data {DATA} | libraries in {time.time()-t:.0f}s", flush=True)
S1P = '{"learning_rate":0.05,"num_leaves":511,"min_data_in_leaf":300,"feature_fraction":0.7,"bagging_fraction":0.8,"lambda_l2":0.0}'
ENV = {**os.environ, "ER_WORK_DIR": W, "ER_NORM_SUBDIR": "norm_v2", "ER_THREADS": "4", "PYTHONUNBUFFERED": "1",
       "ER_FIT_MODE": "entities", "ER_FIT_PCT": "25", "ER_S2_STATES": "tx,up,ka", "ER_MODEL_DIR": f"{W}/p1/model_k1",
       "ER_S1_SINGLE": "1", "ER_S1_PATIENCE": "100", "ER_S1_PARAMS": S1P}
def run(args, env=None):
    t = time.time()
    print("RUN", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=SRC, env=env or ENV)
    print(f"EXIT {r.returncode} after {(time.time()-t)/60:.1f} min", flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)
import json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=False)
link_split("train", ["US", "India"]); link_split("test", ["France", "India", "US"])
mf = find("model_k1/feature_cols.json")[0]; shutil.copytree(os.path.dirname(mf), f"{W}/p1/model_k1", dirs_exist_ok=True)
for c in ["France", "India", "US"]:
    shutil.copy(find(f"p1/test/{c}/stage2_scored_model_k1.parquet", prefer="bgbm-merge")[0], f"{W}/p1/test/{c}/")
M = f"{W}/p1/model_k1"
# charnet artefacts from bgbm-final (no retraining), cross-encoder scores from bgbm-ce / bgbm-ce-test
shutil.copytree(os.path.dirname(find("stage3_k1/cn_valid.parquet")[0]), f"{M}/stage3", dirs_exist_ok=True)
CEV = find("ce_valid.parquet", prefer="bgbm-ce")[0]
CET = os.path.dirname(find("ce_test_US.parquet")[0]) + "/ce_test_{c}.parquet"
print("CE files:", CEV, CET, flush=True)
T3 = f"{CODE}/tools/stage3.py"
t = time.time()
r = subprocess.run([sys.executable, "-W", "ignore", T3, "fit", "--model", M, "--extra", f"ce={CEV}@{CET}"], cwd=SRC, env=ENV)
print(f"stage3 fit+ce exit {r.returncode} ({(time.time()-t)/60:.1f} min)", flush=True)
P = "/kaggle/working/proj"
shutil.copytree(CODE, P, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
os.symlink(W, f"{P}/work") if not os.path.exists(f"{P}/work") else None
tsv = os.path.dirname(find("data/student_resource/dataset/test/test_source1.tsv")[0])
os.makedirs(f"{P}/data/student_resource/dataset", exist_ok=True)
if not os.path.exists(f"{P}/data/student_resource/dataset/test"):
    os.symlink(tsv, f"{P}/data/student_resource/dataset/test")
TG = "shift_pure=1.01,shift_ms_only=1.01,shift_12=0.95,shift_12|corr=0.98,missing|corr=0.98"
import json as _j
old = _j.load(open(find("stage3_k1/report.json")[0]))
rep = _j.load(open(f"{M}/stage3/report.json"))
print("STAGE3 charnet only:", _j.dumps({k: old[k] for k in ("stage2_only", "stage3", "threshold")}, default=float), flush=True)
print("STAGE3 charnet + CE:", _j.dumps({k: rep[k] for k in ("stage2_only", "stage3", "threshold")}, default=float), flush=True)
print(f"CE over charnet: macro {rep['stage3']['macro'] - old['stage3']['macro']:+.5f} | h0 {rep['stage3']['h0'] - old['stage3']['h0']:+.5f} "
      f"h1 {rep['stage3']['h1'] - old['stage3']['h1']:+.5f} | US {rep['stage3']['US'] - old['stage3']['US']:+.5f} "
      f"India {rep['stage3']['India'] - old['stage3']['India']:+.5f}", flush=True)
VOC = "/kaggle/working/decoy_vocab_validated.json"
json.dump(['advisors', 'agro', 'associates', 'bakery', 'biotech', 'builders', 'capital', 'care', 'cargo', 'central', 'coastal', 'consultants', 'consulting', 'data', 'deli', 'dental', 'design', 'developers', 'digital', 'downtown', 'east', 'eastgate', 'electric', 'energy', 'exports', 'farms', 'finance', 'financial', 'foods', 'greater', 'grill', 'group', 'harbor', 'health', 'healthcare', 'highland', 'holdings', 'hospitality', 'india', 'industries', 'infotech', 'infra', 'infratech', 'interiors', 'international', 'investments', 'lakeside', 'liquor', 'logistics', 'marketing', 'media', 'medical', 'metro', 'midtown', 'motors', 'north', 'northside', 'overseas', 'producer', 'products', 'projects', 'properties', 'realty', 'research', 'restaurant', 'retail', 'riverside', 'salon', 'sciences', 'security', 'software', 'solutions', 'south', 'southside', 'spa', 'summit', 'supply', 'systems', 'tax', 'textiles', 'therapy', 'trading', 'transport', 'uptown', 'valley', 'ventures', 'west', 'westgate', 'works'], open(VOC, "w"))
import json
for dst, extra in (("k1_s3ce_rules", []), ("k1_s3ce_voc", ["--drop-vocab", VOC])):
    r = subprocess.run([sys.executable, f"{P}/postprocess/postprocess.py", "--scores", "work", "--texts", "work", "--model", "model_k1_s3",
                        "--t", str(rep["threshold"]), "--tg", TG, "--dst", dst] + extra, cwd=P, capture_output=True, text=True)
    print(dst, "exit", r.returncode, r.stdout[-1500:], r.stderr[-1500:], flush=True)
    v = subprocess.run([sys.executable, f"{P}/data/student_resource/utils/validate_submission.py" if os.path.exists(f"{P}/data/student_resource/utils/validate_submission.py") else find("utils/validate_submission.py")[0],
                        "--matching", f"{P}/output_bucket/{dst}/matching_results.tsv", "--candidate", f"{P}/output_bucket/{dst}/candidate_pairs.tsv",
                        "--test-dir", f"{P}/data/student_resource/dataset/test"], capture_output=True, text=True)
    print(dst, "VALIDATOR", v.returncode, (v.stdout or "")[-600:], flush=True)
FR_TOOL = "/kaggle/working/france_rule.py"
open(FR_TOOL, "w", encoding="utf-8").write('"""France decoy rule (docs/agents_msrit/FRANCE_ANALYSIS.md, rule R2): drop a French predicted pair when the record\'s name\nADDS, relative to its S1 name, one of the French decoy tokens {holding, distribution, participations, international,\nfrance, snc}. Tokens come from the accent-folded raw business_name (the normaliser removes \'france\' and legal forms, so\nraw names are used for every token). An added token that is a near-duplicate of an S1 token (same after dropping a\nfinal \'s\', or edit distance 1 for tokens of length >= 5) is not counted. US and India rows are copied unchanged.\n\nLabel-free: on k1 the agent counted 6,056 dropped pairs (23.3 per 1000 French S1), break-even precision 0.724, estimated\n+0.0005 LB (+0.00084 if every dropped pair is a decoy). Judge it with one LB A/B against the same file without the rule.\n\nUsage: python tools/france_rule.py --src <dir>/matching_results.tsv --test-dir data/student_resource/dataset/test --out <dir2>\n(candidate_pairs.tsv next to --src is copied when present).\n"""\nimport argparse\nimport re\nimport shutil\nimport unicodedata\nfrom pathlib import Path\n\nimport polars as pl\n\nVOCAB = {"holding", "distribution", "participations", "international", "france", "snc"}\n\n\ndef toks(s):\n    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()\n    return set(re.findall(r"[a-z0-9]+", s))\n\n\ndef near(t, pool):\n    for u in pool:\n        if t.rstrip("s") == u.rstrip("s"):\n            return True\n        if len(t) >= 5 and len(u) >= 5 and abs(len(t) - len(u)) <= 1:\n            # edit distance <= 1\n            if len(t) == len(u) and sum(a != b for a, b in zip(t, u)) <= 1:\n                return True\n            a, b = (t, u) if len(t) < len(u) else (u, t)\n            if any(a == b[:i] + b[i + 1:] for i in range(len(b))):\n                return True\n    return False\n\n\ndef flag(s1_name, rec_name):\n    a, b = toks(s1_name), toks(rec_name)\n    return any(t in VOCAB and not near(t, a) for t in b - a)\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--src", required=True)\n    ap.add_argument("--test-dir", required=True)\n    ap.add_argument("--out", required=True)\n    a = ap.parse_args()\n    T = Path(a.test_dir)\n    rd = lambda f: pl.read_csv(T / f, separator="\\t", quote_char=None, infer_schema=False,\n                               columns=["entity_id", "business_name", "country"]).fill_null("")\n    s1 = rd("test_source1.tsv")\n    fr = s1.filter(pl.col("country") == "France").select(pl.col("entity_id").alias("source1_entity_id"),\n                                                         pl.col("business_name").alias("n1"))\n    src = pl.read_csv(a.src, separator="\\t", quote_char=None, infer_schema=False).fill_null("")\n    pairs = (src.join(fr, on="source1_entity_id").with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))\n                .explode("m").filter(pl.col("m") != ""))\n    rec = pl.concat([rd(f"test_source{s}.tsv").filter(pl.col("country") == "France") for s in (2, 3)]).select(\n        pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"))\n    pairs = pairs.join(rec, on="m", how="left").fill_null("")\n    drop = pairs.filter(pl.Series([flag(x, y) for x, y in zip(pairs["n1"].to_list(), pairs["n2"].to_list())]))\n    dset = set(zip(drop["source1_entity_id"].to_list(), drop["m"].to_list()))\n    fr_ids = set(fr["source1_entity_id"].to_list())\n    new = [",".join(x for x in ids.split(",") if x and (s, x) not in dset) if s in fr_ids else ids\n           for s, ids in zip(src["source1_entity_id"].to_list(), src["matched_entity_ids"].to_list())]\n    out = src.with_columns(pl.Series("matched_entity_ids", new))\n    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)\n    out.write_csv(o / "matching_results.tsv", separator="\\t", quote_style="never")\n    cp = Path(a.src).parent / "candidate_pairs.tsv"\n    if cp.exists():\n        shutil.copy(cp, o / "candidate_pairs.tsv")\n    n_fr = len(fr_ids)\n    msg = (f"France rule R2: {len(dset)} French pairs dropped ({1000 * len(dset) / max(n_fr, 1):.1f} per 1000 French S1), "\n           f"{drop[\'source1_entity_id\'].n_unique()} S1 touched, of {pairs.height} French predicted pairs")\n    (o / "FRANCE_RULE.txt").write_text(msg + f"\\nsource: {a.src}\\n")\n    print(msg)\n\n\nif __name__ == "__main__":\n    main()\n')
r = subprocess.run([sys.executable, FR_TOOL, "--src", f"{P}/output_bucket/k1_s3ce_voc/matching_results.tsv",
                    "--test-dir", f"{P}/data/student_resource/dataset/test", "--out", f"{P}/output_bucket/k1_s3ce_voc_fr"],
                   capture_output=True, text=True)
print("k1_s3ce_voc_fr exit", r.returncode, r.stdout[-800:], r.stderr[-800:], flush=True)
v = subprocess.run([sys.executable, f"{P}/data/student_resource/utils/validate_submission.py", "--matching",
                    f"{P}/output_bucket/k1_s3ce_voc_fr/matching_results.tsv", "--candidate",
                    f"{P}/output_bucket/k1_s3ce_voc_fr/candidate_pairs.tsv", "--test-dir", f"{P}/data/student_resource/dataset/test"],
                   capture_output=True, text=True)
print("k1_s3ce_voc_fr VALIDATOR", v.returncode, (v.stdout or "")[-600:], flush=True)
for c in ["France", "India", "US"]:
    f = f"{W}/p1/test/{c}/stage2_scored_model_k1_s3.parquet"
    if os.path.exists(f):
        os.makedirs(f"/kaggle/working/s3ce_scores/{c}", exist_ok=True); shutil.copy(f, f"/kaggle/working/s3ce_scores/{c}/")
shutil.copytree(f"{M}/stage3", "/kaggle/working/stage3_k1_ce", dirs_exist_ok=True)
for l in (f"{P}/work", f"{P}/data/student_resource/dataset/test"):
    if os.path.islink(l):
        os.unlink(l)
print("FINAL2_DONE", os.listdir(f"{P}/output_bucket") if os.path.exists(f"{P}/output_bucket") else "no bucket", flush=True)
