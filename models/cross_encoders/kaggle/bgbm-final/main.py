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
    src = find(f"p1/test/{c}/stage2_scored_model_k1.parquet")[0]
    shutil.copy(src, f"{W}/p1/test/{c}/")
M = f"{W}/p1/model_k1"
T3 = f"{CODE}/tools/stage3.py"
t = time.time()
for args in (["charnet-train", "--model", M, "--max-train", "600000", "--epochs", "3", "--threads", "4"],
             ["charnet-score", "--model", M, "--threads", "4"], ["fit", "--model", M]):
    r = subprocess.run([sys.executable, "-W", "ignore", T3] + args, cwd=SRC, env=ENV)
    print(f"stage3 {args[0]} exit {r.returncode} ({(time.time()-t)/60:.1f} min)", flush=True)
    if r.returncode:
        break
# project copy with writable output_bucket; work/ and raw test TSVs linked in
P = "/kaggle/working/proj"
shutil.copytree(CODE, P, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.zip"))
os.symlink(W, f"{P}/work") if not os.path.exists(f"{P}/work") else None
tsv = os.path.dirname(find("data/student_resource/dataset/test/test_source1.tsv")[0])
os.makedirs(f"{P}/data/student_resource/dataset", exist_ok=True)
if not os.path.exists(f"{P}/data/student_resource/dataset/test"):
    os.symlink(tsv, f"{P}/data/student_resource/dataset/test")
TG = "shift_pure=1.01,shift_ms_only=1.01,shift_12=0.95,shift_12|corr=0.98,missing|corr=0.98"
rep = json.load(open(f"{M}/stage3/report.json")) if os.path.exists(f"{M}/stage3/report.json") else None
import json as _j
print("VALID model_k1:", _j.dumps(_j.load(open(f"{M}/valid_report.json"))["best"]), flush=True)
if rep:
    print("STAGE3 report:", _j.dumps({k: rep[k] for k in ("stage2_only", "stage3", "threshold")}, default=float), flush=True)
jobs = [("model_k1", 0.70, "k1_rules")] + ([("model_k1_s3", rep["threshold"], "k1_s3_rules")] if rep else [])
for model, t_, dst in jobs:
    r = subprocess.run([sys.executable, f"{P}/postprocess/postprocess.py", "--scores", "work", "--texts", "work", "--model", model,
                        "--t", str(t_), "--tg", TG, "--dst", dst], cwd=P, capture_output=True, text=True)
    print(dst, "exit", r.returncode, r.stdout[-1500:], r.stderr[-1500:], flush=True)
print("FINAL_DONE", os.listdir(f"{P}/output_bucket") if os.path.exists(f"{P}/output_bucket") else "no bucket", flush=True)
# keep stage-3 artefacts as real files for a follow-up kernel (cross-encoder extra), drop symlinks out of /kaggle/working
if os.path.isdir(f"{M}/stage3"):
    shutil.copytree(f"{M}/stage3", "/kaggle/working/stage3_k1", dirs_exist_ok=True)
for c in ["France", "India", "US"]:
    f = f"{W}/p1/test/{c}/stage2_scored_model_k1_s3.parquet"
    if os.path.exists(f):
        os.makedirs(f"/kaggle/working/s3_scores/{c}", exist_ok=True); shutil.copy(f, f"/kaggle/working/s3_scores/{c}/")
for l in (f"{P}/work", f"{P}/data/student_resource/dataset/test"):
    if os.path.islink(l):
        os.unlink(l)
print("CLEANUP_DONE", os.listdir("/kaggle/working"), flush=True)
