"""Final submission health check, run on the EXACT files that go into the zip. Complements utils/validate_submission.py (format).
Checks and prints:
  1. header / TAB / one row per test S1, every test S1 present exactly once, no extra S1
  2. every matched id exists in test S2/S3, has an S2-/S3- prefix, no S1 id or blank inside a list, no duplicate id in a row
  3. no record assigned to two S1 (the pipeline is one-S1-per-record; a duplicate means a merge bug)
  4. no cross-country pair (record country != S1 country)
  5. matches are a subset of candidate_pairs (the scorer only warns, but a rider added to matches and not to candidates is a bug)
  6. candidate-set size per S1 (it counts in the final ranking) and per-country match histograms: share of empty S1, P(k=1),
     mean k (generator, all train states: 5.5% of S1 have no record, 5.4% have one, mean 3.465)
Usage: python models/cross_encoders/final_check.py <matching_results.tsv> <candidate_pairs.tsv> <dir with test_source1/2/3 .parquet or .tsv>
Stdlib + polars. About 1-2 minutes, 4-6 GB."""
import sys
from collections import Counter
from pathlib import Path
import polars as pl


def read_src(d, k):
    p = Path(d) / f"test_source{k}.parquet"
    if p.exists():
        return pl.read_parquet(p, columns=["entity_id", "country"])
    return pl.read_csv(Path(d) / f"test_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id", "country"])


def read_lists(path, col):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return head, (df.rename({df.columns[0]: "s1", df.columns[1]: "ids"}) if df.width >= 2 else df)


def main(match_path, cand_path, src_dir):
    bad = []
    s1 = read_src(src_dir, 1)
    rec = pl.concat([read_src(src_dir, 2), read_src(src_dir, 3)])
    S1C = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    RC = dict(zip(rec["entity_id"].to_list(), rec["country"].to_list()))
    head, m = read_lists(match_path, "matched_entity_ids")
    if head != ["source1_entity_id", "matched_entity_ids"]:
        bad.append(f"matching header {head}")
    rows = m["s1"].to_list()
    dup_rows = [s for s, c in Counter(rows).items() if c > 1]
    missing = set(S1C) - set(rows)
    extra = set(rows) - set(S1C)
    print(f"[1] rows {len(rows)} | test S1 {len(S1C)} | duplicate S1 rows {len(dup_rows)} | missing S1 {len(missing)} | unknown S1 {len(extra)}")
    if dup_rows or missing or extra:
        bad.append("S1 row set")
    owner = {}
    two = set()
    unknown = prefix = inrow_dup = blank = xcountry = 0
    k_by_c = {c: Counter() for c in set(S1C.values())}
    for s, ids in zip(rows, m["ids"].to_list()):
        lst = [x for x in ids.split(",")] if ids else []
        if any(x.strip() == "" for x in lst):
            blank += 1
        lst = [x.strip() for x in lst if x.strip()]
        if len(lst) != len(set(lst)):
            inrow_dup += 1
        for x in set(lst):
            if not (x.startswith("S2-") or x.startswith("S3-")):
                prefix += 1
            if x not in RC:
                unknown += 1
            elif s in S1C and RC[x] != S1C[s]:
                xcountry += 1
            if x in owner and owner[x] != s:
                two.add(x)
            owner[x] = s
        if s in S1C:
            k_by_c[S1C[s]][min(len(set(lst)), 8)] += 1
    print(f"[2] matched ids {len(owner)} | not in S2/S3 {unknown} | bad prefix {prefix} | rows with a duplicate id {inrow_dup} | rows with a blank id {blank}")
    print(f"[3] records assigned to two or more S1: {len(two)}" + (f" e.g. {sorted(two)[:3]}" if two else ""))
    print(f"[4] cross-country pairs: {xcountry}")
    for name, v in (("unknown id", unknown), ("prefix", prefix), ("in-row duplicate", inrow_dup), ("blank id", blank), ("two S1", len(two)), ("cross-country", xcountry)):
        if v:
            bad.append(name)
    if cand_path and Path(cand_path).exists():
        chead, c = read_lists(cand_path, "candidate_entity_ids")
        if chead != ["source1_entity_id", "candidate_entity_ids"]:
            bad.append(f"candidate header {chead}")
        cand = {}
        for s, ids in zip(c["s1"].to_list(), c["ids"].to_list()):
            cand.setdefault(s, set()).update(x.strip() for x in ids.split(",") if x.strip())
        outside = sum(1 for x, s in owner.items() if x not in cand.get(s, ()))
        n_c = sum(len(v) for v in cand.values())
        print(f"[5] matched pairs not in candidate_pairs: {outside} | candidate pairs {n_c} = {n_c / len(S1C):.3f} per S1")
        if outside:
            bad.append("matches outside candidates")
    else:
        print("[5] candidate_pairs not given: skipped (it is still expected in the zip)")
    print("[6] per country: S1, empty share, P(k=1), mean k   (generator: empty 5.5%, one 5.4%, mean 3.465 at full recall)")
    for cc, h in sorted(k_by_c.items()):
        n = sum(h.values())
        mean = sum(k * v for k, v in h.items()) / max(n, 1)
        print(f"    {cc:7s} S1 {n:8d} empty {h[0] / n:.4f} one {h[1] / n:.4f} mean k {mean:.3f}")
    print("RESULT:", "OK" if not bad else "PROBLEMS -> " + ", ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "", sys.argv[3] if len(sys.argv) > 3 else "."))
