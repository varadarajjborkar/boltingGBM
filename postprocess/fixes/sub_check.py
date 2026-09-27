"""Submission-pair integrity for merged files (stack hybrids, India restore, A1 rescue). Pure python, streaming, no polars.

merge_candidates(base_cand, adds, out): out = base candidate_pairs.tsv + the (s1_id, m_id) pairs in `adds` (dict s1 -> ids),
  row order and header kept, ids already listed for that S1 not repeated. Returns counts.
check(matching, cand): one row per S1 in the same order in both files, no duplicate S1 rows, no duplicate id inside a
  list, no record under two S1 (one-S1-per-record), every matched id is a candidate of the SAME S1, only S2-/S3- ids.
  Prints candidate pairs per S1 and matched pairs per S1. Exit code 1 on any violation.
Usage: python postprocess/fixes/sub_check.py <dir with matching_results.tsv + candidate_pairs.tsv>   (~2-4 min on test, < 3 GB)"""
import sys
from itertools import zip_longest

MH, CH = "source1_entity_id\tmatched_entity_ids", "source1_entity_id\tcandidate_entity_ids"


def _rows(path, header):
    with open(path, encoding="utf-8") as fh:
        h = fh.readline().rstrip("\n")
        assert h == header, f"{path}: header {h!r} != {header!r}"
        for ln in fh:
            s, _, v = ln.rstrip("\n").partition("\t")
            yield s, [x for x in v.split(",") if x]


def merge_candidates(base_cand, adds, out):
    """adds: dict s1_id -> iterable of m_id. Every key must be a row of base_cand."""
    left = {k: list(dict.fromkeys(v)) for k, v in adds.items()}
    n_rows = n_base = n_add = n_dup = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(CH + "\n")
        for s, ids in _rows(base_cand, CH):
            n_rows += 1
            n_base += len(ids)
            extra = left.pop(s, [])
            if extra:
                have = set(ids)
                new = [m for m in extra if m not in have]
                n_dup += len(extra) - len(new)
                n_add += len(new)
                ids = ids + new
            fo.write(f"{s}\t{','.join(ids)}\n")
    assert not left, f"{len(left)} S1 of the adds are not rows of {base_cand}"
    return {"rows": n_rows, "base_pairs": n_base, "added_pairs": n_add, "already_candidates": n_dup}


def check(matching, cand):
    owner, bad = {}, []
    seen_s1 = set()
    n_s1 = n_match = n_cand = 0
    for a, b in zip_longest(_rows(matching, MH), _rows(cand, CH)):
        if a is None or b is None:
            bad.append(f"row counts differ (matching ended: {a is None}, candidates ended: {b is None})")
            break
        (sm, mids), (sc, cids) = a, b
        n_s1 += 1
        if sm != sc:
            bad.append(f"row {n_s1}: S1 order differs ({sm} vs {sc})")
            break
        if sm in seen_s1:
            bad.append(f"duplicate S1 row {sm}")
        seen_s1.add(sm)
        n_match += len(mids)
        n_cand += len(cids)
        if len(set(mids)) != len(mids):
            bad.append(f"{sm}: duplicate id inside the matched list")
        cs = set(cids)
        for m in mids:
            if not (m.startswith("S2-") or m.startswith("S3-")):
                bad.append(f"{sm}: bad id {m}")
            if m not in cs:
                bad.append(f"{sm}: matched {m} not a candidate of this S1")
            o = owner.setdefault(m, sm)
            if o != sm:
                bad.append(f"{m} matched under two S1 ({o}, {sm})")
        if len(bad) > 50:
            break
    print(f"S1 rows {n_s1:,}; candidate pairs {n_cand:,} ({n_cand / max(n_s1, 1):.2f}/S1); matched {n_match:,} "
          f"({n_match / max(n_s1, 1):.2f}/S1)")
    for b in bad[:50]:
        print("VIOLATION", b)
    print("sub_check", "PASS" if not bad else f"FAIL ({len(bad)}+ violations)")
    return not bad


if __name__ == "__main__":
    d = sys.argv[1].rstrip("/\\")
    sys.exit(0 if check(f"{d}/matching_results.tsv", f"{d}/candidate_pairs.tsv") else 1)
