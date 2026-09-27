"""Competition metric: per-Source-1-entity F0.5, macro-averaged over ALL S1 entities.

Rules (from the problem statement):
- entity with no true matches: 1.0 if we predict an empty list, else 0.0
- otherwise F0.5 = 1.25*P*R / (0.25*P + R); an empty prediction scores 0.0
"""
from typing import Dict, Iterable, Set


def entity_f05(pred: Set[str], true: Set[str]) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], s1_ids: Iterable[str] = None) -> float:
    """Average over s1_ids (default: every key of truth). Missing predictions count as empty."""
    ids = list(truth) if s1_ids is None else list(s1_ids)
    return sum(entity_f05(pred.get(s, set()), truth.get(s, set())) for s in ids) / len(ids)


def oracle_f05(candidates: Dict[str, Set[str]], truth: Dict[str, Set[str]], s1_ids: Iterable[str] = None) -> float:
    """Best achievable score if the matcher were perfect on these candidates (blocking ceiling)."""
    ids = list(truth) if s1_ids is None else list(s1_ids)
    return sum(entity_f05(candidates.get(s, set()) & truth.get(s, set()), truth.get(s, set())) for s in ids) / len(ids)


def parse_id_list(s: str) -> Set[str]:
    return set(s.split(",")) if s else set()


if __name__ == "__main__":
    # worked example from the problem statement: expected 0.714
    got = entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(got - 0.714) < 1e-3, got
    assert entity_f05(set(), set()) == 1.0 and entity_f05({"S2-1"}, set()) == 0.0
    assert entity_f05(set(), {"S2-1"}) == 0.0
    print(f"metric self-test OK: example = {got:.4f}")
