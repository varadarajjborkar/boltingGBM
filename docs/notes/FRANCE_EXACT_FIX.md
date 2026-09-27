# France exact-match fix (msrit, 27 Sep 13:15 IST) - apply on top of stack_pkf_a1_fr (France rows = stack_ce)

**Finding.** French blocking misses exact pairs. 19.6 per 1k French S1: record unassigned although a UNIQUE S1 has the same
name_core, same house-number multiset and same street (US 3.1/1k, India 1.1/1k); for 93% of them the pair (S1, record) is not
in candidate_pairs at all. Examples: 'Bordeaux Elémentaire SARL' vs S1 'Bordeaux Élémentaire SARL' (accent), 'SARL Calais Section'
vs 'Calais Section SAS'. On VALID the same key (unique S1, same core+numbers+street) is the owner in 99.99% of 200,718 records.
French S1 also holds 98.9 S1 per 1k in same-core same-address groups that differ only by legal form (US 2.7, India 13.9 and
none of those share a street): there the legal form decides.

**Lists** (exports/france_fix/, s1_id = the S1 to assign, m_id = record):
| file | rows | rule |
|---|---|---|
| fr_exact_adds.parquet | 4,118 (3,135 unique + 983 group) | record unassigned in stack_e5hyb2; unique S1 with same core+numbers+street, raw body equal after accent folding and legal-form removal, legal forms equal or one side without one; or a same-address group with exactly one member of the same body AND legal form |
| fr_exact_moves.parquet | 905 (257 + 648), column from_s1_id | record currently at from_s1_id whose raw body differs from the record's, while the S1 above matches exactly |

Checks: identical raw duplicates in training always share the owner (58,176/58,176 groups); only 0.1-0.2% of these records
duplicate a record the S1 already holds; the dense-retrieval cross-encoder scored 2,952 of the 2,986 pairs it saw >= 0.5.
Expected: France macro +0.0019..+0.0037 -> **LB +0.0003..+0.0006** (precision 0.8..1.0). 363 of the 3,662 S1 have no pair today.

**Apply** (per row, only if the record's current assignment in YOUR final file is still what the list assumes: unassigned
for adds, from_s1_id for moves; skip rows touched by the 389 French adds): remove (from_s1_id, m_id), add (s1_id, m_id);
assert no record in 2 S1; add the new pairs to candidate_pairs.tsv (+5,023 pairs, +0.003 per S1); run sub_check + validator.
