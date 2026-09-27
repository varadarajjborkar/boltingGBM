# US/India exact-match adds (msrit, 27 Sep 14:05 IST) - same rule as the France fix, unique case only

Test blocking also misses exact pairs outside France: records unassigned in stack_e5hyb2 with exactly ONE S1 of the country
having the same name_core + house-number multiset, street-word Jaccard >= 0.5, raw body equal after accent folding and
legal-form removal, legal forms equal or one side without one, and no decoy-vocabulary word added over the S1's raw name.
exports/usin_fix/usin_exact_adds.parquet (s1_id, m_id, country): **US 1,827 (2.76 per 1k S1), India 260 (0.32)**; 99% / 98% are
NOT in candidate_pairs (VALID rate for the same pocket: 0.18 per 1k, precision 0.968 on 31 records; the unique key overall is
the owner for 99.99% US / 99.94% India per your own check). Street matters: with street Jaccard < 0.5 the same key is
right only 0.2-7% of the time on VALID (other states' businesses), so do not relax it. Legal forms matter in India
('Perfect Enterprises LLP' vs '... Private Limited', same address = distractor), hence the legal-form condition.
Expected LB +0.0002. Apply like the France adds: only where the record is still unassigned in your final file (skip rows
covered by the a1 rescue), add the pairs to candidate_pairs, sub_check + validator.

Also checked (no action): decided pairs by house-number difference kind match VALID on test (up-shifted decoys: 0.00 per 1k
kept in US and France; +1/+2 shifts 14.1 per 1k US test vs 14.7 VALID at precision 1.000), so the decoy rules are not leaking.
France keeps few different-number pairs only because French addresses carry one house number and no postal code.

## 14:15 update after your advisor's France street finding
Re-checked with an IDF-weighted street match (generic words such as st/ave/road/city names weigh ~0): **1,976 of the 2,087
adds keep weighted street Jaccard >= 0.5** (US 1,738, India 238) -> exports/usin_fix/usin_exact_adds_samestreet.parquet
(use this one). Calibration on VALID for the same unique key: weighted Jaccard >= .5 precision 1.000 (US 94,991, India 93,761
records), .3-.5 1.000, < .3 0.988 / 0.993: US/India names are distinctive, unlike the generic French ones. Expected LB +0.0002.
