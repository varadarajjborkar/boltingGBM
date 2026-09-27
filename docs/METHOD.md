# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** amazingRIVER
**Team Members:** Varadaraj, Adit
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary
A precision-oriented, four-stage pipeline:
1. **Deterministic normalisation** undoes the synthetic noise patterns (junk tokens, leetspeak, random accents, native-script transliteration, legal-form and address-abbreviation variants, state/region parsing).
2. **High-recall blocking within each state/region** unions ten TF-IDF top-k search paths.
3. **A two-stage LightGBM matcher** uses about 85 string, numeric, frequency and competition-context features.
4. **A set-level decision layer** enforces one Source-1 entity per record and thresholds for macro F0.5.
5. **Stage 3 (final files):** a LightGBM stacker, fitted on the validation states only, combines the stage-2 score with
   a character-level CNN pair model, a context-aware gradient-boosting model (`models/context_gbm/`), three fine-tuned
   transformer cross-encoders (MiniLM-L6, MiniLM-L12, multilingual e5-small) and 15 context features.
6. **Rules and rescue:** house-number shift rules and a decoy-vocabulary rule; India shift pairs restored at p >= 0.98;
   France taken from the stack without e5 and context; an add-only rescue for never-compared records with an address
   (address-word containment key, v10 stage 2, p >= 0.9).
7. **Label-free post-processing (final file):** test-only decoy families (French descriptor swaps, US house-number
   offsets with an added legal form, India legal-type changes) were found with a truth-density test (test pairs per
   1,000 S1 against the training-truth density of the same cell) and removed; a blend of stage-3 LightGBM variants
   added or dropped marginal pairs where validation showed a gain on both halves. This took the public LB from
   0.981 to 0.988757. Exact pair lists: `code/business_entity_resolution/postprocess/final/`.

Our main innovations:
- a validation design that reproduces the test set's doubled look-alike density
- stage-2 stacking on out-of-fold scores, so a pair is judged against its competitors

---

## 2. Methodology

### 2.1 Problem Analysis
- **Singletons are rare.** Only 5.6% of S1 entities have no match. The mean is 3.46 matches per S1 (1.67 from S2, 1.79 from S3), up to 11.
- **S2/S3 are not deduplicated, but each S2/S3 record matches at most one S1.** This held for 7.6M train pairs without exception.
- **Matches never cross countries.** State agreement is 100% in the US. In India it is 98.8%, and every disagreement is Telangana vs Andhra Pradesh.
- **Test has about 2x more tied look-alike distractors per S1 than train.** These are unmatched same-name records in the same state: ~2.4 vs 1.2 per S1, measured with label-free markers. Our validation weights entities accordingly.
- **The noise is synthetic and patterned:**
  - junk prefixes (`>>`, `--`, `##`)
  - leetspeak (`Imp0rts`)
  - random accents
  - doubled spaces
  - domains used as names
  - `f/k/a` trade names
  - word and component reordering
  - uppercase USPS-style addresses in S2 and spelled-out state names in S3
  - literal `NULL`
- **~23% of Indian S2 names (13% of S3) are phonetic transliterations into native scripts**: Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Odia and Gurmukhi.
- **40% of S1 names are shared by another S1 in the same country**, so the address must disambiguate. The hardest negatives share name, state and street but differ in house number.
- **France (15% of test) has no training data.** A controlled transfer test showed a US-only model losing 12 points on India, but an India-only model losing only 1.3 on the US.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + set-level decision (hybrid, graph-aware through competition features).
**Core Innovation:**
- test-density-aware validation
- stage-2 stacking of out-of-fold stage-1 scores in their S1 and record context
- state/region-partitioned multi-path TF-IDF blocking, including native-script-to-Latin phonetic keys

---

## 3. Candidate Generation (Blocking)
- **Partitioning:** by (country, state/region). Telangana is merged into Andhra Pradesh. French regions come from region or département names. A country is never hard-coded.
- **Search paths:** each is a top-k over character-3-gram TF-IDF, pruning n-grams present in more than 2% of records.
  - core name (k=15)
  - phonetic name key (k=10)
  - canonical address (k=15)
  - name+address combined (k=20)
  - concatenated name for domain-style names (k=5)
  - reverse search from each S2/S3 record to its best 3 S1s
- **Records without a usable state:** they are searched against the whole country's S1, by name, phonetic key, concatenated name, address and combined text. S1 records without a state are searched against the whole country's pool, in chunks.
- **Acronym key:** matches initials such as "AF" to "Ace Foundation".
- **Candidate pairs generated:** 8,173,752 in the final candidate_pairs.tsv (4.72 per S1), including the pairs added by the never-compared rescue and the French exact-match step (`python postprocess/fixes/sub_check.py output/` prints them).
- **How we ensured true matches were not lost:** we measured pair recall and the oracle F0.5 ceiling on held-out states. Phase 0 reached 98.6-98.9% recall and a 0.996 ceiling. The reverse and stateless paths each rescue true pairs that no other path finds.

---

## 4. Matching Model

**Features used (about 85):**
- **Name features:**
  - fuzzy ratio, partial, token-sort and token-set scores; Jaro-Winkler; normalised Levenshtein
  - phonetic-key ratios
  - concatenated-name ratio and acronym match
  - IDF-weighted token Jaccard and the maximum IDF of a shared token
  - legal-form equal / conflict / missing
  - first-token equality
  - best `f/k/a` alternative-name score
- **Address features:**
  - fuzzy scores and IDF-weighted token Jaccard
  - house-number set Jaccard, shared / conflict / missing, longest-number equality, minimum digit edit distance
  - state equal / conflict
  - empty / stateless / native-script flags
- **Other:**
  - TF-IDF cosines from blocking and search-path flags
  - context ranks and gaps within the S1's candidates and within the record's competing S1s
  - count of same-name candidates and the address rank among them
  - name-frequency encodings (how common the name is in the state and country)
  - stage 2: out-of-fold stage-1 score, its rank, gap and sum within the S1, and its margin over the record's best competing S1

**Model type:** LightGBM (MIT) for stages 1-3; pretrained cross-encoders `cross-encoder/ms-marco-MiniLM-L6-v2` and `-L12-v2` (Apache-2.0) and `intfloat/multilingual-e5-small` and `microsoft/mdeberta-v3-base` (MIT), all fine-tuned on training-state pairs only, all far below 8B parameters. One e5 cross-encoder is further adapted on test without labels (self-training on test pairs our rules mark as decoys and on confident test pairs). Stage 1 is a 2-fold ensemble with folds split by state. Stage 2 is trained on out-of-fold stage-1 scores. No country feature is used.
**Threshold selection method:**
- Each S2/S3 record is assigned to its highest-scoring S1 only.
- A threshold is then tuned for macro F0.5 on full-universe validation states.
- An expected-F0.5 set-selection rule with an entity-level "has any match" model was evaluated as an alternative.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation (US NY + India AP/TS, S1-half cross-fit) 0.99003 for the final decision (stage 3 with all cross-encoders, rules and the rescue); public leaderboard 0.988757 (final file, final rank 181 of about 10,300 teams). The gap of about 0.0013 comes mainly from look-alike distractors that are denser on test than in the validation states, and from France (no labels).
- **Common false positives (wrong merges):** same-name businesses in the same state with shared street tokens but different house numbers, and chain names. 62% of false positives have name token-set ≥ 90.
- **Common false negatives (missed matches):** records with empty addresses (name-only, 25% of low-scored misses), true pairs with typo'd house numbers (28%), and trade names or native-script names with low string similarity (18%). Precision was 99.1% and recall 94.9%.

---

## 6. Conclusion
Careful normalisation and wide blocking set the ceiling; a LightGBM stack over string features and several small cross-encoders, then rules for house-number decoys, closed most of the rest. What remains is mostly outside the data: 81% of the missed address-less owners share their exact raw name with another business, and France has no labels and far more generic names. Every change was accepted only if it improved both halves of a held-out split, which kept the leaderboard close to the validation forecast.

---

## Appendix

### A. Code Artefacts
See `code/business_entity_resolution/README.md`. Entry point: `run_all.sh`. The main scripts are `convert_to_parquet.py`, `build_normalized.py`, `p1_block.py`, `p1_train.py` and `p1_score.py`.

### B. Additional Results
The experiment log, including the failures: `code/business_entity_resolution/docs/EXPERIMENT_LOG.md`. EDA figures: `code/business_entity_resolution/docs/figs/`.
