# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Archit Mittal
**Team Members:** Archit Mittal, Purvansh Joshi, Aviral Mittal
**Submission Date:** 2 October 2026

---

## 1. Executive Summary
The pipeline has three stages: **canonicalise** records across 9 Indic scripts and obfuscated Latin text, **retrieve** candidates with a conjunctive-key IDF index plus string re-ranking, and **resolve collectively**. In the resolve stage, a second gradient-boosted model sees how candidates compete for each record and how well a record agrees with the entity's other members. Each entity's match list comes from **expected-F0.5-optimal decoding**, which optimises the leaderboard metric directly. Out-of-fold macro F0.5 on the full training set is **0.9853**.

Apart from Source 1's role as the reference, everything is learned from the data, including a self-supervised token-equivalence table. That table is how the method carries over to France, a country absent from training. The pipeline uses no external data, APIs or pretrained models.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA on the training set)
- **Size:** the training set has 2.21M S1, 5.03M S2 and 5.29M S3 records. The test set has 1.73M / 4.89M / 5.08M, and 15% of test S1 records are French.
- **Structure:** every matched S2/S3 record belongs to **exactly one** S1 entity (0 records are assigned to more than one). 26% of S2/S3 records are distractors, 5.6% of S1 entities are singletons, and the median entity has 3–4 matches.
- **Country** is identical within every true pair (100%). We therefore use it to scope blocking keys, as an open-set string that is never enumerated.
- **Names:** about 7% of S2/S3 names in train (about 10% in test) are written in one of **9 Indic scripts**: Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya and Gurmukhi. Other noise:
  - leetspeak (`C0astal`, `5érvices`)
  - domains and hashtags (`coastaltungsten.com`, `#bordeauxmusique`)
  - initials (`SC`)
  - legal-suffix shuffling (`[LLC] Coastal Tungsten`)
  - token repetition
  - unrelated DBA names that share only the address
- **Addresses:** the noise includes:
  - abbreviations (St/Street, R/Rue, Bd/Boulevard)
  - states in native script
  - reordered components
  - injected unit designators (`H.NO`, `Door No`)
  - truncated house numbers (`2454`→`245`)
  - a city swapped for a neighbouring locality
  - missing addresses (3.4% of records)
- **Vocabulary is small and reused.** Only 59% of S1 core names are unique, while name+address is effectively unique. Single-token blocking at a document-frequency cap of 50 reaches only 61% pair coverage, so identity lives in **token combinations**.
- **Siblings are the hardest negatives.** These are different businesses with the same base name plus one extra word ("… Holdings", "… Public Limited") at a nearby house number on the same street (13565 vs 13558, C-316 vs C-303).
- **Train/test prior shift.** Test has about 2.3 S2/S3 decoys per S1 entity, against 1.2 in train. Test decoys are mostly siblings whose S1 record is absent.

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Conjunctive blocking with string re-ranking, then a two-stage GBDT (pairwise, then collective), then decision-theoretic decoding and a high-precision post-processing rule.

**Core Innovations:**
1. **Offset-table Brahmic transliteration.** All 9 Indic Unicode blocks share the ISCII layout, so one 128-entry table transliterates every script. The table handles the inherent vowel, virama, nukta and schwa deletion. A phonetic skeleton then maps `praivet`, `private` and `prywate` to the same key, `prvt`.
2. **Self-supervised token-equivalence mining.** Some high-confidence blocking pairs have token sets that differ by exactly one token on each side. Those residues reveal surface variants (`tx`/`texas`, `mh`/`maharashtra`, `kanstrakshan`/`construction`). Frequent, dominant residues that also look like plausible spelling variants become a substitution table. Mining runs label-free on the test set, so French variants (`r`/`rue`, `av`/`avenue`) are learned without French labels. A sample-based pass runs *before* blocking, so transliteration misses can still become candidates.
3. **Sibling discrimination.**
   - *Number agreement* across each entity's candidate set: house-number corruption is independent per record, but a sibling's records all share *their* number.
   - *Residue-word target encoding*, out-of-fold: for each extra word on one side of a pair, how often training pairs with that residue are true matches. `llc` behaves like noise; `holdings` signals a sibling.
4. **Collective stage-2 features.**
   - *Exclusivity:* rank and margin against rival S1 candidates for the same record.
   - *Entity coherence:* similarity to the entity's most confident other member, its "anchor".

   Together they rescue DBA-renamed and domain-named records.
5. **Prior-shift correction (S1 dropout).** We remove about 19% of training S1 entities, weighted towards name families, so their S2/S3 records become sibling-like decoys. This matches the training decoy density to the test set's, which is measured label-free from the test files.
6. **Expected-F0.5 decoding** for each S1 entity, using Poisson-binomial dynamic programming. The empty list is an explicit option, so singletons are handled optimally.

---

## 3. Candidate Generation (Blocking)
- **Direction:** S2/S3 → S1. Each S2/S3 record has at most one true S1, so a small per-query top-K bounds the candidate set naturally.
- **Keys:** every key is scoped by country and hashed:
  - unordered name-skeleton pairs
  - name-token × address-word
  - house-number × address-word
  - unordered address-word pairs
  - composite address codes (`c303`), alone and × address word
  - single name tokens and single address words
  - an 8-character space-free name prefix (for domain-style names)
- **Scoring:** keys with S1 document frequency above `cap = 400` are dropped. A pair is scored by Σ log(N/df) over its shared keys. Each query keeps its top 10 S1 candidates scoring at least 0.3 × its best score.
- **Two-stage re-ranking:** from the wider top-50 key-score pool, up to 2 more candidates per query are rescued by plain string similarity (token-set ratio ≥ 0.7 on name + address). This step targets ties among generic names that the key score cannot separate. On test it added 9,971,596 pairs, 20% of candidates.
- **Candidate pairs generated (final, test):** 49,604,823 pairs from the pipeline, plus 21,194 from post-processing, across 1,732,544 S1 entities. Only 9 S1 entities have no candidate. The reduction ratio against the full cross product of 1.73M × 9.97M is 1 − 2.9 × 10⁻⁶.
- **Protecting recall:** the keys are redundant by design:
  - Name-only keys cover missing or garbled addresses.
  - Address-only keys cover DBA, Indic and domain names.
  - Number-free address-pair keys cover corrupted house numbers.
  - The string re-ranker covers ties among generic names.
- **Measured on train:** at `cap = 200` (run 97), blocking pair recall was **0.9772**, with an **F0.5 ceiling of 0.9919** for a perfect matcher, at 4.33 pairs per query. Raising the cap from 80 to 200 to 400 was the largest single source of gains (§5). A larger S1 corpus pushes more keys past a fixed df cap, and this hits India harder than the US: India has 279K unique address words, against 94K for the US.

---

## 4. Matching Model

**Stage-1 features (78):**
- **Name:** ratio, token-sort, token-set, partial ratio and Jaro-Winkler on the canonical core name; ratio and token-set on the phonetic skeleton; ratio and partial ratio on the space-free name (for domains); token-set on the full name including legal form; plain and IDF-weighted Jaccard, and intersection size; skeleton Jaccard; token counts; initials match; script, domain and source flags.
- **Address:** token-set, token-sort, partial and plain ratio on canonical address words; plain and IDF-weighted Jaccard; number-set ratio, intersection and Jaccard; first-number equality; longest shared number length (ZIP/PIN); empty-address flags. Plus one joint name+address token-set score.
- **Sibling discrimination:** house-number relation (signature equality, containment, edit distance, log difference); composite-code overlap; extra/missing name words and their IDF mass; legal-form conflict and equality; 10 label-free number-agreement features across the candidate set; and 7 out-of-fold residue-word target encodings.
- **Blocking context:** key score, number of shared keys, rank, margin to the best rival, ratio to the query's best, re-rank similarity and whether the pair was rescued, number of candidates for the query, and how many queries retrieved this S1 and ranked it first.

**Stage-2 collective features (+21, 99 total):**
- the stage-1 probability, with its rank, margin and sum within the query
- its sum and rank within the S1 entity, and the entity's number of candidates and confident members
- the anchor's probability
- query-vs-anchor similarity on name, skeleton, address and numbers
- anchor-vs-S1 name similarity, and second-anchor address similarity
- sibling-aware sums and maxima of probability over records that share the query's number signature or the S1's exact one

**Model type:** LightGBM (MIT licence; binary, 255 leaves, learning rate 0.08, early stopping on a 4% query hold-out). It is **cross-fitted in 2 folds grouped by S1 entity**. Stage-2 inputs use out-of-fold stage-1 probabilities on train and the average of the two fold models on test, so stage 2 sees the same input distribution in both. The final run trained on a 35% query sample on CPU. XGBoost on GPU, with 100% of the pairs, is supported and was used in run 97. Every model is a tree ensemble, far below the 8B-parameter limit.

**Threshold selection / decoding:**
1. **Exclusivity:** each S2/S3 record is kept only for its arg-max S1, and only if p ≥ 0.02.
2. **Expected-F0.5 decoding:** for each S1 entity, take its top-16 sorted probabilities. Choose the prefix size k that maximises E[F0.5] under Poisson-binomial TP/FN distributions, where k = 0 is the probability that the entity is a singleton.
3. **Decoder selection:** the decoder is chosen on out-of-fold predictions over all training S1 entities. Expected-F0.5 (0.98528) beat every fixed threshold from 0.3 to 0.8; the best fixed threshold, 0.7, scored 0.98515. A learned entity-size decoder is always fitted too, and is adopted only if its OOF score is higher. In the final run it was not adopted.

**Post-processing (`postprocess.py`):** this is a deterministic rule. An unmatched S2/S3 record that has **no address** and a normalised name of at least 8 characters joins the S1 entity of its exact-name twin, when every matched twin in the same country belongs to exactly one S1 entity. On labelled train, the rule has **0.991 precision** over 57,345 records (India 0.990, US 0.991). On test it added 21,194 pairs, none of them to an entity predicted empty. The added IDs are also appended to `candidate_pairs.tsv`, because this rule is the last candidate-generation step.

---

## 5. Results & Error Analysis

**Final submission:** run 98 + post-processing. Validator: **PASS**.
- Train OOF macro F0.5: stage 1 **0.98272**, stage 2 **0.98528** (expected-F decoder).
- Test: 5,850,757 predicted pairs. 1,632,979 S1 entities have matches and 99,565 (5.7%) are predicted empty, consistent with the 5.6% singleton rate in train.

**Progression across full-scale Kaggle runs** (train OOF macro F0.5, 2-fold by S1 entity):

| Run | Blocking | Learner | Stage-1 | Stage-2 |
|---|---|---|---|---|
| first full run | cap 80, top-8 | LightGBM | 0.96321 | 0.97135 |
| intermediate run (+ sibling features) | top-10 (cap not logged) | XGBoost (GPU), 74 features | 0.97931 | 0.98160 |
| run 97 | cap 200, top-10, 74 features | XGBoost (GPU, 100% of pairs) | 0.98150 | 0.98362 |
| **run 98 (final)** | **cap 400 + string re-rank, pass-0 mining** | **LightGBM (CPU, 35% of pairs)** | **0.98272** | **0.98528** |

Later runs evaluate under S1 dropout, which adds harder, sibling-like decoys. Their scores are therefore, if anything, conservative relative to the first row.

**Detail from run 97** (complete train log in `code/business_entity_resolution/logs/`; note: run 98's training log was missing, so its overall OOF scores come from the saved state file, while per-country and precision/recall details below are quoted from run 97):
- **Per country:** India 0.9775 (709,768 S1), US 0.9876 (1,075,161 S1). India lags because its address and name vocabulary is richer.
- **Pair precision 0.9970, recall 0.9598.** The model is precision-heavy, as F0.5 rewards. Recall is bounded mainly by blocking: the ceiling is 0.9919.
- **Top stage-2 features by gain:** `p1`, then `p1_margin` (the competition margin between rival S1 candidates), `b_margin`, `p1_qmax`, `p1_srank`. The collective layer adds about +0.002 to +0.008 F0.5 on top of pairwise matching in every run.

**Common false positives:**
- Siblings: same brand, an extra word, a house number a few digits away.
- Generic low-information names ("Super Hospitality Private Limited") that recur at nearby but distinct addresses.

**Common false negatives:**
- Records whose address is missing (3.4% of S2/S3) and whose DBA/trade name is unrelated to the S1 legal name. The anchor features and the twin post-processing rule recover a share of these.
- True pairs never retrieved by blocking. These account for about 0.8 F0.5 points of the ceiling gap at `cap = 200`.

**Evaluated but not used:** a gap-fill ensemble (`ensemble.py`) added run-97 matches for records that run 98 left unmatched (+37,554 pairs). Run 98's expected-F decoder had already declined these records, and run 97 is the weaker model, so the additions carry a precision risk under F0.5. We kept the single-model submission.

---

## 6. Conclusion
The challenge is mostly a retrieval and disambiguation problem, not a string-similarity one. Our biggest gains came from three sources:
- conjunctive, re-ranked blocking that lifts the recall ceiling
- explicit modelling of competition and siblings within each entity's candidate set
- a decoder that optimises per-entity F0.5 instead of thresholding

Learning substitutions and correcting the decoy prior label-free, on the test data itself, let the same pipeline transfer to France without any French labels.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` is self-contained:
- `src/run.py` trains (cross-fit, both stages) and predicts the test set.
- `src/postprocess.py` adds the address-less twins.
- `reproduce.sh` chains both steps and the official validator: `bash reproduce.sh <student_resource>`.

The README documents every flag, the Kaggle workflow and the module map. `requirements.txt` lists bounded dependency versions: polars, rapidfuzz, lightgbm, numpy, psutil, and optionally xgboost (Apache-2.0). `logs/` holds the Kaggle logs for run 97 (train and test) and for run 98 (test stage).

### B. Additional Results
Run 98, full decoder grid (train OOF macro F0.5):

| decoder | stage 1 | stage 2 |
|---|---|---|
| expected-F0.5 | 0.98272 | **0.98528** |
| threshold 0.3 | 0.97812 | 0.98038 |
| threshold 0.4 | 0.98011 | 0.98235 |
| threshold 0.5 | 0.98158 | 0.98381 |
| threshold 0.6 | 0.98258 | 0.98473 |
| threshold 0.7 | 0.98311 | 0.98515 |
| threshold 0.8 | 0.98315 | 0.98496 |

Equivalences mined label-free on the test set (run 98) include 316 address maps (`mh→maharashtra`, `texas→tx`, `ka→karnataka`), 349 core-name maps (`kanstrakshan→construction`, `venchars→ventures`, `inphotek→infotech`) and 240 skeleton maps. The plausibility filter rejected 204 candidate merges that were co-occurrences rather than variants, such as `atlantique/pays` and `odisha/orissa`.
