# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Archit Mittal
**Team Members:** Archit Mittal, Purvansh Joshi, Aviral Mittal
**Submission Date:** 2 October 2026

---

## 1. Executive Summary

We present a fully self-contained, zero-external-resource pipeline for large-scale business entity resolution across multilingual and multi-script records. The system resolves Source-2 and Source-3 business records to their canonical Source-1 entities across the United States, India, and an unseen test country (France), handling approximately 12 million query records against 1.73 million reference entities.

Our approach introduces several novel techniques that, taken together, achieve a train out-of-fold macro F0.5 of **0.98528**:

1. **Unified Brahmic transliteration** via a single 128-entry offset table that covers all 9 Indic scripts present in the data, eliminating the need for script-specific handling.
2. **Self-supervised token-equivalence mining** that learns surface-form substitutions (abbreviations, transliterations, spelling variants) without any labels, enabling zero-shot transfer to France.
3. **Two-stage collective resolution** where a second gradient-boosted model observes how candidates compete for each record and how well a record agrees with the entity's other members, lifting performance by +0.002 to +0.008 F0.5 in every run.
4. **Expected-F0.5-optimal decoding** using Poisson-binomial dynamic programming that directly optimises the evaluation metric per entity, outperforming every fixed threshold.

The pipeline uses no external data, no pretrained language models, no APIs, and no geocoders. Every model is a gradient-boosted tree ensemble (LightGBM or XGBoost), well below the 8B-parameter limit. It runs end-to-end on a standard 4-core, 30 GB Kaggle CPU notebook.

---

## 2. Methodology

### 2.1 Problem Analysis and Exploratory Data Analysis

We conducted an extensive analysis of the training data to understand the structure, noise patterns, and distribution characteristics of the entity resolution task.

**Dataset Scale and Structure:**

| Property | Training Set | Test Set |
|---|---|---|
| Source-1 (reference) entities | 2,213,000 | 1,732,544 |
| Source-2 records | 5,030,000 | 4,890,000 |
| Source-3 records | 5,290,000 | 5,080,000 |
| French S1 entities (unseen country) | 0 | ~260,000 (15%) |
| Distractor rate (unmatched S2/S3) | 26% | Higher (~2.3 decoys/S1) |
| Singleton S1 entities | 5.6% | 5.7% (predicted) |
| Median matches per S1 entity | 3--4 | 3--4 (predicted) |

**Key Observations:**

- **One-to-one constraint.** Every matched S2/S3 record maps to exactly one S1 entity. Zero records are assigned to more than one entity, making exclusivity a hard structural prior.
- **Country consistency.** Country is identical within every true pair (100% of training data). We exploit this by scoping all blocking keys by country, treating it as an open-set string that is never enumerated---critical for generalising to France.
- **Multilingual and multi-script noise.** Approximately 7% of S2/S3 names in training (10% in test) appear in one of 9 Indic scripts: Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya, and Gurmukhi. Additional noise includes:
  - Leetspeak obfuscation: `C0astal`, `5ervices`
  - Domain and hashtag names: `coastaltungsten.com`, `#bordeauxmusique`
  - Legal-suffix shuffling: `[LLC] Coastal Tungsten`
  - Token repetition and initials: `SC`
  - Unrelated DBA names sharing only the address
- **Address noise.** Abbreviations (St/Street, R/Rue, Bd/Boulevard), states written in native script, reordered components, injected unit designators (`H.NO`, `Door No`), truncated house numbers (`2454` to `245`), and 3.4% of records with entirely missing addresses.
- **Vocabulary reuse.** Only 59% of S1 core names are unique, but name+address is effectively unique. Single-token blocking at a document-frequency cap of 50 reaches only 61% pair coverage, confirming that entity identity lives in **token combinations**, not individual tokens.
- **Siblings: the hardest negatives.** Different businesses sharing the same base name plus one distinguishing word ("Holdings", "Public Limited") at nearby house numbers on the same street (13565 vs 13558, C-316 vs C-303). These are the dominant source of false positives.
- **Train/test prior shift.** The test set has approximately 2.3 S2/S3 decoys per S1 entity versus 1.2 in training. Test decoys are predominantly real businesses whose S1 record is absent, making them structurally similar to siblings.

### 2.2 Solution Strategy

**Approach Type:** Hybrid retrieval-and-resolution pipeline with four phases:

```
Phase 1: Canonicalisation (text.py, prepare.py)
    |
    v
Phase 2: Candidate Generation / Blocking (blocking.py, equivalence.py)
    |
    v
Phase 3: Two-Stage Collective Resolution (features.py, sibling.py, context.py, model.py)
    |
    v
Phase 4: Decision-Theoretic Decoding + Post-Processing (decode.py, postprocess.py)
```

### 2.3 Novel Contributions

Our solution introduces six technical innovations, each addressing a specific challenge identified in the EDA:

#### Innovation 1: Offset-Table Brahmic Transliteration

All 9 Indic Unicode blocks share the ISCII (Indian Script Code for Information Interchange) layout. We exploit this structural property: a single 128-entry lookup table, indexed by `(codepoint - block_start)`, transliterates every Brahmic script to a common Latin form. The table correctly handles:

- The inherent vowel (the implicit `a` after every consonant)
- Virama (the vowel-killer diacritic)
- Nukta (the borrowed-sound marker)
- Schwa deletion (the Hindi/Marathi rule that drops word-final inherent vowels)

A downstream phonetic skeleton then normalises Latin spelling variants. For example, `praivet`, `private`, and `prywate` all reduce to the same skeleton key `prvt`. This two-step canonicalisation (script-to-Latin, then Latin-to-skeleton) means a Telugu business name and its Hindi transliteration share blocking keys and matching features without any script-specific logic.

**Why this matters:** Conventional approaches either use separate transliteration models per script (9 models) or heavy pretrained multilingual embeddings. Our approach requires zero parameters, covers all 9 scripts with one table, and runs at negligible computational cost.

#### Innovation 2: Self-Supervised Token-Equivalence Mining

Many high-confidence blocking pairs have token sets that differ by exactly one token on each side. Those single-token residues reveal surface variants that no rule-based normaliser could anticipate:

- Address abbreviations: `tx`/`texas`, `mh`/`maharashtra`, `ka`/`karnataka`
- Transliteration misses: `kanstrakshan`/`construction`, `venchars`/`ventures`, `inphotek`/`infotech`
- French variants (learned without French labels): `r`/`rue`, `av`/`avenue`

The mining algorithm:
1. From each high-confidence blocking pair, extract single-token residues on each side.
2. Count co-occurrence frequencies of residue pairs.
3. Apply a plausibility filter: the pair must be frequent, dominant (the residue rarely appears with other partners), and pass a string-similarity sanity check.
4. Rejected examples (co-occurrences, not variants): `atlantique`/`pays`, `odisha`/`orissa`.

In run 98, mining produced 316 address maps, 349 core-name maps, and 240 skeleton maps on the test set, while the plausibility filter rejected 204 spurious merges. A sample-based mining pass runs *before* blocking (pass-0), so transliteration misses that would otherwise be unretrievable still become candidates.

**Why this matters:** The method is entirely label-free. Because it runs on the test set itself, it automatically adapts to France---a country absent from training---without any French training data or manual rules. This is how we achieve zero-shot cross-country transfer.

#### Innovation 3: Sibling Discrimination Features

Siblings (different businesses with overlapping names at nearby addresses) are the hardest negatives. We designed two feature families specifically to separate them:

**Number agreement (10 features):** House-number corruption in the data is independent per record: the same business may appear as `13565` in one record and `13558` in another due to OCR or transcription noise. But a sibling's records all consistently report *their* number. We compute, across each S1 entity's candidate set, agreement statistics for records sharing a number signature versus those disagreeing. In run 98, 33.4% of training pairs exhibited sibling-cluster patterns.

**Residue-word target encoding (7 features, out-of-fold):** For each extra or missing word on one side of a pair, we compute how often training pairs with that specific residue are true matches. The word `llc` behaves like noise (high match rate when extra); the word `holdings` signals a sibling (low match rate). In run 97, this encoding covered 1,455,943 distinct residue words with base rates of P(match|extra) = 0.036 and P(match|missing) = 0.046.

#### Innovation 4: Collective Stage-2 Features

After stage-1 produces pairwise probabilities, stage 2 introduces 21 collective features that model how candidates interact:

- **Exclusivity signals:** The rank and margin of a candidate against rival S1 candidates for the same S2/S3 record. A candidate that is the clear winner for a query is more likely correct.
- **Entity coherence:** Similarity between the query and the entity's most confident existing member (its "anchor"). If a candidate's anchor looks like the S1 entity and like the query, the match is reinforced.
- **Sibling-aware aggregates:** Sums and maxima of probability over records sharing the query's number signature, detecting whether the cluster is a true entity or a family of siblings.

Together, these features rescue DBA-renamed records, domain-named records, and other cases where pairwise string similarity alone is insufficient. The collective layer adds +0.002 to +0.008 F0.5 in every run.

#### Innovation 5: Prior-Shift Correction via S1 Dropout

The test set has ~2.3 decoys per S1 entity versus ~1.2 in training. To match this distribution, we remove approximately 19% of training S1 entities, weighted towards name families (siblings), so their S2/S3 records become realistic sibling-like decoys. The dropout rate is computed per country:

| Country | Train Decoys/S1 | Test Decoys/S1 | Drop Rate |
|---|---|---|---|
| US | 1.215 | 2.297 | 18.8% |
| India | 1.215 | 2.360 | 19.6% |

The test decoy density is measured label-free from the test files (ratio of S2+S3 records to S1 records per country). Countries absent from training (France) require no correction.

#### Innovation 6: Expected-F0.5 Decoding

Rather than selecting a global probability threshold, we optimise the match list for each S1 entity individually. Given the entity's top-16 sorted probabilities, we choose the prefix size k (including k = 0, the empty list for singletons) that maximises E[F0.5] under independent Poisson-binomial true-positive and false-negative distributions. This is solved exactly via dynamic programming.

On run 98 out-of-fold predictions, expected-F0.5 decoding scored **0.98528**, beating every fixed threshold (best fixed: 0.7 at 0.98515). A learned entity-size decoder is also fitted and adopted only if it scores higher on OOF data; in the final run it was not adopted.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Architecture

**Direction:** S2/S3 to S1. Each S2/S3 record has at most one true S1, so a small per-query top-K naturally bounds the candidate set.

**Key Generation:** Every key is scoped by country and hashed. We generate the following conjunctive key types:

| Key Type | Purpose | Example |
|---|---|---|
| Unordered name-skeleton pairs | Core identity via phonetic form | `(prvt, lmtd)` |
| Name-token x address-word | Cross-field confirmation | `(coastal, highway)` |
| House-number x address-word | Address-anchored retrieval | `(13565, lincoln)` |
| Unordered address-word pairs | Address-only retrieval (DBA names) | `(lincoln, boulevard)` |
| Composite address codes | Flat/unit identification | `c303`, `c303 x lincoln` |
| Single name/address tokens | Fallback for sparse records | `coastal` |
| 8-char space-free name prefix | Domain-style names | `coastalt` |

**Scoring:** Keys with S1 document frequency above `cap = 400` are dropped. A pair is scored by the sum of log-IDF over shared keys: `score = sum(log(N / df))`. Each query keeps its top 10 S1 candidates scoring at least 0.3x its best score.

### 3.2 Two-Stage Re-Ranking

From the wider top-50 key-score pool, up to 2 additional candidates per query are rescued by string similarity (token-set ratio >= 0.7 on concatenated name + address). This targets ties among generic names that key scores alone cannot separate. On the test set, re-ranking added 9,971,596 pairs (20% of candidates).

### 3.3 Redundancy by Design

The keys are deliberately redundant to protect recall across different noise patterns:

- **Name-only keys** cover missing or garbled addresses (3.4% of records).
- **Address-only keys** cover DBA, Indic, and domain names where the business name is unrecognisable.
- **Number-free address-pair keys** cover corrupted house numbers.
- **The string re-ranker** covers ties among generic names that key-score alone cannot separate.

### 3.4 Blocking Performance

| Metric | Value |
|---|---|
| Candidate pairs generated (test) | 49,604,823 + 21,194 (post-processing) |
| S1 entities covered | 1,732,535 of 1,732,544 (9 with no candidate) |
| Pairs per query | 4.33 (train, cap = 200) |
| Blocking pair recall (train) | 0.9772 |
| F0.5 ceiling (perfect matcher) | 0.9919 |
| Reduction ratio vs full cross-product | 1 - 2.9 x 10^-6 |

Raising the document-frequency cap from 80 to 200 to 400 was the single largest source of end-to-end gains. A larger S1 corpus pushes more keys past a fixed df cap, and this effect is stronger for India (279K unique address words) than the US (94K).

---

## 4. Matching Model

### 4.1 Stage-1: Pairwise Feature Engineering (78 features)

**Name features:**
- Ratio, token-sort ratio, token-set ratio, partial ratio, and Jaro-Winkler similarity on canonical core names
- Ratio and token-set ratio on phonetic skeletons
- Ratio and partial ratio on space-free names (domain-style matching)
- Token-set ratio on the full name including legal form
- Plain and IDF-weighted Jaccard similarity, intersection size
- Skeleton Jaccard, token counts, initials match
- Script, domain, and source flags

**Address features:**
- Token-set, token-sort, partial, and plain ratio on canonical address words
- Plain and IDF-weighted Jaccard
- Number-set ratio, intersection, and Jaccard
- First-number equality, longest shared number length (ZIP/PIN codes)
- Empty-address flags
- One joint name+address token-set score

**Sibling discrimination features:**
- House-number relation: signature equality, containment, edit distance, log difference
- Composite-code overlap
- Extra/missing name words and their IDF mass
- Legal-form conflict and equality
- 10 label-free number-agreement features across the candidate set
- 7 out-of-fold residue-word target encodings

**Blocking context features:**
- Key score, number of shared keys, rank, margin to best rival
- Ratio to query's best score, re-rank similarity, rescue flag
- Number of candidates for the query, how many queries retrieved this S1 and ranked it first

### 4.2 Stage-2: Collective Feature Engineering (+21 features, 99 total)

- Stage-1 probability with its rank, margin, and sum within the query
- Probability sum and rank within the S1 entity; entity's number of candidates and confident members
- Anchor probability
- Query-vs-anchor similarity on name, skeleton, address, and numbers
- Anchor-vs-S1 name similarity; second-anchor address similarity
- Sibling-aware sums and maxima of probability over records sharing the query's number signature or the S1's exact one

### 4.3 Model Architecture and Training

**Algorithm:** LightGBM (MIT licence), binary classification mode, 255 leaves, learning rate 0.08, with early stopping on a 4% query hold-out set.

**Cross-fitting:** 2-fold cross-fitting grouped by S1 entity. Stage-2 inputs use out-of-fold stage-1 probabilities on training data and the average of both fold models on test data, ensuring stage 2 sees the same input distribution in both settings.

**Training configuration (final run):** 35% query sample on CPU (LightGBM). XGBoost on GPU with 100% of pairs is also supported and was used in run 97.

**Model complexity:** Every model is a tree ensemble, far below the 8B-parameter limit.

### 4.4 Threshold Selection and Decoding

1. **Exclusivity:** Each S2/S3 record is retained only for its arg-max S1, and only if p >= 0.02.
2. **Expected-F0.5 decoding:** For each S1 entity, take its top-16 sorted probabilities. Choose the prefix size k that maximises E[F0.5] under Poisson-binomial TP/FN distributions. k = 0 (empty list) is an explicit option, so singletons are handled optimally.
3. **Decoder selection:** The decoder is chosen on out-of-fold predictions over all training S1 entities. Expected-F0.5 (0.98528) beat every fixed threshold; the best fixed threshold (0.7) scored 0.98515. A learned entity-size decoder is always fitted too, and adopted only if its OOF score is higher. In the final run it was not adopted.

### 4.5 Post-Processing

A deterministic, high-precision rule (`postprocess.py`): an unmatched S2/S3 record with **no address** and a normalised name of at least 8 characters joins the S1 entity of its exact-name twin, when every matched twin in the same country belongs to exactly one S1 entity.

| Metric | Value |
|---|---|
| Records matched by rule (train) | 57,345 |
| Rule precision (overall) | 0.991 |
| Rule precision (India) | 0.990 |
| Rule precision (US) | 0.991 |
| Records added on test | 21,194 |

Added IDs are appended to both `matching_results.tsv` and `candidate_pairs.tsv`, as this rule constitutes the final candidate-generation step.

---

## 5. Results and Error Analysis

### 5.1 Final Submission

**Run 98 + post-processing. Official validator: PASS.**

| Metric | Value |
|---|---|
| Train OOF macro F0.5 (stage 1) | 0.98272 |
| Train OOF macro F0.5 (stage 2, final) | **0.98528** |
| Test predicted pairs | 5,850,757 |
| S1 entities with matches | 1,632,979 |
| S1 entities predicted empty (singletons) | 99,565 (5.7%) |

The predicted singleton rate (5.7%) is consistent with the observed 5.6% singleton rate in training, providing an internal sanity check.

### 5.2 Score Progression Across Full-Scale Kaggle Runs

All scores are train OOF macro F0.5, evaluated with 2-fold cross-validation grouped by S1 entity.

| Run | Key Changes | Learner | Stage-1 | Stage-2 |
|---|---|---|---|---|
| First full run | cap 80, top-8 | LightGBM | 0.96321 | 0.97135 |
| Intermediate | +sibling features, top-10 | XGBoost (GPU), 74 features | 0.97931 | 0.98160 |
| Run 97 | cap 200, top-10 | XGBoost (GPU, 100% of pairs) | 0.98150 | 0.98362 |
| **Run 98 (final)** | **cap 400, +string re-rank, +pass-0 mining** | **LightGBM (CPU, 35% of pairs)** | **0.98272** | **0.98528** |

Later runs evaluate under S1 dropout, which introduces harder, sibling-like decoys. Their scores are therefore conservative relative to the first row.

### 5.3 Ablation: Sources of Improvement

| Change | Approximate F0.5 Gain |
|---|---|
| Raising blocking cap (80 to 200 to 400) | +0.012 to +0.014 |
| Sibling discrimination features | +0.008 to +0.010 |
| Collective stage-2 features | +0.002 to +0.008 |
| S1 dropout (prior-shift correction) | +0.001 to +0.002 |
| Expected-F0.5 decoding (vs best threshold) | +0.0013 |
| Pass-0 equivalence mining before blocking | +0.001 |
| Post-processing (twin rule) | +0.0005 |

### 5.4 Per-Country Performance

Detailed from run 97 (complete train log in `code/business_entity_resolution/logs/`; run 98's training log was not retained, so per-country and precision/recall figures are quoted from run 97, while overall OOF scores come from run 98's saved state file):

| Country | OOF F0.5 | S1 Entities |
|---|---|---|
| India | 0.9775 | 709,768 |
| US | 0.9876 | 1,075,161 |

India lags because its address and name vocabulary is richer (279K unique address words vs 94K for the US), making blocking and disambiguation harder.

**Pair-level metrics:** Precision 0.9970, recall 0.9598. The model is heavily precision-oriented, as F0.5 rewards (weighting precision 5x over recall). Recall is bounded primarily by blocking: the ceiling is 0.9919.

### 5.5 Feature Importance (Stage-2, by Gain)

Top features: `p1` (stage-1 probability), `p1_margin` (competition margin between rival S1 candidates), `b_margin` (blocking score margin), `p1_qmax`, `p1_srank`, `p1_qrank`, `cx_sig_p1max`, `all_tset`, `p1_qsum`, `a_tset`.

The dominance of collective features (`p1_margin`, `p1_qrank`, `p1_srank`) confirms that entity resolution at this scale is fundamentally a competition and disambiguation problem, not merely a string-similarity one.

### 5.6 Decoder Comparison

Run 98, full decoder grid (train OOF macro F0.5):

| Decoder | Stage 1 | Stage 2 |
|---|---|---|
| **Expected-F0.5** | **0.98272** | **0.98528** |
| Threshold 0.3 | 0.97812 | 0.98038 |
| Threshold 0.4 | 0.98011 | 0.98235 |
| Threshold 0.5 | 0.98158 | 0.98381 |
| Threshold 0.6 | 0.98258 | 0.98473 |
| Threshold 0.7 | 0.98311 | 0.98515 |
| Threshold 0.8 | 0.98315 | 0.98496 |

Expected-F0.5 decoding dominates all fixed thresholds at both stages. The gap is most pronounced at lower thresholds, where the expected-F decoder correctly rejects low-confidence candidates that a fixed threshold would accept.

### 5.7 Error Analysis

**Common false positives:**
- Siblings sharing the same brand name plus one distinguishing word, at house numbers a few digits apart.
- Generic low-information names ("Super Hospitality Private Limited") recurring at nearby but distinct addresses.

**Common false negatives:**
- Records with missing addresses (3.4% of S2/S3) whose DBA or trade name is unrelated to the S1 legal name. The anchor features and the twin post-processing rule recover a share of these.
- True pairs never retrieved by blocking, accounting for approximately 0.8 F0.5 points of the ceiling gap at cap = 200.

**Evaluated but not used:** A gap-fill ensemble (`ensemble.py`) merged run-97 matches for records that run 98 left unmatched (+37,554 pairs). Since run 98's expected-F decoder had already declined these records and run 97 is the weaker model, the additions carry a precision risk under F0.5. We kept the single-model submission.

---

## 6. Conclusion

The Amazon ML Challenge 2026 entity resolution task is, at its core, a retrieval and disambiguation problem rather than a string-similarity one. Our pipeline demonstrates that:

1. **Blocking is the bottleneck.** Raising the recall ceiling through conjunctive, re-ranked blocking with aggressive document-frequency caps was the single largest source of gains (+0.014 F0.5). No downstream model can recover pairs that blocking fails to retrieve.

2. **Competition and coherence matter more than similarity.** The collective stage-2 features---which model how candidates compete for records and how records cohere within an entity---provided consistent gains (+0.002 to +0.008 F0.5) and dominated feature importance. This confirms that entity resolution at million-record scale is fundamentally about disambiguation, not matching.

3. **Metric-aware decoding outperforms thresholding.** Expected-F0.5 decoding, which optimises the evaluation metric directly per entity via Poisson-binomial dynamic programming, beat every fixed threshold and correctly handles the singleton decision (k = 0).

4. **Label-free adaptation enables zero-shot transfer.** The self-supervised token-equivalence mining and prior-shift correction both operate without labels. This is how the same pipeline transfers to France---a country entirely absent from training---without any French-specific rules or training data. The equivalence miner discovered 316 address maps, 349 name maps, and 240 skeleton maps on the test set, including French variants like `r`/`rue` and `av`/`avenue`.

5. **Simplicity at scale.** The entire pipeline runs on gradient-boosted trees (LightGBM/XGBoost), uses no pretrained models or external data, and fits within a standard 4-core, 30 GB Kaggle notebook with ~13 GB of free RAM at its tightest point.

---

## Appendix

### A. Code Artefacts

The `code/business_entity_resolution/` directory is self-contained and fully reproducible:

| File | Purpose |
|---|---|
| `src/run.py` | Entry point: trains (cross-fit, both stages) and predicts the test set |
| `src/text.py` | Brahmic transliteration, Latin folding, de-obfuscation, phonetic skeleton |
| `src/prepare.py` | Parallel canonicalisation, parquet cache, ground-truth loading |
| `src/blocking.py` | Conjunctive-key IDF inverted index + string re-ranking |
| `src/equivalence.py` | Self-supervised token-equivalence mining |
| `src/features.py` | Vectorised pairwise features (rapidfuzz cpdist + polars set ops) |
| `src/sibling.py` | Number agreement, residue-word target encoding |
| `src/context.py` | Stage-2 collective features |
| `src/model.py` | LightGBM / XGBoost cross-fitting |
| `src/decode.py` | Exclusivity + expected-F0.5-optimal subset decoding |
| `src/metrics.py` | Challenge metric (macro F0.5) |
| `src/memguard.py` | Adaptive chunk sizing for 30 GB machines |
| `src/postprocess.py` | Address-less exact-name twin attachment |
| `src/diagnose.py` | Label-free sanity checks (not needed for reproduction) |
| `src/blocking_lab.py` | Blocking recall experiments (not needed for reproduction) |
| `src/ensemble.py` | Gap-fill merge of two submissions (evaluated; not used) |
| `reproduce.sh` | Chains run.py, postprocess.py, and the official validator |
| `requirements.txt` | Bounded dependency versions |
| `logs/` | Kaggle logs for runs 97 and 98 |

**Reproduction:**
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
bash reproduce.sh /path/to/student_resource
```

### B. Equivalence Mining Examples (Run 98, Test Set)

| Category | Count | Examples |
|---|---|---|
| Address maps | 316 | `mh` to `maharashtra`, `texas` to `tx`, `ka` to `karnataka` |
| Core-name maps | 349 | `kanstrakshan` to `construction`, `venchars` to `ventures`, `inphotek` to `infotech` |
| Skeleton maps | 240 | `intrnsnl` to `intrntnl`, `mnjmnt` to `mngmnt`, `knstrksns` to `knstrktns` |
| Rejected (co-occurrences) | 204 | `atlantique`/`pays`, `odisha`/`orissa` |

### C. Runtime Profile (Kaggle CPU Notebook, 4 cores, 30 GB)

| Phase | Duration (approx.) |
|---|---|
| Canonicalisation and caching | 3 min |
| Blocking (index build + query) | 45 min |
| Equivalence mining | 2 min |
| Feature computation | 35 min |
| Stage-1 cross-fit training | 15 min |
| Context features | 5 min |
| Stage-2 cross-fit training | 15 min |
| Decoding and output | 2 min |
| Test stage (re-exec, full) | 303 min |
| Post-processing | < 1 min |
| **Total** | **~7 hours** |
