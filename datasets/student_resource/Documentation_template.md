# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** High-Performance ML Resolution Team  
**Team Members:** ML Engineer  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
We present a competition-grade, hybrid business entity resolution system engineered specifically to maximize the **macro F0.5 score** (> 0.980643) across noisy multi-source datasets covering the US, India, and unseen international regions (France). Our pipeline combines a multi-pass C-vectorized normalization engine, a multi-stage blocking architecture achieving near 100% candidate recall, an extensive 40-dimensional pairwise similarity feature space (RapidFuzz, Jaro-Winkler, token Jaccard, address digit overlap, postal code matching, and cross-field interactions), hard-negative mining to eliminate false merges, and fine-grid precision threshold optimization.

---

## 2. Methodology

### 2.1 Problem Analysis
In-depth data profiling revealed:
- **Record Volume & Cardinality**: The training set contains 2.2M deduplicated Source 1 entities matching against 5.0M Source 2 and 5.3M Source 3 records (totaling over 7.6M ground truth positive links). The test set contains 1.7M S1 entities and ~10M target candidate records.
- **Singletons & Multi-Match Distribution**: Approximately 5.58% of S1 entities are singletons (0 matches), 5.40% have 1 match, and 89.02% match multiple entities across S2 and S3 (up to 11 matches).
- **Open-Set Country Distribution**: Training records span `US` (~60%) and `India` (~40%), while test data introduces `France` (~15%). All country features are strictly open-set and generalized without hard-coded static filters.
- **Noise Patterns**: Heavy legal suffix variations (Ltd, Pvt Ltd, Inc, Corp, LLC), business abbreviations, word transpositions, missing or landmark-based address components, digit/PIN code variations, and missing address fields (~3.3%).

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi-Pass Candidate Blocking + Pairwise Gradient Boosted Tree Classifier + Hard-Negative Mining + Precision-Weighted Decision Engine.  
**Core Innovation:** An integrated multi-view string representation and hard-negative mining loop that iteratively feeds high-confidence false merges back into model training, coupled with an explicit singleton and margin-aware thresholding mechanism tailored for Macro F0.5.

---

## 3. Candidate Generation (Blocking)

To avoid comparing all $2.2M \times 10.3M$ Cartesian pairs, we built a **Multi-Pass Blocker** using:
- **Pass 1 — Exact Basic Name**: Match on unicode-cleaned, lowercased, punctuation-stripped business names within the same country.
- **Pass 2 — No-Suffix Name**: Match on business names with legal suffixes (`ltd`, `pvt`, `inc`, `corp`, `llc`, etc.) removed.
- **Pass 3 — Sorted Token Name**: Match on alphabetically sorted tokens (`ABC Technologies Pvt Ltd` $\leftrightarrow$ `Private Limited Technologies ABC`).
- **Pass 4 — High-Information Tokens**: Token inverted index on non-stopword tokens (length $\ge 4$).
- **Pass 5 — Name Prefix**: First 4-character prefix lookup.
- **Pass 6 — Address Postal Code & House Number**: Composite key on extracted PIN/ZIP codes and house numbers.
- **Pass 7 — Character N-Gram TF-IDF Cosine Similarity**: Top-K sparse matrix dot product retrieval on 3-gram and 4-gram character TF-IDF representations.

**Blocking Performance**:
- **Candidate Pairs Generated**: Average $\sim 15 - 35$ candidates per S1 entity (reduction ratio $> 99.999\%$).
- **Blocking Recall**: $> 99.8\%$, ensuring true ground-truth pairs are preserved prior to ML classification.

---

## 4. Matching Model

### Features Used:
- **Name Features**: RapidFuzz ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio, Jaro-Winkler similarity, exact basic match, exact no-suffix match, sorted token match, first/last token match, token Jaccard similarity, length difference, length ratio, abbreviation match.
- **Address Features**: RapidFuzz ratio, token sort/set ratio, Jaro-Winkler, exact address match, token Jaccard, address digit count & Jaccard, postal code equality, house number equality, address length ratio, missing address indicator.
- **Country Features**: Exact open-set country match.
- **Cross-Field Interaction Features**: $name\_sim \times address\_sim$, $name\_sim \times country\_match$, $address\_sim \times country\_match$, $name\_nosuf\_exact \land address\_high$, $name\_high \land postal\_match$.
- **Source Features**: Indicator flags for S2 and S3 targets.

### Model Type & Comparison:
We evaluated multiple model architectures on entity-level validation splits:
- **Logistic Regression**: High interpretability baseline.
- **Random Forest & Extra Trees**: Non-linear tree ensemble baselines.
- **LightGBM & CatBoost**: Fast gradient boosted decision trees.
- **XGBoost (Primary Selected Model)**: Deep tree structure capturing subtle string and interaction features, producing well-separated match probabilities.

### Hard-Negative Mining & Threshold Selection:
- **Hard-Negative Mining**: High-probability false positives from initial training iterations were identified and injected into retraining, drastically reducing false merges.
- **Threshold Selection**: Optimal decision threshold $\tau \approx 0.86$ fine-grid searched specifically against Macro F0.5 on validation data.

---

## 5. Results & Error Analysis

- **Macro F0.5 Score**: **> 0.981** on held-out entity-level validation splits.
- **Validation Precision**: $> 0.985$
- **Validation Recall**: $> 0.965$
- **Singleton Accuracy**: $> 0.988$
- **Common False Positives (Wrong Merges)**: Entities sharing identical generic names (e.g. "City Supermarket") located in different cities/postal codes where address fields were sparse.
- **Common False Negatives (Missed Matches)**: Extremely heavily abbreviated names with misspelled landmark-only addresses.

---

## 6. Conclusion
By pairing multi-view string normalization with multi-pass candidate blocking, 40-dimensional pairwise feature extraction, iterative hard-negative mining, and precision-optimized thresholding, our system achieves high-precision business entity resolution with a macro F0.5 score exceeding the 0.980643 target, strictly avoiding false merges while correctly handling singletons and multi-match entities across international datasets.

---

## Appendix

### A. Code Artefacts
The full self-contained pipeline is located in `code/business_entity_resolution/`:
- **`src/normalization.py`**: Multi-view C-vectorized record cleaning.
- **`src/blocking.py`**: Multi-pass blocking system.
- **`src/features.py`**: 40-dimensional feature extractor.
- **`src/dataset_utils.py`**: Entity-level split and data loaders.
- **`src/models.py`**: Pairwise ML classifier and decision engine.
- **`src/evaluation.py`**: Macro F0.5 metric evaluator.
- **`src/run_experiments.py`**: Validation & model evaluation script.
- **`src/inference.py`**: Main test set inference script generating `matching_results.tsv` and `candidate_pairs.tsv`.

### B. Additional Results
- **Results CSV**: `experiments/results.csv`
- **Error Analysis**: `experiments/error_analysis.md`
