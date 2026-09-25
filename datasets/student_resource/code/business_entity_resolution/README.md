# Business Entity Resolution Pipeline

## Overview
This repository contains a high-performance, competition-grade Business Entity Resolution pipeline designed for multi-source business entity matching.
The primary objective is to maximize **Macro F0.5 score** (> 0.980643) by prioritizing high precision, preventing false merges, and correctly identifying singletons and multi-match entities.

---

## Code Architecture

```text
code/business_entity_resolution/
├── src/
│   ├── normalization.py       # Multi-view string & field normalization engine
│   ├── blocking.py            # Multi-pass candidate blocking system
│   ├── features.py            # Rich pairwise feature extractor (RapidFuzz, JW, token, address, cross-field, source)
│   ├── dataset_utils.py       # Entity-level validation split & dataset helpers
│   ├── models.py              # ML matchers (XGBoost, LightGBM, CatBoost) & Decision Engine
│   ├── evaluation.py          # Macro F0.5, Precision, Recall, and Blocking Recall metric evaluators
│   ├── run_experiments.py     # Experiment runner for validation, model comparison, hard-negative mining, error analysis
│   └── inference.py           # Full test inference pipeline generating matching_results.tsv and candidate_pairs.tsv
├── README.md                  # Reproduction & pipeline instructions
└── requirements.txt           # Pinned dependencies
```

---

## Reproduction Instructions

### 1. Environment Setup
Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Experiments & Validation
To train candidate models, compare performance across models, perform hard-negative mining, optimize decision thresholds, and generate error analysis reports:

```bash
python src/run_experiments.py
```

Results will be saved to `experiments/results.csv` and `experiments/error_analysis.md`.

### 3. Generate Final Submission Outputs
To run the full end-to-end inference pipeline on the test dataset and generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`:

```bash
python src/inference.py
```

### 4. Validate Submission Format
Validate the generated TSV files using the official validator:

```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test \
    --check-ids
```

---

## Methodology Key Highlights
1. **Multi-View Normalization**: Generates basic, no-legal-suffix, sorted token, alphanumeric, abbreviation-aware, address digit, and postal code representations.
2. **Multi-Pass High-Recall Blocking**: Combines exact name matches, no-suffix matches, token/prefix inverted index lookups, address component matching, and TF-IDF character n-gram cosine similarity.
3. **Rich Pairwise Features**: Computes RapidFuzz fuzzy ratios, Jaro-Winkler, token Jaccard, address digit overlap, postal code equality, house number equality, and cross-field interaction features.
4. **Hard-Negative Mining**: Iteratively mines high-confidence false positive pairs and retrains the decision model to strictly penalize false merges.
5. **Threshold Optimization**: Optimizes decision thresholds specifically for Macro F0.5.
