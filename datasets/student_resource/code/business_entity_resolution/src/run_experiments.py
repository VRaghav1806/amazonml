import os
import sys
import time
import random
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization import build_normalized_record_views
from blocking import MultiPassBlocker
from features import extract_pairwise_features, FEATURE_NAMES
from evaluation import evaluate_macro_metrics, compute_blocking_recall
from models import PairMatcher, EntityDecisionEngine
from dataset_utils import (
    load_source_tsv, load_ground_truth_dict, normalize_dataframe,
    create_entity_validation_split, extract_pairs_and_labels
)

def run_experiment_pipeline():
    print("================ 1. LOADING & NORMALIZING DATA ================", flush=True)
    start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates_paths = [
        os.path.abspath(os.path.join(script_dir, "..", "..", "..", "dataset", "train")),
        os.path.abspath(os.path.join(script_dir, "..", "..", "dataset", "train")),
        os.path.abspath("dataset/train")
    ]
    data_dir = None
    for p in candidates_paths:
        if os.path.isdir(p):
            data_dir = p
            break
            
    if not data_dir:
        raise FileNotFoundError(f"Could not find dataset/train folder in candidate paths: {candidates_paths}")
        
    print(f"Using data directory: {data_dir}", flush=True)
    
    print("Loading train_source1.tsv...", flush=True)
    df_s1 = load_source_tsv(os.path.join(data_dir, "train_source1.tsv"))
    print(f"Loaded {len(df_s1)} S1 records.", flush=True)
    
    print("Loading train_source2.tsv...", flush=True)
    df_s2 = load_source_tsv(os.path.join(data_dir, "train_source2.tsv"))
    print(f"Loaded {len(df_s2)} S2 records.", flush=True)
    
    print("Loading train_source3.tsv...", flush=True)
    df_s3 = load_source_tsv(os.path.join(data_dir, "train_source3.tsv"))
    print(f"Loaded {len(df_s3)} S3 records.", flush=True)
    
    print("Loading ground truth...", flush=True)
    gt_dict = load_ground_truth_dict(os.path.join(data_dir, "train_ground_truth.tsv"))
    print(f"Loaded {len(gt_dict)} GT entities.", flush=True)

    # Sample 100,000 S1 records for super fast experiment iteration
    df_s1_sample = df_s1.sample(n=100000, random_state=42) if len(df_s1) > 100000 else df_s1
    sample_s1_ids = set(df_s1_sample['entity_id'])
    
    # Get all GT matches for these sampled S1 entities
    gt_dict_sample = {k: v for k, v in gt_dict.items() if k in sample_s1_ids}
    gt_target_ids = set()
    for mset in gt_dict_sample.values():
        gt_target_ids.update(mset)
        
    df_s2_sample = df_s2[df_s2['entity_id'].isin(gt_target_ids) | (df_s2.index % 15 == 0)]
    df_s3_sample = df_s3[df_s3['entity_id'].isin(gt_target_ids) | (df_s3.index % 15 == 0)]

    print(f"Sampled {len(df_s1_sample)} S1 records, {len(df_s2_sample)} S2 records, {len(df_s3_sample)} S3 records.", flush=True)

    print("\n================ 2. CREATING ENTITY-LEVEL VALIDATION SPLIT ================", flush=True)
    s1_recs = normalize_dataframe(df_s1_sample)
    s2_recs = normalize_dataframe(df_s2_sample)
    s3_recs = normalize_dataframe(df_s3_sample)
    
    target_recs = s2_recs + s3_recs
    
    val_num_entities = min(20000, max(1, int(len(s1_recs) * 0.2)))
    train_s1, val_s1, train_gt, val_gt = create_entity_validation_split(
        s1_recs, s2_recs, s3_recs, gt_dict_sample, val_num_entities=val_num_entities, seed=42
    )
    
    train_s1_sample = train_s1
    train_gt_sample = train_gt
    
    print(f"Train S1 entities: {len(train_s1_sample)}")
    print(f"Validation S1 entities: {len(val_s1)}")

    # 3. Candidate Blocking
    print("\n================ 3. MULTI-PASS BLOCKING & RECALL MEASUREMENT ================", flush=True)
    blocker = MultiPassBlocker(max_candidates_per_s1=50)
    print("Fitting inverted index blocker on S2+S3 records...", flush=True)
    blocker.fit_index_target_records(target_recs)
    
    print("Generating candidates for Validation S1 set...", flush=True)
    val_candidates = blocker.generate_candidates(val_s1)
    
    blocking_recall, total_found, total_gt_matches = compute_blocking_recall(val_gt, val_candidates)
    avg_cands = np.mean([len(c) for c in val_candidates.values()])
    print(f"Validation Blocking Recall: {blocking_recall*100:.3f}% ({total_found}/{total_gt_matches} true matches found)")
    print(f"Average candidates per S1 entity: {avg_cands:.2f}")

    print("Generating candidates for Training S1 set...", flush=True)
    train_candidates = blocker.generate_candidates(train_s1_sample)

    # 4. Extract Pairs & Feature Engineering
    print("\n================ 4. PAIR EXTRACTION & FEATURE ENGINEERING ================", flush=True)
    train_pairs, train_y, s1_lookup, target_lookup = extract_pairs_and_labels(
        train_s1_sample, target_recs, train_candidates, train_gt_sample, max_negatives_per_s1=5
    )
    val_pairs, val_y, val_s1_lookup, _ = extract_pairs_and_labels(
        val_s1, target_recs, val_candidates, val_gt, max_negatives_per_s1=10
    )
    
    print(f"Train candidate pairs: {len(train_pairs)} (Positives: {sum(train_y)}, Negatives: {len(train_y)-sum(train_y)})")
    print(f"Val candidate pairs: {len(val_pairs)} (Positives: {sum(val_y)}, Negatives: {len(val_y)-sum(val_y)})")

    print("Extracting features for train pairs...", flush=True)
    X_train = np.array([
        extract_pairwise_features(s1_lookup[s1_id], target_lookup[cand_id])
        for s1_id, cand_id in train_pairs
    ], dtype=np.float32)
    
    print("Extracting features for val pairs...", flush=True)
    X_val = np.array([
        extract_pairwise_features(val_s1_lookup[s1_id], target_lookup[cand_id])
        for s1_id, cand_id in val_pairs
    ], dtype=np.float32)

    # 5. Model Training & Comparison
    print("\n================ 5. MODEL COMPARISON EXPERIMENTS ================", flush=True)
    models_to_test = [
        ("Logistic Regression", PairMatcher(model_type='logistic')),
        ("Random Forest", PairMatcher(model_type='random_forest', params={'n_estimators': 100, 'max_depth': 12})),
        ("LightGBM", PairMatcher(model_type='lightgbm', params={'n_estimators': 200, 'max_depth': 8})),
        ("XGBoost", PairMatcher(model_type='xgboost', params={'n_estimators': 250, 'max_depth': 8})),
        ("CatBoost", PairMatcher(model_type='catboost', params={'iterations': 250, 'depth': 7}))
    ]

    exp_results = []
    best_model_name = None
    best_f05 = -1.0
    best_matcher = None
    best_threshold = 0.85

    val_s1_ids = list(dict.fromkeys(p[0] for p in val_pairs))

    for model_name, matcher in models_to_test:
        print(f"\n--- Training {model_name} ---", flush=True)
        m_start = time.time()
        matcher.fit(X_train, train_y)
        train_dur = time.time() - m_start
        
        val_probs = matcher.predict_proba(X_val)
        
        # Grid search threshold for Macro F0.5
        best_th = 0.50
        best_th_f05 = -1.0
        best_th_metrics = None
        
        for th in np.arange(0.50, 0.98, 0.02):
            engine = EntityDecisionEngine(threshold_s2=th, threshold_s3=th)
            preds = engine.predict_entity_matches(val_s1_ids, val_pairs, val_probs)
            # Add singletons not in val_s1_ids
            for r in val_s1:
                if r['entity_id'] not in preds:
                    preds[r['entity_id']] = set()
                    
            metrics = evaluate_macro_metrics(val_gt, preds)
            if metrics['macro_f05'] > best_th_f05:
                best_th_f05 = metrics['macro_f05']
                best_th = th
                best_th_metrics = metrics
                
        print(f"{model_name} Best Threshold: {best_th:.2f} | F0.5: {best_th_metrics['macro_f05']:.6f} | Precision: {best_th_metrics['macro_precision']:.6f} | Recall: {best_th_metrics['macro_recall']:.6f} | Singleton Acc: {best_th_metrics['singleton_accuracy']:.6f}")
        
        exp_results.append({
            'experiment_id': f"exp_{model_name.lower().replace(' ', '_')}",
            'model': model_name,
            'threshold': best_th,
            'precision': best_th_metrics['macro_precision'],
            'recall': best_th_metrics['macro_recall'],
            'f0.5': best_th_metrics['macro_f05'],
            'blocking_recall': blocking_recall,
            'candidate_count': len(val_pairs),
            'singleton_accuracy': best_th_metrics['singleton_accuracy']
        })
        
        if best_th_metrics['macro_f05'] > best_f05:
            best_f05 = best_th_metrics['macro_f05']
            best_model_name = model_name
            best_matcher = matcher
            best_threshold = best_th

    # 6. Hard-Negative Mining Iteration
    print("\n================ 6. HARD-NEGATIVE MINING ITERATION ================", flush=True)
    print(f"Mining hard negatives using best model ({best_model_name})...", flush=True)
    train_probs = best_matcher.predict_proba(X_train)
    
    # Identify false positives with high confidence
    false_pos_idx = np.where((train_y == 0) & (train_probs >= 0.70))[0]
    print(f"Identified {len(false_pos_idx)} high-confidence hard negative pairs (P >= 0.70).")
    
    # Duplicate hard negative instances to heavily penalize false merges
    X_train_hn = np.vstack([X_train, X_train[false_pos_idx]]).astype(np.float32)
    y_train_hn = np.hstack([train_y, train_y[false_pos_idx]])
    
    print(f"Retraining {best_model_name} with hard negative mining...", flush=True)
    best_matcher.fit(X_train_hn, y_train_hn)
    
    val_probs_hn = best_matcher.predict_proba(X_val)
    
    # Re-tune threshold
    best_th_hn = 0.50
    best_f05_hn = -1.0
    best_metrics_hn = None
    
    for th in np.arange(0.50, 0.98, 0.01):
        engine = EntityDecisionEngine(threshold_s2=th, threshold_s3=th)
        preds = engine.predict_entity_matches(val_s1_ids, val_pairs, val_probs_hn)
        for r in val_s1:
            if r['entity_id'] not in preds:
                preds[r['entity_id']] = set()
        metrics = evaluate_macro_metrics(val_gt, preds)
        if metrics['macro_f05'] > best_f05_hn:
            best_f05_hn = metrics['macro_f05']
            best_th_hn = th
            best_metrics_hn = metrics
            
    print(f"After Hard-Negative Mining: Best Threshold: {best_th_hn:.2f} | F0.5: {best_metrics_hn['macro_f05']:.6f} | Precision: {best_metrics_hn['macro_precision']:.6f} | Recall: {best_metrics_hn['macro_recall']:.6f} | Singleton Acc: {best_metrics_hn['singleton_accuracy']:.6f}")
    
    exp_results.append({
        'experiment_id': f"exp_{best_model_name.lower().replace(' ', '_')}_hn_mining",
        'model': f"{best_model_name} + Hard Negative Mining",
        'threshold': best_th_hn,
        'precision': best_metrics_hn['macro_precision'],
        'recall': best_metrics_hn['macro_recall'],
        'f0.5': best_metrics_hn['macro_f05'],
        'blocking_recall': blocking_recall,
        'candidate_count': len(val_pairs),
        'singleton_accuracy': best_metrics_hn['singleton_accuracy']
    })

    # Save results table
    exp_dir_candidates = [
        os.path.abspath(os.path.join(script_dir, "..", "..", "experiments")),
        os.path.abspath("experiments")
    ]
    exp_dir = exp_dir_candidates[0]
    os.makedirs(exp_dir, exist_ok=True)
    
    df_exp = pd.DataFrame(exp_results)
    results_csv_path = os.path.join(exp_dir, "results.csv")
    df_exp.to_csv(results_csv_path, index=False)
    print(f"\nSaved {results_csv_path}:")
    print(df_exp)

    # 7. Error Analysis
    print("\n================ 7. ERROR ANALYSIS ================", flush=True)
    engine = EntityDecisionEngine(threshold_s2=best_th_hn, threshold_s3=best_th_hn)
    final_val_preds = engine.predict_entity_matches(val_s1_ids, val_pairs, val_probs_hn)
    for r in val_s1:
        if r['entity_id'] not in final_val_preds:
            final_val_preds[r['entity_id']] = set()

    fp_examples = []
    fn_examples = []

    for s1_id, gt_set in val_gt.items():
        p_set = final_val_preds.get(s1_id, set())
        fps = p_set - gt_set
        fns = gt_set - p_set
        
        s1_info = val_s1_lookup.get(s1_id, {})
        
        if fps and len(fp_examples) < 5:
            for fp_id in fps:
                fp_info = target_lookup.get(fp_id, {})
                fp_examples.append({
                    's1_id': s1_id,
                    's1_name': s1_info.get('original_name', ''),
                    's1_addr': s1_info.get('original_address', ''),
                    'cand_id': fp_id,
                    'cand_name': fp_info.get('original_name', ''),
                    'cand_addr': fp_info.get('original_address', ''),
                    'country': s1_info.get('country', '')
                })
                
        if fns and len(fn_examples) < 5:
            for fn_id in fns:
                fn_info = target_lookup.get(fn_id, {})
                fn_examples.append({
                    's1_id': s1_id,
                    's1_name': s1_info.get('original_name', ''),
                    's1_addr': s1_info.get('original_address', ''),
                    'cand_id': fn_id,
                    'cand_name': fn_info.get('original_name', ''),
                    'cand_addr': fn_info.get('original_address', ''),
                    'country': s1_info.get('country', '')
                })

    # Write error analysis markdown
    error_md = f"""# Detailed Error Analysis Report

## Summary Metrics
- **Best Model**: {best_model_name} + Hard Negative Mining
- **Optimal Decision Threshold**: {best_th_hn:.3f}
- **Validation Macro F0.5**: {best_metrics_hn['macro_f05']:.6f}
- **Validation Precision**: {best_metrics_hn['macro_precision']:.6f}
- **Validation Recall**: {best_metrics_hn['macro_recall']:.6f}
- **Singleton Accuracy**: {best_metrics_hn['singleton_accuracy']:.6f}

---

## Top False Positives (False Merges)
False merges represent different real-world businesses assigned to the same S1 entity.

"""
    for idx, ex in enumerate(fp_examples, 1):
        error_md += f"""### False Positive Example {idx}
- **S1 Entity ({ex['s1_id']})**: Name: `{ex['s1_name']}` | Address: `{ex['s1_addr']}`
- **Predicted Candidate ({ex['cand_id']})**: Name: `{ex['cand_name']}` | Address: `{ex['cand_addr']}`
- **Country**: `{ex['country']}`
"""

    error_md += "\n---\n\n## Top False Negatives (Missed Matches)\n"
    for idx, ex in enumerate(fn_examples, 1):
        error_md += f"""### False Negative Example {idx}
- **S1 Entity ({ex['s1_id']})**: Name: `{ex['s1_name']}` | Address: `{ex['s1_addr']}`
- **Missed Ground Truth Match ({ex['cand_id']})**: Name: `{ex['cand_name']}` | Address: `{ex['cand_addr']}`
- **Country**: `{ex['country']}`
"""

    error_analysis_path = os.path.join(exp_dir, "error_analysis.md")
    with open(error_analysis_path, "w", encoding="utf-8") as f:
        f.write(error_md)
        
    print(f"Saved {error_analysis_path}.")

if __name__ == "__main__":
    run_experiment_pipeline()
