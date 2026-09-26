import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_dataframe_fast
from blocking import MultiPassBlocker
from features import extract_features_parallel
from models import PairMatcher, EntityDecisionEngine
from evaluation import evaluate_macro_metrics, compute_blocking_recall
from dataset_utils import load_source_tsv, load_ground_truth_dict

def validate_accuracy(
    data_dir: str,
    model_path: str,
    num_val_entities: int = 15000,
    seed: int = 999
):
    print("================ 1. LOADING VALIDATION SPLIT ================", flush=True)
    t0 = time.time()
    
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    
    print("Loading Ground Truth and Source-1...", flush=True)
    gt_dict_full = load_ground_truth_dict(gt_path)
    df_s1 = load_source_tsv(s1_path)
    
    # Pick held-out validation entities (disjoint seed 999 to guarantee unseen entities)
    print(f"Sampling {num_val_entities} held-out validation S1 entities (seed={seed})...", flush=True)
    val_s1_df = df_s1.sample(n=num_val_entities, random_state=seed).reset_index(drop=True)
    val_s1_ids = set(val_s1_df['entity_id'])
    val_gt = {eid: gt_dict_full.get(eid, set()) for eid in val_s1_ids}
    
    # Identify target entities needed for validation
    needed_targets = set()
    for mset in val_gt.values():
        needed_targets.update(mset)
        
    print(f"Validation entities: {len(val_s1_ids):,} (True matches: {sum(len(v) for v in val_gt.values()):,}, Singletons: {sum(1 for v in val_gt.values() if not v):,})", flush=True)

    print("Loading Target sources S2 and S3...", flush=True)
    df_s2 = load_source_tsv(s2_path)
    df_s3 = load_source_tsv(s3_path)
    
    # Add negative targets for realistic blocking competition
    extra_s2 = set(df_s2.sample(n=min(50000, len(df_s2)), random_state=seed)['entity_id'])
    extra_s3 = set(df_s3.sample(n=min(50000, len(df_s3)), random_state=seed)['entity_id'])
    needed_targets.update(extra_s2)
    needed_targets.update(extra_s3)
    
    df_s2_val = df_s2[df_s2['entity_id'].isin(needed_targets)].reset_index(drop=True)
    df_s3_val = df_s3[df_s3['entity_id'].isin(needed_targets)].reset_index(drop=True)
    del df_s2, df_s3, df_s1
    
    print("Normalizing records...", flush=True)
    val_s1_recs = normalize_dataframe_fast(val_s1_df)
    s2_recs = normalize_dataframe_fast(df_s2_val)
    s3_recs = normalize_dataframe_fast(df_s3_val)
    target_recs = s2_recs + s3_recs
    
    val_s1_lookup = {r.entity_id: r for r in val_s1_recs}
    target_lookup = {r.entity_id: r for r in target_recs}
    
    print(f"Data prepared in {time.time() - t0:.2f}s", flush=True)
    
    print("\n================ 2. MEASURING CANDIDATE BLOCKING RECALL ================", flush=True)
    t_block = time.time()
    blocker = MultiPassBlocker(max_candidates_per_s1=35)
    blocker.fit_index_target_records(target_recs)
    val_candidates = blocker.generate_candidates(val_s1_recs)
    
    blocking_recall, total_found, total_gt_matches = compute_blocking_recall(val_gt, val_candidates)
    print(f"Blocking Time:    {time.time() - t_block:.2f}s", flush=True)
    print(f"Blocking Recall:  {blocking_recall * 100:.2f}% ({total_found:,} / {total_gt_matches:,} true matches captured)", flush=True)
    
    print("\n================ 3. FEATURE EXTRACTION & SCORING ================", flush=True)
    t_score = time.time()
    print(f"Loading trained matcher from: {model_path}...", flush=True)
    matcher = joblib.load(model_path)
    
    # Form validation candidate pairs
    val_pairs = []
    val_s1_cand_slices = []
    for r in val_s1_recs:
        s1_id = r.entity_id
        cands = list(val_candidates.get(s1_id, set()))[:12]
        s_idx = len(val_pairs)
        for cid in cands:
            val_pairs.append((s1_id, cid))
        e_idx = len(val_pairs)
        val_s1_cand_slices.append((s1_id, cands, s_idx, e_idx))
        
    print(f"Scoring {len(val_pairs):,} validation pairs with XGBoost...", flush=True)
    X_val = extract_features_parallel(val_pairs, val_s1_lookup, target_lookup)
    val_probs = matcher.predict_proba(X_val)
    print(f"Pairs scored in {time.time() - t_score:.2f}s ({len(val_pairs)/(time.time()-t_score):.0f} pairs/s)", flush=True)
    
    print("\n================ 4. ACCURACY & MACRO F0.5 METRICS ================", flush=True)
    
    thresholds = [0.65, 0.70, 0.75, 0.78, 0.80, 0.85]
    print(f"{'Threshold':>10} | {'Macro F0.5':>12} | {'Precision':>10} | {'Recall':>10} | {'Singleton Acc':>14}")
    print("-" * 65)
    
    best_f05 = 0.0
    best_thresh = 0.78
    best_metrics = None
    
    for thresh in thresholds:
        pred_dict = {}
        for s1_id, cands, s_idx, e_idx in val_s1_cand_slices:
            if e_idx == s_idx:
                pred_dict[s1_id] = set()
                continue
            probs = val_probs[s_idx:e_idx]
            selected = set()
            for cid, prob in zip(cands, probs):
                if prob >= thresh:
                    selected.add(cid)
            pred_dict[s1_id] = selected
            
        metrics = evaluate_macro_metrics(val_gt, pred_dict)
        f05 = metrics['macro_f05']
        p = metrics['macro_precision']
        r = metrics['macro_recall']
        s_acc = metrics['singleton_accuracy']
        
        print(f"{thresh:>10.2f} | {f05:>11.4f}% | {p*100:>9.2f}% | {r*100:>9.2f}% | {s_acc*100:>13.2f}%")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = metrics
            
    print("-" * 65)
    print(f"\n>>> BEST MACRO F0.5 SCORE: {best_f05:.4f} at threshold {best_thresh:.2f} <<<")
    print(f"    - Macro Precision:    {best_metrics['macro_precision']*100:.2f}%")
    print(f"    - Macro Recall:       {best_metrics['macro_recall']*100:.2f}%")
    print(f"    - Singleton Accuracy: {best_metrics['singleton_accuracy']*100:.2f}%")
    print(f"    - Total GT Matches:   {best_metrics['num_gt_matches']:,}")
    print(f"    - Predicted Matches:  {best_metrics['num_pred_matches']:,}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    data_dir = os.path.join(root_dir, "dataset", "train")
    exp_dir = os.path.join(root_dir, "code", "business_entity_resolution", "experiments")
    model_path = os.path.join(exp_dir, "xgb_pair_matcher.joblib")
    
    validate_accuracy(data_dir=data_dir, model_path=model_path, num_val_entities=15000, seed=999)
