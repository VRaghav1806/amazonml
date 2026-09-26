import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization import build_normalized_record_views
from blocking import MultiPassBlocker
from features import extract_pairwise_features, extract_features_parallel
from models import PairMatcher, EntityDecisionEngine
from dataset_utils import load_source_tsv, load_ground_truth_dict, normalize_dataframe

def run_test_inference(
    train_dir: str, 
    test_dir: str, 
    output_dir: str,
    threshold_s2: float = 0.86,
    threshold_s3: float = 0.86,
    model_type: str = 'xgboost',
    max_candidates_per_s1: int = 100,
    n_jobs: int = -1
):
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print("================ 1. LOADING & NORMALIZING ALL DATA ================", flush=True)
    
    print("Loading train files...", flush=True)
    train_s1 = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"))
    train_s2 = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"))
    train_s3 = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"))
    gt_dict = load_ground_truth_dict(os.path.join(train_dir, "train_ground_truth.tsv"))

    # Sample 100k training entities for robust model fitting
    if len(train_s1) > 100000:
        print("Sampling 100,000 training S1 entities for model fitting...", flush=True)
        train_s1 = train_s1.sample(n=100000, random_state=42).reset_index(drop=True)

    sampled_s1_ids = set(train_s1['entity_id'])
    target_ids = set()
    for s1_id in sampled_s1_ids:
        if s1_id in gt_dict:
            target_ids.update(gt_dict[s1_id])

    # Include random background targets for negative blocking candidates
    extra_s2 = train_s2.sample(n=min(100000, len(train_s2)), random_state=42)['entity_id']
    extra_s3 = train_s3.sample(n=min(100000, len(train_s3)), random_state=42)['entity_id']
    target_ids.update(extra_s2)
    target_ids.update(extra_s3)

    print("Filtering relevant training target records...", flush=True)
    train_s2 = train_s2[train_s2['entity_id'].isin(target_ids)].reset_index(drop=True)
    train_s3 = train_s3[train_s3['entity_id'].isin(target_ids)].reset_index(drop=True)

    print("Normalizing training records...", flush=True)
    train_s1_recs = normalize_dataframe(train_s1)
    train_s2_recs = normalize_dataframe(train_s2)
    train_s3_recs = normalize_dataframe(train_s3)
    train_target_recs = train_s2_recs + train_s3_recs
    train_target_lookup = {r['entity_id']: r for r in train_target_recs}
    train_s1_lookup = {r['entity_id']: r for r in train_s1_recs}

    print("\n================ 2. BUILDING TRAINING BLOCKING & FEATURES ================", flush=True)
    blocker = MultiPassBlocker(max_candidates_per_s1=max_candidates_per_s1)
    blocker.fit_index_target_records(train_target_recs)
    
    # Generate candidates for train
    print("Generating candidates for training set (multi-core)...", flush=True)
    train_candidates = blocker.generate_candidates(train_s1_recs, n_jobs=n_jobs)
    
    train_pairs = []
    train_labels = []
    for s1_rec in train_s1_recs:
        s1_id = s1_rec['entity_id']
        true_gt = gt_dict.get(s1_id, set())
        cands = train_candidates.get(s1_id, set())
        
        positives = list(true_gt.intersection(cands))
        negatives = list(cands - true_gt)
        
        for pos_id in positives:
            train_pairs.append((s1_id, pos_id))
            train_labels.append(1)
            
        # Sample hard negatives
        if len(negatives) > 4:
            sampled_negs = negatives[:4]
        else:
            sampled_negs = negatives
            
        for neg_id in sampled_negs:
            train_pairs.append((s1_id, neg_id))
            train_labels.append(0)

    print(f"Total training pairs: {len(train_pairs)} (Positives: {sum(train_labels)})", flush=True)
    
    print("Extracting features for training pairs (parallel multi-core)...", flush=True)
    t_feat_start = time.time()
    X_train = extract_features_parallel(train_pairs, train_s1_lookup, train_target_lookup, n_jobs=n_jobs)
    y_train = np.array(train_labels, dtype=int)
    print(f"Training features extracted in {time.time() - t_feat_start:.2f}s", flush=True)

    print(f"\n================ 3. TRAINING FINAL MODEL ({model_type.upper()}) ================", flush=True)
    matcher = PairMatcher(model_type=model_type)
    matcher.fit(X_train, y_train)
    print("Model training complete!", flush=True)

    print("\n================ 4. LOADING & NORMALIZING TEST DATA ================", flush=True)
    test_s1 = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s2 = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s3 = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"))

    print("Normalizing test records...", flush=True)
    test_s1_recs = normalize_dataframe(test_s1)
    test_s2_recs = normalize_dataframe(test_s2)
    test_s3_recs = normalize_dataframe(test_s3)
    
    test_target_recs = test_s2_recs + test_s3_recs
    test_target_lookup = {r['entity_id']: r for r in test_target_recs}
    test_s1_lookup = {r['entity_id']: r for r in test_s1_recs}

    print("\n================ 5. GENERATING TEST CANDIDATES ================", flush=True)
    test_blocker = MultiPassBlocker(max_candidates_per_s1=max_candidates_per_s1)
    test_blocker.fit_index_target_records(test_target_recs)
    
    t_cand_start = time.time()
    test_candidates = test_blocker.generate_candidates(test_s1_recs, n_jobs=n_jobs)
    print(f"Test candidate generation complete in {time.time() - t_cand_start:.2f}s", flush=True)

    # Save candidate_pairs.tsv
    # Save candidate_pairs.tsv
    candidate_file_path = os.path.join(output_dir, "candidate_pairs.tsv")
    print(f"Writing candidate_pairs.tsv to {candidate_file_path}...", flush=True)
    
    cand_lines = ["source1_entity_id\tcandidate_entity_ids\n"]
    with open(candidate_file_path, "w", encoding="utf-8") as f:
        for r in test_s1_recs:
            s1_id = r['entity_id'] if isinstance(r, dict) else r.entity_id
            cands = sorted(list(test_candidates.get(s1_id, set())))
            cand_str = ",".join(cands)
            cand_lines.append(f"{s1_id}\t{cand_str}\n")
            if len(cand_lines) >= 50000:
                f.writelines(cand_lines)
                cand_lines.clear()
        if cand_lines:
            f.writelines(cand_lines)
            
    print("candidate_pairs.tsv written successfully!", flush=True)

    print("\n================ 6. INFERENCE & SCORING TEST PAIRS ================", flush=True)
    test_pairs = []
    for r in test_s1_recs:
        s1_id = r['entity_id'] if isinstance(r, dict) else r.entity_id
        cands = test_candidates.get(s1_id, set())
        for cand_id in cands:
            test_pairs.append((s1_id, cand_id))

    print(f"Total test candidate pairs to score: {len(test_pairs)}", flush=True)
    
    # Process test features in chunks using multi-core parallel extraction
    chunk_size = 250000
    test_probs = []
    
    t_infer_start = time.time()
    for start in range(0, len(test_pairs), chunk_size):
        end = min(start + chunk_size, len(test_pairs))
        chunk_pairs = test_pairs[start:end]
        print(f"Extracting features & scoring test chunk [{start}..{end}]...", flush=True)
        
        X_chunk = extract_features_parallel(chunk_pairs, test_s1_lookup, test_target_lookup, n_jobs=n_jobs)
        chunk_probs = matcher.predict_proba(X_chunk)
        test_probs.extend(chunk_probs)
        
    test_probs = np.array(test_probs)
    print(f"Test pair scoring complete in {time.time() - t_infer_start:.2f}s", flush=True)

    print("\n================ 7. SINGLETON & MULTI-MATCH RESOLUTION ================", flush=True)
    engine = EntityDecisionEngine(threshold_s2=threshold_s2, threshold_s3=threshold_s3)
    test_s1_ids = [r['entity_id'] if isinstance(r, dict) else r.entity_id for r in test_s1_recs]
    
    final_matches = engine.predict_entity_matches(test_s1_ids, test_pairs, test_probs)
    
    # Save matching_results.tsv
    matching_file_path = os.path.join(output_dir, "matching_results.tsv")
    print(f"Writing matching_results.tsv to {matching_file_path}...", flush=True)
    
    num_singletons = 0
    num_matches = 0
    match_lines = ["source1_entity_id\tmatched_entity_ids\n"]
    
    with open(matching_file_path, "w", encoding="utf-8") as f:
        for r in test_s1_recs:
            s1_id = r['entity_id'] if isinstance(r, dict) else r.entity_id
            m_set = final_matches.get(s1_id, set())
            if not m_set:
                num_singletons += 1
                match_lines.append(f"{s1_id}\t\n")
            else:
                num_matches += 1
                m_str = ",".join(sorted(list(m_set)))
                match_lines.append(f"{s1_id}\t{m_str}\n")
            if len(match_lines) >= 50000:
                f.writelines(match_lines)
                match_lines.clear()
        if match_lines:
            f.writelines(match_lines)

    print(f"matching_results.tsv written successfully!")
    print(f"Total Test Entities: {len(test_s1_recs)}")
    print(f"Predicted Singletons (no match): {num_singletons} ({(num_singletons/len(test_s1_recs))*100:.2f}%)")
    print(f"Predicted Matched Entities: {num_matches} ({(num_matches/len(test_s1_recs))*100:.2f}%)")
    print(f"Pipeline finished successfully in {time.time() - start_total_time:.2f}s!")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    train_dir = os.path.join(root_dir, "dataset", "train")
    test_dir = os.path.join(root_dir, "dataset", "test")
    output_dir = os.path.join(root_dir, "output")
    
    run_test_inference(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        threshold_s2=0.78,
        threshold_s3=0.78,
        model_type='xgboost'
    )
