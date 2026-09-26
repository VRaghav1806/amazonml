import os
import sys
import time
import gc
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_dataframe_fast
from blocking import MultiPassBlocker
from features import extract_features_parallel, extract_pairwise_features
from models import PairMatcher, EntityDecisionEngine
from dataset_utils import load_source_tsv, load_ground_truth_dict

def train_or_load_model(train_dir: str, model_save_path: str, model_type: str = 'xgboost', n_jobs: int = -1):
    if os.path.exists(model_save_path):
        print(f"Loading pre-trained model from {model_save_path}...", flush=True)
        return joblib.load(model_save_path)
    
    print("================ 1. TRAINING ML MATCHER ================", flush=True)
    t0 = time.time()
    
    print("Loading train files...", flush=True)
    train_s1 = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"))
    train_s2 = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"))
    train_s3 = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"))
    gt_dict = load_ground_truth_dict(os.path.join(train_dir, "train_ground_truth.tsv"))

    # Sample 50,000 training S1 entities for rapid, high-accuracy model fitting
    if len(train_s1) > 50000:
        print("Sampling 50,000 training S1 entities...", flush=True)
        train_s1 = train_s1.sample(n=50000, random_state=42).reset_index(drop=True)

    sampled_s1_ids = set(train_s1['entity_id'])
    target_ids = set()
    for s1_id in sampled_s1_ids:
        if s1_id in gt_dict:
            target_ids.update(gt_dict[s1_id])

    # Add random background targets
    extra_s2 = train_s2.sample(n=min(50000, len(train_s2)), random_state=42)['entity_id']
    extra_s3 = train_s3.sample(n=min(50000, len(train_s3)), random_state=42)['entity_id']
    target_ids.update(extra_s2)
    target_ids.update(extra_s3)

    train_s2 = train_s2[train_s2['entity_id'].isin(target_ids)].reset_index(drop=True)
    train_s3 = train_s3[train_s3['entity_id'].isin(target_ids)].reset_index(drop=True)

    print("Normalizing training samples...", flush=True)
    train_s1_recs = normalize_dataframe_fast(train_s1)
    train_s2_recs = normalize_dataframe_fast(train_s2)
    train_s3_recs = normalize_dataframe_fast(train_s3)
    del train_s1, train_s2, train_s3
    gc.collect()

    train_target_recs = train_s2_recs + train_s3_recs
    train_target_lookup = {r.entity_id: r for r in train_target_recs}
    train_s1_lookup = {r.entity_id: r for r in train_s1_recs}

    print("Building candidate training pairs...", flush=True)
    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.fit_index_target_records(train_target_recs)
    train_candidates = blocker.generate_candidates(train_s1_recs, n_jobs=n_jobs)

    train_pairs = []
    train_labels = []
    for s1_rec in train_s1_recs:
        s1_id = s1_rec.entity_id
        true_gt = gt_dict.get(s1_id, set())
        cands = train_candidates.get(s1_id, set())
        
        positives = list(true_gt.intersection(cands))
        negatives = list(cands - true_gt)
        
        for pos_id in positives:
            train_pairs.append((s1_id, pos_id))
            train_labels.append(1)
            
        sampled_negs = negatives[:4] if len(negatives) > 4 else negatives
        for neg_id in sampled_negs:
            train_pairs.append((s1_id, neg_id))
            train_labels.append(0)

    print(f"Total training pairs: {len(train_pairs)} (Positives: {sum(train_labels)})", flush=True)
    print("Extracting training features...", flush=True)
    X_train = extract_features_parallel(train_pairs, train_s1_lookup, train_target_lookup, n_jobs=n_jobs)
    y_train = np.array(train_labels, dtype=int)

    print("Fitting XGBoost matcher...", flush=True)
    matcher = PairMatcher(model_type=model_type)
    matcher.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(matcher, model_save_path)
    print(f"Model saved to {model_save_path} in {time.time() - t0:.2f}s", flush=True)

    del train_s1_recs, train_s2_recs, train_s3_recs, train_target_recs
    del train_target_lookup, train_s1_lookup, train_pairs, train_labels, X_train, y_train
    del blocker, train_candidates, gt_dict
    gc.collect()

    return matcher

def run_fast_stream_scoring(
    test_dir: str,
    output_dir: str,
    candidate_tsv_path: str,
    matcher: PairMatcher,
    threshold_s2: float = 0.78,
    threshold_s3: float = 0.78,
    target_candidate_budget: int = 6,
    batch_entities: int = 15000,
    n_jobs: int = -1
):
    print("\n================ 2. LOADING TEST LOOKUPS ================", flush=True)
    t_load_start = time.time()
    
    print("Loading & normalizing test_source1.tsv...", flush=True)
    test_s1 = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s1_recs = normalize_dataframe_fast(test_s1)
    test_s1_lookup = {r.entity_id: r for r in test_s1_recs}
    del test_s1, test_s1_recs
    gc.collect()
    print(f"Loaded {len(test_s1_lookup):,} S1 test entities in {time.time() - t_load_start:.1f}s", flush=True)

    t_targ_start = time.time()
    print("Loading & normalizing test_source2.tsv...", flush=True)
    test_s2 = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s2_recs = normalize_dataframe_fast(test_s2)
    del test_s2
    gc.collect()

    print("Loading & normalizing test_source3.tsv...", flush=True)
    test_s3 = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"))
    test_s3_recs = normalize_dataframe_fast(test_s3)
    del test_s3
    gc.collect()

    print("Building target lookup index...", flush=True)
    test_target_lookup = {}
    for r in test_s2_recs:
        test_target_lookup[r.entity_id] = r
    for r in test_s3_recs:
        test_target_lookup[r.entity_id] = r
    del test_s2_recs, test_s3_recs
    gc.collect()
    print(f"Loaded {len(test_target_lookup):,} Target test entities in {time.time() - t_targ_start:.1f}s", flush=True)

    print("\n================ 3. STREAMING & SCORING ALL CANDIDATES ================", flush=True)
    matching_tsv_path = os.path.join(output_dir, "matching_results.tsv")
    cand_tsv_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    print(f"Reading from:             {candidate_tsv_path}", flush=True)
    print(f"Writing matches to:       {matching_tsv_path}", flush=True)
    print(f"Writing compact cands to: {cand_tsv_out_path}", flush=True)
    print(f"Decision Thresholds:      S2={threshold_s2}, S3={threshold_s3}", flush=True)
    print(f"Compact Candidate Budget: {target_candidate_budget} per entity", flush=True)

    total_entities_processed = 0
    total_singletons = 0
    total_matches = 0
    total_scored_pairs = 0
    total_compact_candidates = 0
    t_stream_start = time.time()

    with open(candidate_tsv_path, 'r', encoding='utf-8') as f_in, \
         open(matching_tsv_path, 'w', encoding='utf-8') as f_match_out, \
         open(cand_tsv_out_path, 'w', encoding='utf-8') as f_cand_out:
        
        # Headers
        header = f_in.readline()
        f_match_out.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        batch_s1_ids = []
        batch_cands = []
        
        def process_entity_batch(s1_ids, cands_list):
            nonlocal total_singletons, total_matches, total_scored_pairs, total_compact_candidates
            
            pairs_to_score = []
            entity_cand_slices = []
            
            for s1_id, cands in zip(s1_ids, cands_list):
                s1_rec = test_s1_lookup.get(s1_id)
                s1_cnt = s1_rec.country if s1_rec else ''
                
                # Pre-filter candidate list: keep valid target candidates, skip strict country mismatch
                filtered_cands = []
                for cid in cands:
                    t_rec = test_target_lookup.get(cid)
                    if t_rec is not None:
                        if s1_cnt and t_rec.country and s1_cnt != t_rec.country:
                            continue
                        filtered_cands.append(cid)
                        
                start_idx = len(pairs_to_score)
                for cid in filtered_cands:
                    pairs_to_score.append((s1_id, cid))
                end_idx = len(pairs_to_score)
                entity_cand_slices.append((s1_id, filtered_cands, start_idx, end_idx))
                
            total_scored_pairs += len(pairs_to_score)
            
            # Predict probabilities in parallel
            if pairs_to_score:
                X_batch = extract_features_parallel(pairs_to_score, test_s1_lookup, test_target_lookup, batch_size=25000, n_jobs=n_jobs)
                probs = matcher.predict_proba(X_batch)
            else:
                probs = np.array([])
                
            match_lines = []
            cand_lines = []
            
            for s1_id, cands, s_idx, e_idx in entity_cand_slices:
                ent_probs = probs[s_idx:e_idx] if e_idx > s_idx else np.array([])
                
                # Pair candidates with their predicted probabilities
                cand_prob_pairs = list(zip(cands, ent_probs))
                cand_prob_pairs.sort(key=lambda x: x[1], reverse=True)
                
                matched = []
                for cid, prob in cand_prob_pairs:
                    thresh = threshold_s2 if cid.startswith('S2-') else threshold_s3
                    if prob >= thresh:
                        matched.append(cid)
                        
                if not matched:
                    total_singletons += 1
                    match_lines.append(f"{s1_id}\t\n")
                else:
                    total_matches += 1
                    m_str = ",".join(sorted(matched))
                    match_lines.append(f"{s1_id}\t{m_str}\n")
                    
                # Build compact candidate set:
                # 1. ALWAYS include all predicted matches
                # 2. Add highest-confidence remaining candidates up to budget
                compact_set = set(matched)
                for cid, prob in cand_prob_pairs:
                    if len(compact_set) >= max(len(matched), target_candidate_budget):
                        break
                    compact_set.add(cid)
                    
                total_compact_candidates += len(compact_set)
                c_str = ",".join(sorted(compact_set))
                cand_lines.append(f"{s1_id}\t{c_str}\n")
                    
            f_match_out.writelines(match_lines)
            f_cand_out.writelines(cand_lines)

        batch_count = 0
        for line in f_in:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split('\t')
            s1_id = parts[0]
            cands = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            
            batch_s1_ids.append(s1_id)
            batch_cands.append(cands)
            
            if len(batch_s1_ids) >= batch_entities:
                batch_count += 1
                process_entity_batch(batch_s1_ids, batch_cands)
                total_entities_processed += len(batch_s1_ids)
                rate = total_entities_processed / (time.time() - t_stream_start)
                avg_cands = total_compact_candidates / total_entities_processed if total_entities_processed > 0 else 0
                pct_match = (total_matches / total_entities_processed) * 100
                pct_sing = (total_singletons / total_entities_processed) * 100
                print(
                    f"[{time.strftime('%H:%M:%S')}] Batch {batch_count:3d} | "
                    f"Entities: {total_entities_processed:,} | "
                    f"Rate: {rate:.0f} ent/s | "
                    f"Matches: {total_matches:,} ({pct_match:.1f}%) | "
                    f"Singletons: {total_singletons:,} ({pct_sing:.1f}%) | "
                    f"Avg Cands: {avg_cands:.2f}",
                    flush=True
                )
                batch_s1_ids.clear()
                batch_cands.clear()

        # Final batch
        if batch_s1_ids:
            batch_count += 1
            process_entity_batch(batch_s1_ids, batch_cands)
            total_entities_processed += len(batch_s1_ids)
            batch_s1_ids.clear()
            batch_cands.clear()

    total_time = time.time() - t_stream_start
    final_avg_cands = total_compact_candidates / total_entities_processed if total_entities_processed > 0 else 0
    print(f"\n================ 4. STREAMING SCORING COMPLETE ================", flush=True)
    print(f"Total Entities Processed: {total_entities_processed:,}", flush=True)
    print(f"Total Pairs Scored:       {total_scored_pairs:,}", flush=True)
    print(f"Total Matches Found:      {total_matches:,} ({(total_matches/total_entities_processed)*100:.2f}%)", flush=True)
    print(f"Total Singletons:         {total_singletons:,} ({(total_singletons/total_entities_processed)*100:.2f}%)", flush=True)
    print(f"Avg Compact Candidates:   {final_avg_cands:.2f} per entity", flush=True)
    print(f"Scoring Time:             {total_time:.2f}s ({total_entities_processed/total_time:.0f} ent/s)", flush=True)
    print(f"Matches saved to:         {matching_tsv_path}", flush=True)
    print(f"Candidate pairs saved to: {cand_tsv_out_path}", flush=True)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    train_dir = os.path.join(root_dir, "dataset", "train")
    test_dir = os.path.join(root_dir, "dataset", "test")
    output_dir = os.path.join(root_dir, "output")
    exp_dir = os.path.join(root_dir, "code", "business_entity_resolution", "experiments")
    model_path = os.path.join(exp_dir, "xgb_pair_matcher.joblib")
    cand_path = os.path.join(output_dir, "candidate_pairs_full_100.tsv")

    if not os.path.exists(cand_path):
        raise FileNotFoundError(f"Candidate file not found at {cand_path}")

    # Step 1: Train or Load Model
    matcher = train_or_load_model(train_dir=train_dir, model_save_path=model_path, model_type='xgboost')

    # Step 2: Stream and Score
    run_fast_stream_scoring(
        test_dir=test_dir,
        output_dir=output_dir,
        candidate_tsv_path=cand_path,
        matcher=matcher,
        threshold_s2=0.78,
        threshold_s3=0.78,
        target_candidate_budget=6,
        batch_entities=15000,
        n_jobs=-1
    )
