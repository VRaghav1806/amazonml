import os
import sys
import pandas as pd
import numpy as np

sys.path.append("code/business_entity_resolution/src")

from normalization import normalize_dataframe_fast
from dataset_utils import load_ground_truth_dict
from blocking import MultiPassBlocker

def diagnose():
    data_dir = "dataset/train"
    print("Loading data for exact blocking diagnosis...", flush=True)
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str, nrows=2000)
    sub_s1_ids = set(df_s1['entity_id'])
    
    print("Reading ground truth...", flush=True)
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
    df_gt_sub = df_gt[df_gt['source1_entity_id'].isin(sub_s1_ids)]
    
    gt_sub = {}
    gt_s2_ids = set()
    gt_s3_ids = set()
    for _, row in df_gt_sub.iterrows():
        s1_id = row['source1_entity_id']
        m_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        m_ids = set([x.strip() for x in m_str.split(',') if x.strip()])
        gt_sub[s1_id] = m_ids
        for mid in m_ids:
            if mid.startswith('S2-'):
                gt_s2_ids.add(mid)
            elif mid.startswith('S3-'):
                gt_s3_ids.add(mid)
                
    print(f"Sampled S1: {len(df_s1)}, GT S2 targets: {len(gt_s2_ids)}, GT S3 targets: {len(gt_s3_ids)}", flush=True)

    print("Loading matching GT records from S2 and S3...", flush=True)
    df_s2_all = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str)
    df_s3_all = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str)

    df_s2_gt = df_s2_all[df_s2_all['entity_id'].isin(gt_s2_ids)]
    df_s3_gt = df_s3_all[df_s3_all['entity_id'].isin(gt_s3_ids)]
    
    # Add random background distractor noise records
    df_s2_bg = df_s2_all.iloc[:50000]
    df_s3_bg = df_s3_all.iloc[:50000]
    
    df_s2_target = pd.concat([df_s2_gt, df_s2_bg]).drop_duplicates(subset=['entity_id'])
    df_s3_target = pd.concat([df_s3_gt, df_s3_bg]).drop_duplicates(subset=['entity_id'])
    
    print(f"Target pool S2: {len(df_s2_target)}, S3: {len(df_s3_target)}", flush=True)

    s1_recs = normalize_dataframe_fast(df_s1)
    s2_recs = normalize_dataframe_fast(df_s2_target)
    s3_recs = normalize_dataframe_fast(df_s3_target)
    
    target_recs = s2_recs + s3_recs
    target_dict = {r['entity_id']: r for r in target_recs}
    s1_dict = {r['entity_id']: r for r in s1_recs}

    blocker = MultiPassBlocker(max_candidates_per_s1=150, tfidf_top_k=25)
    print("Fitting blocker...", flush=True)
    blocker.fit_index_target_records(target_recs)
    
    print("Generating candidates...", flush=True)
    cands = blocker.generate_candidates(s1_recs)

    missed_count = 0
    total_eval_gt = 0
    print("\n--- MISSED GT EXAMPLES ---", flush=True)
    for s1_id, gt_set in gt_sub.items():
        found = cands.get(s1_id, set())
        eval_gt_set = set([gid for gid in gt_set if gid in target_dict])
        total_eval_gt += len(eval_gt_set)
        for gt_id in eval_gt_set:
            if gt_id not in found:
                missed_count += 1
                if missed_count <= 10:
                    s1_r = s1_dict[s1_id]
                    gt_r = target_dict.get(gt_id, {})
                    print(f"\nMissed #{missed_count}:", flush=True)
                    s1_name_safe = s1_r.get('original_name', '').encode('ascii', 'ignore').decode()
                    s1_addr_safe = s1_r.get('original_address', '').encode('ascii', 'ignore').decode()
                    gt_name_safe = gt_r.get('original_name', '').encode('ascii', 'ignore').decode()
                    gt_addr_safe = gt_r.get('original_address', '').encode('ascii', 'ignore').decode()
                    print(f"  S1 ID: {s1_id} | Name: '{s1_name_safe}' | Addr: '{s1_addr_safe}'", flush=True)
                    print(f"  GT ID: {gt_id} | Name: '{gt_name_safe}' | Addr: '{gt_addr_safe}'", flush=True)
                    
    recall = (total_eval_gt - missed_count) / max(1, total_eval_gt)
    print(f"\nExact Blocking Recall (on true targets): {recall*100:.3f}% ({total_eval_gt - missed_count}/{total_eval_gt})", flush=True)

if __name__ == "__main__":
    diagnose()
