import os
import sys
import time
import gc
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_dataframe_fast
from features import extract_features_parallel
from dataset_utils import load_source_tsv

import re

DISCRIMINATIVE_MODIFIERS = {
    'west', 'east', 'north', 'south', 'central', 'holdings', 'holding', 
    'exports', 'export', 'services', 'service', 'international', 'industries', 
    'logistics', 'properties', 'realty', 'finance', 'financial', 'motors'
}

def extract_primary_num(addr):
    if not addr or pd.isna(addr) or str(addr) == 'nan':
        return ''
    m = re.search(r'\b(\d+[a-z]?)\b', str(addr).lower())
    return m.group(1) if m else ''

def nums_compatible(n1, n2):
    if not n1 or not n2:
        return True
    if n1 == n2:
        return True
    d1 = re.sub(r'[^0-9]', '', n1)
    d2 = re.sub(r'[^0-9]', '', n2)
    if d1 == d2 and d1:
        return True
    if abs(len(d1) - len(d2)) == 1 and (d1.startswith(d2) or d2.startswith(d1)):
        return True
    return False

def name_modifiers_compatible(name1, name2):
    toks1 = set(re.findall(r'[a-z]+', str(name1).lower()))
    toks2 = set(re.findall(r'[a-z]+', str(name2).lower()))
    for mod in DISCRIMINATIVE_MODIFIERS:
        if mod in toks2 and mod not in toks1:
            return False
        if mod in toks1 and mod not in toks2:
            return False
    return True

def is_candidate_compatible(s1_rec, t_rec):
    if t_rec is None:
        return False
    if s1_rec.country and t_rec.country and s1_rec.country != t_rec.country:
        return False
    if not name_modifiers_compatible(s1_rec.original_name, t_rec.original_name):
        return False
    n1 = extract_primary_num(s1_rec.original_address)
    n2 = extract_primary_num(t_rec.original_address)
    if not nums_compatible(n1, n2):
        return False
    if s1_rec.addr_basic and t_rec.addr_basic:
        s1_a = str(s1_rec.addr_basic).lower()
        t_a = str(t_rec.addr_basic).lower()
        tsort = fuzz.token_sort_ratio(s1_a, t_a)
        tset = fuzz.token_set_ratio(s1_a, t_a)
        if tsort < 25 and tset < 35:
            return False
    return True

def run_precision_refinement(
    test_dir: str,
    output_dir: str,
    model_path: str,
    min_threshold: float = 0.88,
    margin: float = 0.08,
    max_per_source: int = 4,
    batch_entities: int = 25000,
    n_jobs: int = -1
):
    print("================ 1. LOADING MODEL & LOOKUPS ================", flush=True)
    t0 = time.time()
    
    print(f"Loading trained matcher from: {model_path}...", flush=True)
    matcher = joblib.load(model_path)
    
    print("Loading test_source1.tsv...", flush=True)
    df_s1 = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s1_recs = normalize_dataframe_fast(df_s1)
    test_s1_lookup = {r.entity_id: r for r in test_s1_recs}
    del df_s1, test_s1_recs
    gc.collect()
    print(f"Loaded {len(test_s1_lookup):,} S1 records.", flush=True)
    
    print("Loading test_source2.tsv...", flush=True)
    df_s2 = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s2_recs = normalize_dataframe_fast(df_s2)
    del df_s2
    gc.collect()
    
    print("Loading test_source3.tsv...", flush=True)
    df_s3 = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"))
    test_s3_recs = normalize_dataframe_fast(df_s3)
    del df_s3
    gc.collect()
    
    test_target_lookup = {}
    for r in test_s2_recs:
        test_target_lookup[r.entity_id] = r
    for r in test_s3_recs:
        test_target_lookup[r.entity_id] = r
    del test_s2_recs, test_s3_recs
    gc.collect()
    print(f"Loaded {len(test_target_lookup):,} Target records in {time.time() - t0:.1f}s", flush=True)
    
    backup_unpruned = os.path.join(output_dir, "matching_results_unpruned_0.78.tsv")
    matching_in_path = backup_unpruned if os.path.exists(backup_unpruned) else os.path.join(output_dir, "matching_results.tsv")
    matching_out_path = os.path.join(output_dir, "matching_results_refined.tsv")
    
    print("\n================ 2. STREAMING & RE-SCORING MATCHES ================", flush=True)
    print(f"Input:             {matching_in_path}", flush=True)
    print(f"Output:            {matching_out_path}", flush=True)
    print(f"Min Threshold:     {min_threshold}", flush=True)
    print(f"Confidence Margin: {margin}", flush=True)
    print(f"Max Per Source:    {max_per_source} (max S2={max_per_source}, max S3={max_per_source})", flush=True)
    
    total_entities = 0
    total_matches = 0
    total_singletons = 0
    total_matched_ids = 0
    t_stream_start = time.time()
    
    with open(matching_in_path, 'r', encoding='utf-8') as f_in, \
         open(matching_out_path, 'w', encoding='utf-8') as f_out:
         
        header = f_in.readline()
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        
        batch_s1_ids = []
        batch_cands = []
        
        def process_batch(s1_ids, cands_list):
            nonlocal total_entities, total_matches, total_singletons, total_matched_ids
            
            pairs = []
            slices = []
            
            for s1_id, cands in zip(s1_ids, cands_list):
                s_idx = len(pairs)
                for cid in cands:
                    pairs.append((s1_id, cid))
                e_idx = len(pairs)
                slices.append((s1_id, cands, s_idx, e_idx))
                
            if pairs:
                X_batch = extract_features_parallel(pairs, test_s1_lookup, test_target_lookup, batch_size=25000, n_jobs=n_jobs)
                probs = matcher.predict_proba(X_batch)
            else:
                probs = np.array([])
                
            out_lines = []
            for s1_id, cands, s_idx, e_idx in slices:
                s1_rec = test_s1_lookup[s1_id]
                ent_probs = probs[s_idx:e_idx] if e_idx > s_idx else np.array([])
                
                if len(ent_probs) == 0:
                    total_singletons += 1
                    out_lines.append(f"{s1_id}\t\n")
                    continue
                    
                max_p = np.max(ent_probs)
                if max_p < min_threshold:
                    total_singletons += 1
                    out_lines.append(f"{s1_id}\t\n")
                    continue
                    
                s2_matches = []
                s3_matches = []
                
                for cid, p in zip(cands, ent_probs):
                    if p < min_threshold:
                        continue
                    if (max_p - p) > margin:
                        continue
                        
                    t_rec = test_target_lookup.get(cid)
                    if not is_candidate_compatible(s1_rec, t_rec):
                        continue
                            
                    if cid.startswith('S2-'):
                        s2_matches.append((cid, p))
                    else:
                        s3_matches.append((cid, p))
                        
                s2_matches.sort(key=lambda x: x[1], reverse=True)
                s3_matches.sort(key=lambda x: x[1], reverse=True)
                
                selected = [c for c, p in s2_matches[:max_per_source]] + [c for c, p in s3_matches[:max_per_source]]
                
                if not selected:
                    total_singletons += 1
                    out_lines.append(f"{s1_id}\t\n")
                else:
                    total_matches += 1
                    total_matched_ids += len(selected)
                    m_str = ",".join(sorted(selected))
                    out_lines.append(f"{s1_id}\t{m_str}\n")
                    
            f_out.writelines(out_lines)
            total_entities += len(s1_ids)

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
                process_batch(batch_s1_ids, batch_cands)
                rate = total_entities / (time.time() - t_stream_start)
                avg_m = total_matched_ids / total_entities if total_entities > 0 else 0
                pct_m = (total_matches / total_entities) * 100
                pct_s = (total_singletons / total_entities) * 100
                print(
                    f"[{time.strftime('%H:%M:%S')}] Batch {batch_count:2d} | "
                    f"Entities: {total_entities:,} | "
                    f"Rate: {rate:.0f} ent/s | "
                    f"Matches: {total_matches:,} ({pct_m:.1f}%) | "
                    f"Singletons: {total_singletons:,} ({pct_s:.1f}%) | "
                    f"AvgMatches: {avg_m:.2f}",
                    flush=True
                )
                batch_s1_ids.clear()
                batch_cands.clear()

        if batch_s1_ids:
            batch_count += 1
            process_batch(batch_s1_ids, batch_cands)
            batch_s1_ids.clear()
            batch_cands.clear()

    total_time = time.time() - t_stream_start
    final_avg_m = total_matched_ids / total_entities if total_entities > 0 else 0
    print(f"\n================ 3. REFINEMENT COMPLETE ================", flush=True)
    print(f"Total Entities Processed: {total_entities:,}", flush=True)
    print(f"Total Matches Found:      {total_matches:,} ({(total_matches/total_entities)*100:.2f}%)", flush=True)
    print(f"Total Singletons:         {total_singletons:,} ({(total_singletons/total_entities)*100:.2f}%)", flush=True)
    print(f"Average Matches/Entity:   {final_avg_m:.2f}", flush=True)
    print(f"Refinement Time:          {total_time:.2f}s ({total_entities/total_time:.0f} ent/s)", flush=True)
    print(f"Saved to:                 {matching_out_path}", flush=True)

    # Atomic swap to matching_results.tsv
    final_target_path = os.path.join(output_dir, "matching_results.tsv")
    if os.path.exists(final_target_path):
        os.remove(final_target_path)
    os.rename(matching_out_path, final_target_path)
    print(f"Swapped refined file to {final_target_path} (source pool preserved at {matching_in_path})", flush=True)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    test_dir = os.path.join(root_dir, "dataset", "test")
    output_dir = os.path.join(root_dir, "output")
    exp_dir = os.path.join(root_dir, "code", "business_entity_resolution", "experiments")
    model_path = os.path.join(exp_dir, "xgb_pair_matcher.joblib")

    run_precision_refinement(
        test_dir=test_dir,
        output_dir=output_dir,
        model_path=model_path,
        min_threshold=0.88,
        margin=0.08,
        max_per_source=4,
        batch_entities=25000,
        n_jobs=-1
    )

