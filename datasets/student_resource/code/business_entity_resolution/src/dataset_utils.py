import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from normalization import build_normalized_record_views, normalize_dataframe_fast

def load_source_tsv(file_path: str) -> pd.DataFrame:
    """
    Loads TSV file safely.
    """
    return pd.read_csv(file_path, sep="\t", dtype=str)

def load_ground_truth_dict(gt_file_path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(gt_file_path, sep="\t", dtype=str)
    s1_ids = df['source1_entity_id'].values
    m_ids_col = df['matched_entity_ids'].fillna('').values
    gt_dict = {}
    for i in range(len(s1_ids)):
        m_str = m_ids_col[i].strip()
        if not m_str:
            gt_dict[s1_ids[i]] = set()
        else:
            gt_dict[s1_ids[i]] = set([x.strip() for x in m_str.split(',') if x.strip()])
    return gt_dict

def normalize_dataframe(df: pd.DataFrame) -> List[dict]:
    return normalize_dataframe_fast(df)

def create_entity_validation_split(
    s1_records: List[dict],
    s2_records: List[dict],
    s3_records: List[dict],
    gt_dict: Dict[str, Set[str]],
    val_num_entities: int = 50000,
    seed: int = 42
) -> Tuple[List[dict], List[dict], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Creates an entity-level validation split without any record leakage.
    Returns (train_s1, val_s1, train_gt, val_gt).
    """
    random.seed(seed)
    all_s1_ids = [r['entity_id'] for r in s1_records]
    
    # Shuffle and select val_num_entities
    shuffled = list(all_s1_ids)
    random.shuffle(shuffled)
    val_s1_ids = set(shuffled[:val_num_entities])
    
    train_s1 = [r for r in s1_records if r['entity_id'] not in val_s1_ids]
    val_s1 = [r for r in s1_records if r['entity_id'] in val_s1_ids]
    
    train_gt = {k: v for k, v in gt_dict.items() if k not in val_s1_ids}
    val_gt = {k: v for k, v in gt_dict.items() if k in val_s1_ids}
    
    return train_s1, val_s1, train_gt, val_gt

def extract_pairs_and_labels(
    s1_records: List[dict],
    target_records: List[dict],
    candidates_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    max_negatives_per_s1: int = 5
) -> Tuple[List[Tuple[str, str]], np.ndarray, Dict[str, dict]]:
    """
    Extracts positive pairs and candidate hard-negative pairs for training/validation.
    """
    target_dict = {r['entity_id']: r for r in target_records}
    s1_dict = {r['entity_id']: r for r in s1_records}
    
    pairs = []
    labels = []
    
    for s1_rec in s1_records:
        s1_id = s1_rec['entity_id']
        true_gt = gt_dict.get(s1_id, set())
        cands = candidates_dict.get(s1_id, set())
        
        positives = list(true_gt.intersection(cands))
        negatives = list(cands - true_gt)
        
        # Add all positives
        for pos_id in positives:
            if pos_id in target_dict:
                pairs.append((s1_id, pos_id))
                labels.append(1)
                
        # Subsample hard negatives if too many
        if len(negatives) > max_negatives_per_s1:
            sampled_negs = random.sample(negatives, max_negatives_per_s1)
        else:
            sampled_negs = negatives
            
        for neg_id in sampled_negs:
            if neg_id in target_dict:
                pairs.append((s1_id, neg_id))
                labels.append(0)
                
    return pairs, np.array(labels, dtype=int), s1_dict, target_dict
