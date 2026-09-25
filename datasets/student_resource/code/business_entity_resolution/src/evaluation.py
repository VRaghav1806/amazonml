import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, List

def compute_f05_single_entity(gt_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F0.5 for a single S1 entity.
    Returns (precision, recall, f05).
    """
    if len(gt_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0
    
    if len(pred_set) == 0:
        return 1.0, 0.0, 0.0
        
    tp = len(gt_set.intersection(pred_set))
    if tp == 0:
        return 0.0, 0.0, 0.0
        
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return precision, recall, f05

def evaluate_macro_metrics(gt_dict: Dict[str, Set[str]], pred_dict: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Computes macro-averaged metrics across all S1 entities in gt_dict.
    """
    precisions = []
    recalls = []
    f05s = []
    
    num_singletons = 0
    singleton_correct = 0
    
    num_gt_matches = 0
    num_pred_matches = 0
    
    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        
        p, r, f = compute_f05_single_entity(gt_set, pred_set)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f)
        
        num_gt_matches += len(gt_set)
        num_pred_matches += len(pred_set)
        
        if len(gt_set) == 0:
            num_singletons += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05s))
    singleton_acc = float(singleton_correct / num_singletons) if num_singletons > 0 else 1.0
    
    return {
        'macro_f05': macro_f05,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'singleton_accuracy': singleton_acc,
        'num_entities': len(gt_dict),
        'num_gt_matches': num_gt_matches,
        'num_pred_matches': num_pred_matches,
        'num_singletons': num_singletons,
        'singleton_correct': singleton_correct
    }

def compute_blocking_recall(gt_dict: Dict[str, Set[str]], candidate_dict: Dict[str, Set[str]]) -> Tuple[float, int, int]:
    """
    Computes blocking recall: total true positive candidates found / total ground truth matches.
    Returns (blocking_recall, total_found, total_gt_matches).
    """
    total_gt = 0
    total_found = 0
    
    for s1_id, gt_set in gt_dict.items():
        total_gt += len(gt_set)
        cand_set = candidate_dict.get(s1_id, set())
        found = len(gt_set.intersection(cand_set))
        total_found += found
        
    recall = total_found / total_gt if total_gt > 0 else 1.0
    return recall, total_found, total_gt
