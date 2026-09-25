import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class PairMatcher:
    def __init__(self, model_type: str = 'xgboost', params: dict = None):
        self.model_type = model_type.lower()
        self.params = params if params else {}
        self.model = self._build_model()
        
    def _build_model(self):
        if self.model_type == 'logistic':
            base_params = {'max_iter': 1000, 'random_state': 42}
            base_params.update(self.params)
            return LogisticRegression(**base_params)
        elif self.model_type == 'random_forest':
            base_params = {'n_estimators': 100, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1}
            base_params.update(self.params)
            return RandomForestClassifier(**base_params)
        elif self.model_type == 'extra_trees':
            base_params = {'n_estimators': 100, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1}
            base_params.update(self.params)
            return ExtraTreesClassifier(**base_params)
        elif self.model_type == 'xgboost':
            base_params = {
                'n_estimators': 250,
                'max_depth': 8,
                'learning_rate': 0.08,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            base_params.update(self.params)
            return xgb.XGBClassifier(**base_params)
        elif self.model_type == 'lightgbm':
            base_params = {
                'n_estimators': 250,
                'max_depth': 8,
                'learning_rate': 0.08,
                'num_leaves': 63,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            base_params.update(self.params)
            return lgb.LGBMClassifier(**base_params)
        elif self.model_type == 'catboost':
            base_params = {
                'iterations': 250,
                'depth': 7,
                'learning_rate': 0.08,
                'random_seed': 42,
                'verbose': 0
            }
            base_params.update(self.params)
            return CatBoostClassifier(**base_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs[:, 0]
        else:
            # Fallback
            preds = self.model.predict(X)
            return preds.astype(float)

class EntityDecisionEngine:
    """
    Decides final matches for each S1 entity given pairwise match probabilities.
    Features threshold tuning and explicit singleton detection.
    """
    def __init__(self, threshold_s2: float = 0.85, threshold_s3: float = 0.85, margin_threshold: float = 0.05):
        self.threshold_s2 = threshold_s2
        self.threshold_s3 = threshold_s3
        self.margin_threshold = margin_threshold

    def predict_entity_matches(
        self, 
        s1_ids: List[str], 
        pairs: List[Tuple[str, str]], 
        probs: np.ndarray
    ) -> Dict[str, Set[str]]:
        """
        Groups predictions by s1_id and applies singleton and multi-match decision thresholding.
        """
        s1_to_cand_probs = {}
        for s1_id in s1_ids:
            s1_to_cand_probs[s1_id] = []
            
        for (s1_id, cand_id), prob in zip(pairs, probs):
            s1_to_cand_probs[s1_id].append((cand_id, prob))
            
        final_matches = {}
        for s1_id, cand_list in s1_to_cand_probs.items():
            if not cand_list:
                final_matches[s1_id] = set()
                continue
                
            # Filter candidates based on thresholds
            selected = set()
            for cand_id, prob in cand_list:
                thresh = self.threshold_s2 if cand_id.startswith('S2-') else self.threshold_s3
                if prob >= thresh:
                    selected.add(cand_id)
                    
            final_matches[s1_id] = selected
            
        return final_matches
