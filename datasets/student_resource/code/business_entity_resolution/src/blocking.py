import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

COMMON_NAME_STOPWORDS = {
    'company', 'store', 'shop', 'india', 'private', 'limited', 'restaurant', 
    'services', 'solutions', 'enterprises', 'group', 'holdings', 'international', 
    'tech', 'technologies', 'industries', 'ltd', 'pvt', 'inc', 'corp', 'llc', 
    'co', 'and', 'the', 'for', 'center', 'centre', 'mart', 'plaza', 'traders'
}

COMMON_ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'highway', 'hwy',
    'apartment', 'apt', 'building', 'bldg', 'floor', 'fl', 'suite', 'ste', 'house',
    'h', 'no', 'number', 'num', 'opposite', 'opp', 'near', 'post', 'office', 'po',
    'district', 'dist', 'city', 'state', 'country', 'india', 'usa', 'united', 'states'
}

def extract_digits(text: str) -> str:
    if not text:
        return ""
    nums = re.findall(r'\d+', str(text))
    return "_".join(nums) if nums else ""

def extract_addr_codes(text: str) -> List[str]:
    if not text:
        return []
    text_clean = str(text).lower()
    # Extract hyphenated/slashed codes like 'd-72', 'd-754', '39-17-48/1', 'c-105'
    toks = re.findall(r'\b[a-z0-9]+(?:[\-\/][a-z0-9]+)+\b|\b[a-z0-9]{2,10}\b', text_clean)
    codes = [t for t in toks if any(c.isdigit() for c in t) and t not in COMMON_ADDR_STOPWORDS]
    return codes

def extract_domain_stem(name: str) -> str:
    if not name:
        return ""
    m = re.search(r'\b([a-z0-9]+)\.(com|org|net|in|co|gov|edu)\b', str(name).lower())
    return m.group(1) if m else ""

def extract_compact_name(name: str) -> str:
    if not name:
        return ""
    clean = re.sub(r'[^a-z0-9]', '', str(name).lower())
    return clean

class MultiPassBlocker:
    def __init__(self, max_candidates_per_s1: int = 300, tfidf_top_k: int = 35):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.tfidf_top_k = tfidf_top_k
        self.inverted_indexes = defaultdict(lambda: defaultdict(list))
        
    def fit_index_target_records(self, target_records: List[dict]):
        """
        Builds high-recall multi-pass inverted indexes for candidate records.
        """
        self.target_records = target_records
        self.target_ids = [r['entity_id'] for r in target_records]
        
        for idx, r in enumerate(target_records):
            eid = r['entity_id']
            
            # 1. Exact Name & Variant Keys
            nb = r.get('name_basic', '')
            if nb:
                self.inverted_indexes['name_basic'][nb].append(eid)
                
            ns = r.get('name_no_suf', '')
            if ns:
                self.inverted_indexes['name_no_suf'][ns].append(eid)

            na = r.get('name_alnum', '')
            if na:
                self.inverted_indexes['name_alnum'][na].append(eid)
                
            nss = r.get('name_no_suf_sorted', '') or (" ".join(sorted(ns.split())) if ns else '')
            if nss:
                self.inverted_indexes['name_no_suf_sorted'][nss].append(eid)
                
            # Compact Name (e.g. krystynasenergy, cornerpilates, tulsapediatricdentistry)
            ncomp = extract_compact_name(ns)
            if ncomp and len(ncomp) >= 4:
                self.inverted_indexes['name_compact'][ncomp].append(eid)

            # Domain Stem
            dstem = extract_domain_stem(r.get('original_name', ''))
            if dstem:
                self.inverted_indexes['domain_stem'][dstem].append(eid)
                self.inverted_indexes['name_compact'][dstem].append(eid)
                
            # 2. Adjacent Token Pairs
            tokens = [t for t in ns.split() if t not in COMMON_NAME_STOPWORDS and len(t) >= 2]
            if len(tokens) >= 2:
                for i in range(len(tokens) - 1):
                    pair_key = f"{tokens[i]}_{tokens[i+1]}"
                    self.inverted_indexes['token_pair'][pair_key].append(eid)
            elif len(tokens) == 1:
                self.inverted_indexes['single_token'][tokens[0]].append(eid)

            # 3. All Sorted 2-Token Combinations up to 10 tokens
            all_name_tokens = sorted(list(set([t for t in ns.split() if len(t) >= 2])))
            if len(all_name_tokens) >= 2:
                for t1, t2 in combinations(all_name_tokens[:10], 2):
                    sort_pair = f"{t1}_{t2}"
                    self.inverted_indexes['sort_pair'][sort_pair].append(eid)

            # 4. Address 2-Token Pairs
            ab = r.get('addr_basic', '')
            addr_toks = [t for t in re.findall(r'\b[a-z0-9]{3,}\b', ab) if t not in COMMON_ADDR_STOPWORDS and not t.isdigit()]
            if len(addr_toks) >= 2:
                for i in range(len(addr_toks) - 1):
                    apair = f"{addr_toks[i]}_{addr_toks[i+1]}"
                    self.inverted_indexes['addr_pair'][apair].append(eid)

            # 5. Address Codes (e.g. d-72, d-754, 13822, 8935, c-105, 39-17-48/1)
            acodes = extract_addr_codes(ab)
            for ac in acodes:
                self.inverted_indexes['addr_code'][ac].append(eid)
                if tokens:
                    self.inverted_indexes['code_token'][f"{ac}_{tokens[0]}"].append(eid)

            # 6. House Number / Street Digits + Name Token
            hnum = r.get('house_num', '') or extract_digits(ab)
            pcode = r.get('postal_code', '')
            
            if hnum:
                for t in tokens[:4]:
                    self.inverted_indexes['num_token'][f"{hnum}_{t}"].append(eid)
                if addr_toks:
                    self.inverted_indexes['num_addr_token'][f"{hnum}_{addr_toks[0]}"].append(eid)

            if pcode and len(pcode) >= 3:
                self.inverted_indexes['postal'][pcode].append(eid)
                if hnum:
                    self.inverted_indexes['postal_house'][pcode + '_' + hnum].append(eid)
                if tokens:
                    for t in tokens[:3]:
                        self.inverted_indexes['postal_token'][pcode + '_' + t].append(eid)

            # 7. Address Digits Sequence
            addr_digits = extract_digits(ab)
            if addr_digits and len(addr_digits) >= 3:
                self.inverted_indexes['addr_digits'][addr_digits].append(eid)

    def generate_candidates(self, s1_records: List[dict]) -> Dict[str, Set[str]]:
        """
        Generates candidate pool with priority-ordered candidate selection.
        High-precision passes (exact name, token pairs) are preserved first.
        """
        try:
            s1_names = [r.get('name_no_suf', '') for r in s1_records]
            target_names = [r.get('name_no_suf', '') for r in self.target_records]
            tfidf_cands_list = generate_tfidf_candidates_chunked(
                s1_names, target_names, self.target_ids, top_k=self.tfidf_top_k, min_sim=0.15
            )
        except Exception:
            tfidf_cands_list = [[] for _ in s1_records]

        final_candidates = {}
        
        for r, tfidf_cands in zip(s1_records, tfidf_cands_list):
            s1_id = r['entity_id']
            ordered_cands = []
            seen = set()
            
            def add_cands(cand_list):
                for cid in cand_list:
                    if cid not in seen:
                        seen.add(cid)
                        ordered_cands.append(cid)

            # Tier 1: Exact Name & Variants (Highest Confidence)
            nb = r.get('name_basic', '')
            if nb:
                add_cands(self.inverted_indexes['name_basic'].get(nb, []))
                
            ns = r.get('name_no_suf', '')
            if ns:
                add_cands(self.inverted_indexes['name_no_suf'].get(ns, []))

            na = r.get('name_alnum', '')
            if na:
                add_cands(self.inverted_indexes['name_alnum'].get(na, []))
                
            nss = r.get('name_no_suf_sorted', '') or (" ".join(sorted(ns.split())) if ns else '')
            if nss:
                add_cands(self.inverted_indexes['name_no_suf_sorted'].get(nss, []))

            # Compact Name & Domain Stem
            ncomp = extract_compact_name(ns)
            if ncomp and len(ncomp) >= 4:
                add_cands(self.inverted_indexes['name_compact'].get(ncomp, []))

            dstem = extract_domain_stem(r.get('original_name', ''))
            if dstem:
                add_cands(self.inverted_indexes['domain_stem'].get(dstem, []))
                add_cands(self.inverted_indexes['name_compact'].get(dstem, []))

            # Tier 2: Token Pairs (Adjacent & Combinations)
            tokens = [t for t in ns.split() if t not in COMMON_NAME_STOPWORDS and len(t) >= 2]
            if len(tokens) >= 2:
                for i in range(len(tokens) - 1):
                    pair_key = f"{tokens[i]}_{tokens[i+1]}"
                    add_cands(self.inverted_indexes['token_pair'].get(pair_key, []))
            elif len(tokens) == 1:
                matches = self.inverted_indexes['single_token'].get(tokens[0], [])
                if len(matches) <= 500:
                    add_cands(matches)

            all_name_tokens = sorted(list(set([t for t in ns.split() if len(t) >= 2])))
            if len(all_name_tokens) >= 2:
                for t1, t2 in combinations(all_name_tokens[:10], 2):
                    sort_pair = f"{t1}_{t2}"
                    add_cands(self.inverted_indexes['sort_pair'].get(sort_pair, []))

            # Tier 3: House Number / Postal Code / Address Codes
            ab = r.get('addr_basic', '')
            hnum = r.get('house_num', '') or extract_digits(ab)
            pcode = r.get('postal_code', '')
            addr_toks = [t for t in re.findall(r'\b[a-z0-9]{3,}\b', ab) if t not in COMMON_ADDR_STOPWORDS and not t.isdigit()]

            acodes = extract_addr_codes(ab)
            for ac in acodes:
                ac_matches = self.inverted_indexes['addr_code'].get(ac, [])
                if len(ac_matches) <= 300:
                    add_cands(ac_matches)
                if tokens:
                    add_cands(self.inverted_indexes['code_token'].get(f"{ac}_{tokens[0]}", []))

            if hnum:
                for t in tokens[:4]:
                    num_token_key = f"{hnum}_{t}"
                    add_cands(self.inverted_indexes['num_token'].get(num_token_key, []))
                if addr_toks:
                    num_addr_key = f"{hnum}_{addr_toks[0]}"
                    add_cands(self.inverted_indexes['num_addr_token'].get(num_addr_key, []))
                
            if pcode and len(pcode) >= 3:
                if hnum:
                    add_cands(self.inverted_indexes['postal_house'].get(pcode + '_' + hnum, []))
                if tokens:
                    for t in tokens[:3]:
                        add_cands(self.inverted_indexes['postal_token'].get(pcode + '_' + t, []))
                p_matches = self.inverted_indexes['postal'].get(pcode, [])
                if len(p_matches) <= 300:
                    add_cands(p_matches)

            # Tier 4: Address Token Pairs & Digits
            if len(addr_toks) >= 2:
                for i in range(len(addr_toks) - 1):
                    apair = f"{addr_toks[i]}_{addr_toks[i+1]}"
                    matches = self.inverted_indexes['addr_pair'].get(apair, [])
                    if len(matches) <= 300:
                        add_cands(matches)

            addr_digits = extract_digits(ab)
            if addr_digits and len(addr_digits) >= 3:
                ad_matches = self.inverted_indexes['addr_digits'].get(addr_digits, [])
                if len(ad_matches) <= 300:
                    add_cands(ad_matches)

            # Tier 5: TF-IDF Character N-gram Cosine Candidates
            add_cands(tfidf_cands)

            # Cap ordered candidates safely
            final_candidates[s1_id] = set(ordered_cands[:self.max_candidates_per_s1])
            
        return final_candidates

def generate_tfidf_candidates_chunked(
    s1_names: List[str], 
    target_names: List[str], 
    target_ids: List[str],
    top_k: int = 35,
    min_sim: float = 0.15,
    chunk_size: int = 5000
) -> List[List[str]]:
    """
    Computes top-K TF-IDF character n-gram cosine candidates using sparse dot product.
    """
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
    all_names = s1_names + target_names
    vectorizer.fit(all_names)
    
    s1_mat = vectorizer.transform(s1_names)
    target_mat = vectorizer.transform(target_names)
    
    target_ids_arr = np.array(target_ids)
    num_s1 = s1_mat.shape[0]
    
    results = []
    for start in range(0, num_s1, chunk_size):
        end = min(start + chunk_size, num_s1)
        s1_chunk = s1_mat[start:end]
        
        sim_mat = s1_chunk.dot(target_mat.T)
        
        for i in range(sim_mat.shape[0]):
            row = sim_mat[i]
            if row.nnz == 0:
                results.append([])
                continue
                
            data = row.data
            indices = row.indices
            
            mask = data >= min_sim
            data_filtered = data[mask]
            indices_filtered = indices[mask]
            
            if len(data_filtered) == 0:
                results.append([])
                continue
                
            if len(data_filtered) > top_k:
                top_idx = np.argpartition(data_filtered, -top_k)[-top_k:]
                indices_filtered = indices_filtered[top_idx]
                
            cand_eids = target_ids_arr[indices_filtered].tolist()
            results.append(cand_eids)
            
    return results
