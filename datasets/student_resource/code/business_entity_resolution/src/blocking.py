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
    def __init__(self, max_candidates_per_s1: int = 8, tfidf_top_k: int = 5):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.tfidf_top_k = tfidf_top_k
        self.inverted_indexes = defaultdict(lambda: defaultdict(list))
        self.tfidf_vectorizer = None
        self.target_tfidf_mat = None
        self.target_ids_arr = None
        
    def fit_index_target_records(self, target_records: List[dict]):
        """
        Builds high-recall multi-pass inverted indexes for candidate records.
        Also pre-fits and pre-calculates TF-IDF target embeddings.
        """
        self.target_records = target_records
        self.target_ids = [r['entity_id'] if isinstance(r, dict) else r.entity_id for r in target_records]
        self.target_ids_arr = np.array(self.target_ids)
        
        target_names = []
        for idx, r in enumerate(target_records):
            if isinstance(r, dict):
                eid = r.get('entity_id', '')
                nb = r.get('name_basic', '')
                ns = r.get('name_no_suf', '')
                na = r.get('name_alnum', '')
                nss = r.get('name_no_suf_sorted', '') or (" ".join(sorted(ns.split())) if ns else '')
                orig_name = r.get('original_name', '')
                ab = r.get('addr_basic', '')
                hnum = r.get('house_num', '') or extract_digits(ab)
                pcode = r.get('postal_code', '')
            else:
                eid = r.entity_id
                nb = r.name_basic
                ns = r.name_no_suf
                na = r.name_alnum
                nss = r.name_no_suf_sorted or (" ".join(sorted(ns.split())) if ns else '')
                orig_name = r.original_name
                ab = r.addr_basic
                hnum = r.house_num or extract_digits(ab)
                pcode = r.postal_code

            target_names.append(ns)
            
            # 1. Exact Name & Variant Keys
            if nb:
                self.inverted_indexes['name_basic'][nb].append(eid)
            if ns:
                self.inverted_indexes['name_no_suf'][ns].append(eid)
            if na:
                self.inverted_indexes['name_alnum'][na].append(eid)
            if nss:
                self.inverted_indexes['name_no_suf_sorted'][nss].append(eid)
                
            # Compact Name (e.g. krystynasenergy, cornerpilates, tulsapediatricdentistry)
            ncomp = extract_compact_name(ns)
            if ncomp and len(ncomp) >= 4:
                self.inverted_indexes['name_compact'][ncomp].append(eid)

            # Domain Stem
            dstem = extract_domain_stem(orig_name)
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

            # 6. House Number / Street Digits + Name Token / Address Fallbacks
            if hnum:
                for t in tokens[:4]:
                    self.inverted_indexes['num_token'][f"{hnum}_{t}"].append(eid)
                if addr_toks:
                    self.inverted_indexes['num_addr_token'][f"{hnum}_{addr_toks[0]}"].append(eid)
                if len(addr_toks) >= 2:
                    self.inverted_indexes['num_addr_pair'][f"{hnum}_{addr_toks[0]}_{addr_toks[1]}"].append(eid)

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

        # Pre-fit TF-IDF Vectorizer once for all target records with memory-safe settings
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4), min_df=5, max_features=100000, dtype=np.float32
        )
        self.tfidf_vectorizer.fit(target_names)
        self.target_tfidf_mat = self.tfidf_vectorizer.transform(target_names)

    def generate_candidates(self, s1_records: List[dict], n_jobs: int = -1) -> Dict[str, Set[str]]:
        """
        Generates candidate pool with priority-ordered candidate selection.
        Supports multi-core parallel execution with process-shared indexes.
        """
        num_s1 = len(s1_records)
        s1_names = [r.get('name_no_suf', '') if isinstance(r, dict) else r.name_no_suf for r in s1_records]

        try:
            tfidf_cands_list = generate_tfidf_candidates_prefit(
                self.tfidf_vectorizer, s1_names, self.target_tfidf_mat, self.target_ids_arr, top_k=self.tfidf_top_k, min_sim=0.15
            )
        except Exception:
            tfidf_cands_list = [[] for _ in s1_records]

        inv_idx = dict(self.inverted_indexes)
        max_cands = self.max_candidates_per_s1

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(16, os.cpu_count() or 4) if n_jobs in (-1, None) else max(1, n_jobs)

        if num_s1 <= 2000 or max_workers == 1:
            return _process_candidate_chunk(s1_records, tfidf_cands_list, inv_idx, max_cands)

        # Multi-thread processing: threads share inverted_indexes in-memory with ZERO pickling overhead
        batch_size = max(5000, (num_s1 + max_workers * 4 - 1) // (max_workers * 4))
        num_batches = (num_s1 + batch_size - 1) // batch_size

        batches = [
            (s1_records[i * batch_size:(i + 1) * batch_size],
             tfidf_cands_list[i * batch_size:(i + 1) * batch_size],
             max_cands)
            for i in range(num_batches)
        ]

        final_candidates = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_candidate_chunk, b[0], b[1], inv_idx, b[2])
                for b in batches
            ]
            for future in as_completed(futures):
                final_candidates.update(future.result())

        return final_candidates


def _process_candidate_chunk(s1_chunk, tfidf_chunk, inverted_indexes, max_candidates_per_s1):
    final_candidates = {}
    for r, tfidf_cands in zip(s1_chunk, tfidf_chunk):
        if isinstance(r, dict):
            s1_id = r['entity_id']
            nb = r.get('name_basic', '')
            ns = r.get('name_no_suf', '')
            na = r.get('name_alnum', '')
            nss = r.get('name_no_suf_sorted', '') or (" ".join(sorted(ns.split())) if ns else '')
            orig_name = r.get('original_name', '')
            ab = r.get('addr_basic', '')
            hnum = r.get('house_num', '') or extract_digits(ab)
            pcode = r.get('postal_code', '')
        else:
            s1_id = r.entity_id
            nb = r.name_basic
            ns = r.name_no_suf
            na = r.name_alnum
            nss = r.name_no_suf_sorted or (" ".join(sorted(ns.split())) if ns else '')
            orig_name = r.original_name
            ab = r.addr_basic
            hnum = r.house_num or extract_digits(ab)
            pcode = r.postal_code

        ordered_cands = []
        seen = set()

        # Tier 1: Exact Name & Variants (Highest Confidence)
        if nb:
            for cid in inverted_indexes.get('name_basic', {}).get(nb, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)
            
        if ns:
            for cid in inverted_indexes.get('name_no_suf', {}).get(ns, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)

        if na:
            for cid in inverted_indexes.get('name_alnum', {}).get(na, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)
            
        if nss:
            for cid in inverted_indexes.get('name_no_suf_sorted', {}).get(nss, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)

        # Compact Name & Domain Stem
        ncomp = extract_compact_name(ns)
        if ncomp and len(ncomp) >= 4:
            for cid in inverted_indexes.get('name_compact', {}).get(ncomp, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)

        dstem = extract_domain_stem(orig_name)
        if dstem:
            for cid in inverted_indexes.get('domain_stem', {}).get(dstem, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)
            for cid in inverted_indexes.get('name_compact', {}).get(dstem, []):
                if cid not in seen:
                    seen.add(cid); ordered_cands.append(cid)

        # Tier 2: Token Pairs (Adjacent & Combinations)
        tokens = [t for t in ns.split() if t not in COMMON_NAME_STOPWORDS and len(t) >= 2]
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                pair_key = f"{tokens[i]}_{tokens[i+1]}"
                for cid in inverted_indexes.get('token_pair', {}).get(pair_key, []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)
        elif len(tokens) == 1:
            matches = inverted_indexes.get('single_token', {}).get(tokens[0], [])
            if len(matches) <= 500:
                for cid in matches:
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)

        all_name_tokens = sorted(list(set([t for t in ns.split() if len(t) >= 2])))
        if len(all_name_tokens) >= 2:
            for t1, t2 in combinations(all_name_tokens[:10], 2):
                sort_pair = f"{t1}_{t2}"
                for cid in inverted_indexes.get('sort_pair', {}).get(sort_pair, []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)

        # Tier 3: House Number / Postal Code / Address Codes
        addr_toks = [t for t in re.findall(r'\b[a-z0-9]{3,}\b', ab) if t not in COMMON_ADDR_STOPWORDS and not t.isdigit()]
        acodes = extract_addr_codes(ab)
        for ac in acodes:
            ac_matches = inverted_indexes.get('addr_code', {}).get(ac, [])
            if len(ac_matches) <= 300:
                for cid in ac_matches:
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)
            if tokens:
                for cid in inverted_indexes.get('code_token', {}).get(f"{ac}_{tokens[0]}", []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)

        if hnum:
            for t in tokens[:4]:
                for cid in inverted_indexes.get('num_token', {}).get(f"{hnum}_{t}", []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)
            if addr_toks:
                for cid in inverted_indexes.get('num_addr_token', {}).get(f"{hnum}_{addr_toks[0]}", []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)
            if len(addr_toks) >= 2:
                for cid in inverted_indexes.get('num_addr_pair', {}).get(f"{hnum}_{addr_toks[0]}_{addr_toks[1]}", []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)
            
        if pcode and len(pcode) >= 3:
            if hnum:
                for cid in inverted_indexes.get('postal_house', {}).get(pcode + '_' + hnum, []):
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)
            if tokens:
                for t in tokens[:3]:
                    for cid in inverted_indexes.get('postal_token', {}).get(pcode + '_' + t, []):
                        if cid not in seen:
                            seen.add(cid); ordered_cands.append(cid)
            p_matches = inverted_indexes.get('postal', {}).get(pcode, [])
            if len(p_matches) <= 300:
                for cid in p_matches:
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)

        # Tier 4: Address Token Pairs & Digits
        if len(addr_toks) >= 2:
            for i in range(len(addr_toks) - 1):
                apair = f"{addr_toks[i]}_{addr_toks[i+1]}"
                matches = inverted_indexes.get('addr_pair', {}).get(apair, [])
                if len(matches) <= 300:
                    for cid in matches:
                        if cid not in seen:
                            seen.add(cid); ordered_cands.append(cid)

        addr_digits = extract_digits(ab)
        if addr_digits and len(addr_digits) >= 3:
            ad_matches = inverted_indexes.get('addr_digits', {}).get(addr_digits, [])
            if len(ad_matches) <= 300:
                for cid in ad_matches:
                    if cid not in seen:
                        seen.add(cid); ordered_cands.append(cid)

        # Tier 5: TF-IDF Character N-gram Cosine Candidates
        for cid in tfidf_cands:
            if cid not in seen:
                seen.add(cid); ordered_cands.append(cid)

        # Cap ordered candidates safely
        final_candidates[s1_id] = set(ordered_cands[:max_candidates_per_s1])

    return final_candidates

def generate_tfidf_candidates_prefit(
    vectorizer: TfidfVectorizer,
    s1_names: List[str], 
    target_mat: csr_matrix, 
    target_ids_arr: np.ndarray,
    top_k: int = 35,
    min_sim: float = 0.15,
    chunk_size: int = 10000
) -> List[List[str]]:
    """
    Computes top-K TF-IDF character n-gram cosine candidates using pre-fitted target TF-IDF matrix.
    """
    s1_mat = vectorizer.transform(s1_names)
    num_s1 = s1_mat.shape[0]
    
    results = []
    for start in range(0, num_s1, chunk_size):
        end = min(start + chunk_size, num_s1)
        s1_chunk = s1_mat[start:end]
        
        sim_mat = s1_chunk.dot(target_mat.T)
        indptr = sim_mat.indptr
        data = sim_mat.data
        indices = sim_mat.indices
        
        for i in range(sim_mat.shape[0]):
            r_start = indptr[i]
            r_end = indptr[i + 1]
            if r_start == r_end:
                results.append([])
                continue
                
            r_data = data[r_start:r_end]
            r_indices = indices[r_start:r_end]
            
            mask = r_data >= min_sim
            data_filtered = r_data[mask]
            indices_filtered = r_indices[mask]
            
            if len(data_filtered) == 0:
                results.append([])
                continue
                
            if len(data_filtered) > top_k:
                top_idx = np.argpartition(data_filtered, -top_k)[-top_k:]
                indices_filtered = indices_filtered[top_idx]
                
            cand_eids = target_ids_arr[indices_filtered].tolist()
            results.append(cand_eids)
            
    return results


