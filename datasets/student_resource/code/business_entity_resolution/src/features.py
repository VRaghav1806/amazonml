import numpy as np
from rapidfuzz import distance, fuzz
from typing import Dict, List, Any, Tuple

FEATURE_NAMES = [
    # Name features
    'name_fuzz_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_wratio',
    'name_jaro_winkler',
    'name_basic_exact',
    'name_nosuf_exact',
    'name_sorted_exact',
    'name_first_token_exact',
    'name_last_token_exact',
    'name_token_jaccard',
    'name_common_tokens',
    'name_len_diff',
    'name_len_ratio',
    'name_abbr_match',
    
    # Address features
    'addr_fuzz_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    'addr_jaro_winkler',
    'addr_basic_exact',
    'addr_token_jaccard',
    'addr_common_tokens',
    'addr_digits_common',
    'addr_digits_jaccard',
    'postal_code_exact',
    'house_num_exact',
    'addr_len_diff',
    'addr_len_ratio',
    'addr_missing',
    
    # Country features
    'country_exact',
    
    # Cross-field interactions
    'name_sim_x_addr_sim',
    'name_sim_x_country',
    'addr_sim_x_country',
    'name_exact_and_addr_high',
    'name_high_and_postal_match',
    'name_nosuf_exact_and_addr_sim',
    
    # Source features
    'is_s2',
    'is_s3'
]

def extract_pairwise_features(s1_rec: Any, cand_rec: Any) -> List[float]:
    # 1. Name representations
    if hasattr(s1_rec, 'name_basic'):
        s1_nb = s1_rec.name_basic
        c_nb = cand_rec.name_basic
        s1_ns = s1_rec.name_no_suf
        c_ns = cand_rec.name_no_suf
        s1_nss = s1_rec.name_no_suf_sorted
        c_nss = cand_rec.name_no_suf_sorted
        s1_abbr = s1_rec.name_abbr
        c_abbr = cand_rec.name_abbr
        s1_ab = s1_rec.addr_basic
        c_ab = cand_rec.addr_basic
        s1_digs_str = s1_rec.addr_digits
        c_digs_str = cand_rec.addr_digits
        s1_pcode = s1_rec.postal_code
        c_pcode = cand_rec.postal_code
        s1_hnum = s1_rec.house_num
        c_hnum = cand_rec.house_num
        s1_cnt = s1_rec.country
        c_cnt = cand_rec.country
        cand_id = cand_rec.entity_id
    else:
        s1_nb = s1_rec.get('name_basic', '')
        c_nb = cand_rec.get('name_basic', '')
        s1_ns = s1_rec.get('name_no_suf', '')
        c_ns = cand_rec.get('name_no_suf', '')
        s1_nss = s1_rec.get('name_no_suf_sorted', '')
        c_nss = cand_rec.get('name_no_suf_sorted', '')
        s1_abbr = s1_rec.get('name_abbr', '')
        c_abbr = cand_rec.get('name_abbr', '')
        s1_ab = s1_rec.get('addr_basic', '')
        c_ab = cand_rec.get('addr_basic', '')
        s1_digs_str = s1_rec.get('addr_digits', '')
        c_digs_str = cand_rec.get('addr_digits', '')
        s1_pcode = s1_rec.get('postal_code', '')
        c_pcode = cand_rec.get('postal_code', '')
        s1_hnum = s1_rec.get('house_num', '')
        c_hnum = cand_rec.get('house_num', '')
        s1_cnt = s1_rec.get('country', '')
        c_cnt = cand_rec.get('country', '')
        cand_id = cand_rec.get('entity_id', '')

    # RapidFuzz scores with fast paths for identical/empty strings
    if s1_ns and c_ns:
        if s1_ns == c_ns:
            name_ratio = name_partial = name_sort = name_set = name_wratio = name_jw = 1.0
        else:
            name_ratio = fuzz.ratio(s1_ns, c_ns) / 100.0
            name_partial = fuzz.partial_ratio(s1_ns, c_ns) / 100.0
            name_sort = fuzz.token_sort_ratio(s1_ns, c_ns) / 100.0
            name_set = fuzz.token_set_ratio(s1_ns, c_ns) / 100.0
            name_wratio = fuzz.WRatio(s1_ns, c_ns) / 100.0
            name_jw = distance.JaroWinkler.similarity(s1_ns, c_ns)
    else:
        name_ratio = name_partial = name_sort = name_set = name_wratio = name_jw = 0.0

    name_basic_exact = 1.0 if (s1_nb and s1_nb == c_nb) else 0.0
    name_nosuf_exact = 1.0 if (s1_ns and s1_ns == c_ns) else 0.0
    name_sorted_exact = 1.0 if (s1_nss and s1_nss == c_nss) else 0.0

    # Token features
    s1_tokens = set(s1_ns.split())
    c_tokens = set(c_ns.split())
    
    s1_t_list = s1_ns.split()
    c_t_list = c_ns.split()
    
    first_token_exact = 1.0 if (s1_t_list and c_t_list and s1_t_list[0] == c_t_list[0]) else 0.0
    last_token_exact = 1.0 if (s1_t_list and c_t_list and s1_t_list[-1] == c_t_list[-1]) else 0.0

    common_toks = s1_tokens.intersection(c_tokens)
    union_toks = s1_tokens.union(c_tokens)
    num_common_toks = len(common_toks)
    name_token_jaccard = num_common_toks / len(union_toks) if union_toks else 0.0

    name_len_diff = abs(len(s1_ns) - len(c_ns))
    name_len_ratio = min(len(s1_ns), len(c_ns)) / max(len(s1_ns), len(c_ns)) if max(len(s1_ns), len(c_ns)) > 0 else 0.0
    name_abbr_match = 1.0 if (s1_abbr and c_abbr and s1_abbr == c_abbr and len(s1_abbr) >= 2) else 0.0

    # 2. Address representations
    addr_missing = 1.0 if (not c_ab or len(c_ab) == 0) else 0.0

    if not addr_missing:
        if s1_ab == c_ab:
            addr_ratio = addr_sort = addr_set = addr_jw = 1.0
        else:
            addr_ratio = fuzz.ratio(s1_ab, c_ab) / 100.0
            addr_sort = fuzz.token_sort_ratio(s1_ab, c_ab) / 100.0
            addr_set = fuzz.token_set_ratio(s1_ab, c_ab) / 100.0
            addr_jw = distance.JaroWinkler.similarity(s1_ab, c_ab)
        addr_basic_exact = 1.0 if s1_ab == c_ab else 0.0

        s1_atoks = set(s1_ab.split())
        c_atoks = set(c_ab.split())
        common_atoks = s1_atoks.intersection(c_atoks)
        union_atoks = s1_atoks.union(c_atoks)
        num_common_atoks = len(common_atoks)
        addr_token_jaccard = num_common_atoks / len(union_atoks) if union_atoks else 0.0

        s1_digs = set(s1_digs_str)
        c_digs = set(c_digs_str)
        common_digs = s1_digs.intersection(c_digs)
        union_digs = s1_digs.union(c_digs)
        addr_digits_common = float(len(common_digs))
        addr_digits_jaccard = len(common_digs) / len(union_digs) if union_digs else 0.0

        postal_code_exact = 1.0 if (s1_pcode and c_pcode and s1_pcode == c_pcode) else 0.0
        house_num_exact = 1.0 if (s1_hnum and c_hnum and s1_hnum == c_hnum) else 0.0

        addr_len_diff = float(abs(len(s1_ab) - len(c_ab)))
        addr_len_ratio = min(len(s1_ab), len(c_ab)) / max(len(s1_ab), len(c_ab)) if max(len(s1_ab), len(c_ab)) > 0 else 0.0
    else:
        addr_ratio = 0.0
        addr_sort = 0.0
        addr_set = 0.0
        addr_jw = 0.0
        addr_basic_exact = 0.0
        num_common_atoks = 0.0
        addr_token_jaccard = 0.0
        addr_digits_common = 0.0
        addr_digits_jaccard = 0.0
        postal_code_exact = 0.0
        house_num_exact = 0.0
        addr_len_diff = 0.0
        addr_len_ratio = 0.0

    # 3. Country feature
    country_exact = 1.0 if (s1_cnt and c_cnt and s1_cnt == c_cnt) else 0.0

    # 4. Cross-field interaction features
    name_sim_x_addr_sim = name_sort * addr_sort
    name_sim_x_country = name_sort * country_exact
    addr_sim_x_country = addr_sort * country_exact
    name_exact_and_addr_high = 1.0 if (name_nosuf_exact == 1.0 and addr_sort >= 0.8) else 0.0
    name_high_and_postal_match = 1.0 if (name_sort >= 0.8 and postal_code_exact == 1.0) else 0.0
    name_nosuf_exact_and_addr_sim = name_nosuf_exact * addr_sort

    # 5. Source indicator
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    return [
        name_ratio,
        name_partial,
        name_sort,
        name_set,
        name_wratio,
        name_jw,
        name_basic_exact,
        name_nosuf_exact,
        name_sorted_exact,
        first_token_exact,
        last_token_exact,
        name_token_jaccard,
        float(num_common_toks),
        float(name_len_diff),
        name_len_ratio,
        name_abbr_match,
        
        addr_ratio,
        addr_sort,
        addr_set,
        addr_jw,
        addr_basic_exact,
        addr_token_jaccard,
        float(num_common_atoks),
        addr_digits_common,
        addr_digits_jaccard,
        postal_code_exact,
        house_num_exact,
        addr_len_diff,
        addr_len_ratio,
        addr_missing,
        
        country_exact,
        
        name_sim_x_addr_sim,
        name_sim_x_country,
        addr_sim_x_country,
        name_exact_and_addr_high,
        name_high_and_postal_match,
        name_nosuf_exact_and_addr_sim,
        
        is_s2,
        is_s3
    ]

_G_S1_LOOKUP = None
_G_TARGET_LOOKUP = None

def _init_feature_worker(s1_lookup, target_lookup):
    global _G_S1_LOOKUP, _G_TARGET_LOOKUP
    _G_S1_LOOKUP = s1_lookup
    _G_TARGET_LOOKUP = target_lookup

def _batch_extract_pairwise_features_fast(pairs_chunk):
    return [
        extract_pairwise_features(_G_S1_LOOKUP[s1_id], _G_TARGET_LOOKUP[cand_id])
        for s1_id, cand_id in pairs_chunk
    ]

def extract_features_parallel(
    pairs: List[Tuple[str, str]],
    s1_lookup: Dict[str, Any],
    target_lookup: Dict[str, Any],
    batch_size: int = 50000,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Extracts pairwise features in parallel across multiple CPU cores without IPC pickling overhead.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not pairs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    max_workers = max(1, os.cpu_count() or 1) if n_jobs in (-1, None) else max(1, n_jobs)

    if len(pairs) < 5000 or max_workers == 1:
        return np.array([
            extract_pairwise_features(s1_lookup[s1_id], target_lookup[cand_id])
            for s1_id, cand_id in pairs
        ], dtype=np.float32)

    num_batches = (len(pairs) + batch_size - 1) // batch_size
    batches = [
        pairs[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches)
    ]

    results = [None] * num_batches
    with ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=_init_feature_worker, 
        initargs=(s1_lookup, target_lookup)
    ) as executor:
        future_to_idx = {
            executor.submit(_batch_extract_pairwise_features_fast, batch): idx 
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    flat_feats = [feat for batch in results for feat in batch]
    return np.array(flat_feats, dtype=np.float32)

