import re
from rapidfuzz import fuzz

COMMON_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 
    'blvd', 'boulevard', 'way', 'rue', 'court', 'ct', 'place', 'pl', 'nouvelle', 
    'aquitaine', 'gironde', 'france', 'india', 'odisha', 'orissa', 'khurda', 
    'khordha', 'bhubaneswar', 'texas', 'tyler', 'tx', 'state', 'country', 'city', 
    'near', 'opp', 'opposite', 'behind', 'beside', 'floor', 'fl', 'suite', 'ste', 
    'apt', 'apartment', 'unit', 'bldg', 'building'
}

DISCRIMINATIVE_MODIFIERS = {
    'west', 'east', 'north', 'south', 'central', 'holdings', 'holding', 
    'exports', 'export', 'services', 'service', 'international', 'industries', 
    'logistics', 'properties', 'realty', 'finance', 'financial', 'motors'
}

def extract_primary_num(addr):
    if not addr or addr == 'nan':
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

def streets_compatible(a1, a2):
    if not a1 or not a2 or a1 == 'nan' or a2 == 'nan':
        return True
    toks1 = set(w for w in re.findall(r'[a-z]{4,}', str(a1).lower()) if w not in COMMON_STOPWORDS)
    toks2 = set(w for w in re.findall(r'[a-z]{4,}', str(a2).lower()) if w not in COMMON_STOPWORDS)
    if not toks1 or not toks2:
        return True
    return len(toks1.intersection(toks2)) >= 1

def name_modifiers_compatible(name1, name2):
    toks1 = set(re.findall(r'[a-z]+', str(name1).lower()))
    toks2 = set(re.findall(r'[a-z]+', str(name2).lower()))
    for mod in DISCRIMINATIVE_MODIFIERS:
        if mod in toks2 and mod not in toks1:
            return False
        if mod in toks1 and mod not in toks2:
            return False
    return True

def is_pair_valid(s1_name, s1_addr, cand_name, cand_addr):
    if not name_modifiers_compatible(s1_name, cand_name):
        return False
    n1 = extract_primary_num(s1_addr)
    n2 = extract_primary_num(cand_addr)
    if not nums_compatible(n1, n2):
        return False
    if not streets_compatible(s1_addr, cand_addr):
        return False
    return True
