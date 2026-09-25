import re
import unicodedata
import pandas as pd
from typing import List, Dict

LEGAL_SUFFIXES = {
    'ltd', 'limited', 'pvt', 'private', 'llc', 'inc', 'incorporated', 
    'corp', 'corporation', 'co', 'company', 'llp', 'plc', 'gmbh', 'sa', 
    'bv', 'srl', 'spa', 'services', 'solutions', 'enterprises', 'group', 
    'holdings', 'international', 'tech', 'technologies', 'industries',
    'pvt ltd', 'private limited', 'co ltd', 'co limited', 'inc corp'
}

# Regex to trim legal suffixes from the end of string or token list
SUFFIX_PATTERN = re.compile(
    r'\b(' + '|'.join(sorted(list(LEGAL_SUFFIXES), key=len, reverse=True)) + r')\b', 
    re.IGNORECASE
)

ADDRESS_ABBREVS = {
    r'\broad\b': 'rd',
    r'\bstreet\b': 'st',
    r'\bavenue\b': 'ave',
    r'\blane\b': 'ln',
    r'\bhighway\b': 'hwy',
    r'\bapartment\b': 'apt',
    r'\bbuilding\b': 'bldg',
    r'\bsuite\b': 'ste',
    r'\bfloor\b': 'fl',
    r'\bdrive\b': 'dr',
    r'\bboulevard\b': 'blvd',
    r'\bpost office\b': 'po',
    r'\bopposite\b': 'opp',
    r'\bnear\b': 'nr',
    r'\bsector\b': 'sec',
    r'\bphase\b': 'ph',
    r'\bblock\b': 'blk',
    r'\bnumber\b': 'no',
}

def unicode_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'&', ' and ', text)
    return text.strip()

def normalize_name_basic(name: str) -> str:
    cleaned = unicode_clean(name)
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def normalize_name_alnum(name: str) -> str:
    cleaned = unicode_clean(name)
    cleaned = re.sub(r'[^a-z0-9]', '', cleaned)
    return cleaned

def normalize_name_no_suffix(name: str) -> str:
    basic = normalize_name_basic(name)
    no_suf = SUFFIX_PATTERN.sub('', basic)
    no_suf = re.sub(r'\s+', ' ', no_suf).strip()
    return no_suf if no_suf else basic

def get_sorted_tokens(text: str) -> str:
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def get_abbreviation(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    return "".join(t[0] for t in tokens if t[0].isalnum())

def normalize_address_basic(address: str) -> str:
    cleaned = unicode_clean(address)
    for pattern, repl in ADDRESS_ABBREVS.items():
        cleaned = re.sub(pattern, repl, cleaned)
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_address_digits(address: str) -> str:
    if not isinstance(address, str):
        return ""
    return "".join(re.findall(r'\d+', address))

def extract_postal_code(address: str) -> str:
    """Extract potential 5 or 6 digit PIN / ZIP codes."""
    if not isinstance(address, str):
        return ""
    # Search for 6-digit (India) or 5-digit (US/France) codes
    match6 = re.findall(r'\b\d{6}\b', address)
    if match6:
        return match6[0]
    match5 = re.findall(r'\b\d{5}\b', address)
    if match5:
        return match5[0]
    return ""

def extract_house_number(address: str) -> str:
    """Extract leading or house number from address."""
    if not isinstance(address, str):
        return ""
    match = re.search(r'\b\d+[a-z]?\b', address.lower())
    if match:
        return match.group(0)
    return ""

def build_normalized_record_views(record: dict) -> dict:
    raw_name = record.get('business_name', '')
    raw_addr = record.get('business_address', '')
    raw_country = record.get('country', '')
    
    name_basic = normalize_name_basic(raw_name)
    name_alnum = normalize_name_alnum(raw_name)
    name_no_suf = normalize_name_no_suffix(raw_name)
    name_sorted = get_sorted_tokens(name_basic)
    name_no_suf_sorted = get_sorted_tokens(name_no_suf)
    name_abbr = get_abbreviation(name_no_suf)
    
    addr_basic = normalize_address_basic(raw_addr)
    addr_alnum = re.sub(r'[^a-z0-9]', '', addr_basic)
    addr_sorted = get_sorted_tokens(addr_basic)
    addr_digits = extract_address_digits(raw_addr)
    postal_code = extract_postal_code(raw_addr)
    house_num = extract_house_number(raw_addr)
    
    country_clean = unicode_clean(raw_country)
    
    return {
        'entity_id': record.get('entity_id', ''),
        'original_name': str(raw_name) if pd.notna(raw_name) else '',
        'name_basic': name_basic,
        'name_alnum': name_alnum,
        'name_no_suf': name_no_suf,
        'name_sorted': name_sorted,
        'name_no_suf_sorted': name_no_suf_sorted,
        'name_abbr': name_abbr,
        
        'original_address': str(raw_addr) if pd.notna(raw_addr) else '',
        'addr_basic': addr_basic,
        'addr_alnum': addr_alnum,
        'addr_sorted': addr_sorted,
        'addr_digits': addr_digits,
        'postal_code': postal_code,
        'house_num': house_num,
        
        'country': country_clean
    }

RE_AMP = re.compile(r'&')
RE_NON_ALNUM_SPACE = re.compile(r'[^a-z0-9\s]')
RE_NON_ALNUM = re.compile(r'[^a-z0-9]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_POSTAL = re.compile(r'\b\d{5,6}\b')
RE_HOUSE = re.compile(r'\b\d+[a-z]?\b')
RE_DIGITS = re.compile(r'\d+')

class Record:
    __slots__ = (
        'entity_id', 'original_name', 'name_basic', 'name_alnum', 'name_no_suf',
        'name_sorted', 'name_no_suf_sorted', 'name_abbr', 'original_address',
        'addr_basic', 'addr_alnum', 'addr_sorted', 'addr_digits', 'postal_code',
        'house_num', 'country'
    )
    def __init__(
        self, entity_id, original_name, name_basic, name_alnum, name_no_suf,
        name_sorted, name_no_suf_sorted, name_abbr, original_address,
        addr_basic, addr_alnum, addr_sorted, addr_digits, postal_code,
        house_num, country
    ):
        self.entity_id = entity_id
        self.original_name = original_name
        self.name_basic = name_basic
        self.name_alnum = name_alnum
        self.name_no_suf = name_no_suf
        self.name_sorted = name_sorted
        self.name_no_suf_sorted = name_no_suf_sorted
        self.name_abbr = name_abbr
        self.original_address = original_address
        self.addr_basic = addr_basic
        self.addr_alnum = addr_alnum
        self.addr_sorted = addr_sorted
        self.addr_digits = addr_digits
        self.postal_code = postal_code
        self.house_num = house_num
        self.country = country

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=""):
        return getattr(self, item, default)

    def __contains__(self, item):
        return hasattr(self, item)

def normalize_dataframe_fast(df: pd.DataFrame, chunk_size: int = 250000) -> List[Record]:
    """
    Ultra-fast, memory-safe normalization immune to pandas type inference bugs.
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    records = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        sub_df = df.iloc[start:end]
        
        eids = sub_df['entity_id'].fillna('').astype(str).tolist() if 'entity_id' in sub_df.columns else [''] * len(sub_df)
        names = sub_df['business_name'].fillna('').astype(str).tolist() if 'business_name' in sub_df.columns else [''] * len(sub_df)
        addrs = sub_df['business_address'].fillna('').astype(str).tolist() if 'business_address' in sub_df.columns else [''] * len(sub_df)
        countries = sub_df['country'].fillna('').astype(str).tolist() if 'country' in sub_df.columns else [''] * len(sub_df)
        
        chunk_len = len(sub_df)
        for i in range(chunk_len):
            raw_name = names[i]
            raw_addr = addrs[i]
            raw_country = countries[i]
            
            name_lower = raw_name.lower()
            name_amp = RE_AMP.sub(' and ', name_lower)
            name_basic = RE_MULTI_SPACE.sub(' ', RE_NON_ALNUM_SPACE.sub(' ', name_amp)).strip()
            name_alnum = RE_NON_ALNUM.sub('', name_lower)
            
            no_suf = SUFFIX_PATTERN.sub('', name_basic)
            name_no_suf = RE_MULTI_SPACE.sub(' ', no_suf).strip()
            if not name_no_suf:
                name_no_suf = name_basic
                
            name_basic_toks = name_basic.split()
            name_no_suf_toks = name_no_suf.split()
            name_sorted = " ".join(sorted(name_basic_toks))
            name_no_suf_sorted = " ".join(sorted(name_no_suf_toks))
            name_abbr = "".join(t[0] for t in name_no_suf_toks if t[0].isalnum())
            
            addr_lower = raw_addr.lower()
            addr_basic = RE_MULTI_SPACE.sub(' ', RE_NON_ALNUM_SPACE.sub(' ', addr_lower)).strip()
            addr_alnum = RE_NON_ALNUM.sub('', addr_lower)
            addr_sorted = " ".join(sorted(addr_basic.split()))
            
            addr_digits = "".join(RE_DIGITS.findall(raw_addr))
            m_post = RE_POSTAL.search(raw_addr)
            postal_code = m_post.group(0) if m_post else ""
            
            m_house = RE_HOUSE.search(addr_lower)
            house_num = m_house.group(0) if m_house else ""
            
            country_clean = raw_country.lower().strip()
            
            records.append(Record(
                entity_id=eids[i],
                original_name=raw_name,
                name_basic=name_basic,
                name_alnum=name_alnum,
                name_no_suf=name_no_suf,
                name_sorted=name_sorted,
                name_no_suf_sorted=name_no_suf_sorted,
                name_abbr=name_abbr,
                original_address=raw_addr,
                addr_basic=addr_basic,
                addr_alnum=addr_alnum,
                addr_sorted=addr_sorted,
                addr_digits=addr_digits,
                postal_code=postal_code,
                house_num=house_num,
                country=country_clean
            ))
            
    return records


