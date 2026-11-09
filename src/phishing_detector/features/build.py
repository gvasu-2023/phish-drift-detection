"""Extract features from raw URLs including advanced adversarial indicators."""
import math
import re
from urllib.parse import urlparse
import tldextract
import unicodedata
import warnings

warnings.filterwarnings("ignore")

extractor = tldextract.TLDExtract(suffix_list_urls=())

IP_REGEX = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
OBFUSCATED_IP_REGEX = re.compile(r"^(?:0x[0-9a-fA-F]+|[0-7]+)(?:\.(?:0x[0-9a-fA-F]+|[0-7]+)){3}$")
ZWC_CHARS = ["\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"]
BRANDS = ["google", "paypal", "amazon", "microsoft", "apple", "netflix", "facebook"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_min_brand_distance(domain):
    if not domain:
        return 50.0
    domain = domain.lower()
    min_dist = 100.0
    for brand in BRANDS:
        parts = domain.split('.')
        base = parts[0] if parts else domain
        base = base.split('-')[0] # Handle combosquatting
        dist = levenshtein(base, brand)
        if dist < min_dist:
            min_dist = dist
    return float(min_dist)

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    entropy = 0.0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        entropy += - p_x * math.log2(p_x)
    return entropy

def extract_features(url: str) -> dict:
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        ext = extractor(url)
        
        dom = ext.domain + "." + ext.suffix if ext.suffix else ext.domain
        tld = ext.suffix
        subdom = ext.subdomain
        
        url_len = len(url)
        dom_len = len(dom)
        tld_len = len(tld)
        subdom_cnt = len(subdom.split(".")) if subdom else 0
        
        letter_cnt = sum(c.isalpha() for c in url)
        digit_cnt = sum(c.isdigit() for c in url)
        special_cnt = url_len - letter_cnt - digit_cnt
        
        eq_cnt = url.count("=")
        qm_cnt = url.count("?")
        amp_cnt = url.count("&")
        dot_cnt = url.count(".")
        dash_cnt = url.count("-")
        under_cnt = url.count("_")
        
        is_https = 1 if parsed.scheme == "https" else 0
        slash_cnt = url.count("/")
        
        path_len = len(parsed.path)
        query_len = len(parsed.query)
        
        is_ip = 1 if IP_REGEX.match(parsed.netloc.split(":")[0]) else 0
        
        # --- NEW ADVANCED FEATURES ---
        contains_non_ascii = 1 if any(ord(c) > 127 for c in dom) else 0
        
        scripts = set()
        for c in dom:
            if c.isalpha():
                try:
                    script = unicodedata.name(c).split(' ')[0]
                    scripts.add(script)
                except ValueError:
                    pass
        script_mixing_count = len(scripts)
        
        contains_zwc = 1 if any(zwc in url for zwc in ZWC_CHARS) else 0
        is_shortener = 1 if dom.lower() in SHORTENERS else 0
        is_obfuscated_ip = 1 if OBFUSCATED_IP_REGEX.match(parsed.netloc.split(":")[0]) else 0
        min_brand_dist = get_min_brand_distance(dom)
        
        return {
            "url_len": url_len, "dom_len": dom_len, "is_ip": is_ip, "tld": tld, "tld_len": tld_len,
            "subdom_cnt": subdom_cnt, "letter_cnt": letter_cnt, "digit_cnt": digit_cnt, "special_cnt": special_cnt,
            "eq_cnt": eq_cnt, "qm_cnt": qm_cnt, "amp_cnt": amp_cnt, "dot_cnt": dot_cnt, "dash_cnt": dash_cnt,
            "under_cnt": under_cnt, "letter_ratio": letter_cnt / url_len if url_len else 0.0,
            "digit_ratio": digit_cnt / url_len if url_len else 0.0, "spec_ratio": special_cnt / url_len if url_len else 0.0,
            "is_https": is_https, "slash_cnt": slash_cnt, "entropy": calculate_entropy(url),
            "path_len": path_len, "query_len": query_len, "contains_non_ascii": contains_non_ascii,
            "script_mixing_count": script_mixing_count, "contains_zwc": contains_zwc,
            "is_shortener": is_shortener, "is_obfuscated_ip": is_obfuscated_ip, "min_brand_dist": min_brand_dist
        }
    except Exception:
        return {col: 0 for col in [
            "url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt", "special_cnt", 
            "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt", "letter_ratio", "digit_ratio", 
            "spec_ratio", "is_https", "slash_cnt", "entropy", "path_len", "query_len", "contains_non_ascii", 
            "script_mixing_count", "contains_zwc", "is_shortener", "is_obfuscated_ip", "min_brand_dist"
        ]}