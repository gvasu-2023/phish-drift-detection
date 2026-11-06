"""Extract the 22 lexical/structural features from raw URLs."""
import math
import re
from urllib.parse import urlparse
import tldextract

# Use local snapshot to avoid network calls during extraction
extractor = tldextract.TLDExtract(suffix_list_urls=())

IP_REGEX = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")


def calculate_entropy(text: str) -> float:
    """Shannon entropy of a string."""
    if not text:
        return 0.0
    entropy = 0.0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        entropy += - p_x * math.log2(p_x)
    return entropy


def extract_features(url: str) -> dict:
    """Extract features matching the URL-Phish dataset schema."""
    try:
        parsed = urlparse(url)
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
        
        return {
            "url_len": url_len,
            "dom_len": dom_len,
            "is_ip": is_ip,
            "tld": tld,
            "tld_len": tld_len,
            "subdom_cnt": subdom_cnt,
            "letter_cnt": letter_cnt,
            "digit_cnt": digit_cnt,
            "special_cnt": special_cnt,
            "eq_cnt": eq_cnt,
            "qm_cnt": qm_cnt,
            "amp_cnt": amp_cnt,
            "dot_cnt": dot_cnt,
            "dash_cnt": dash_cnt,
            "under_cnt": under_cnt,
            "letter_ratio": letter_cnt / url_len if url_len else 0.0,
            "digit_ratio": digit_cnt / url_len if url_len else 0.0,
            "spec_ratio": special_cnt / url_len if url_len else 0.0,
            "is_https": is_https,
            "slash_cnt": slash_cnt,
            "entropy": calculate_entropy(url),
            "path_len": path_len,
            "query_len": query_len
        }
    except Exception:
        # Fallback for malformed URLs
        return {col: 0 for col in [
            "url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", 
            "letter_cnt", "digit_cnt", "special_cnt", "eq_cnt", "qm_cnt", 
            "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt", "letter_ratio", 
            "digit_ratio", "spec_ratio", "is_https", "slash_cnt", "entropy", 
            "path_len", "query_len"
        ]}