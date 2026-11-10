"""Real-World Threat Intelligence Layer: Redirects and Domain Age."""
import requests
from urllib.parse import urlparse
from datetime import datetime
import tldextract
import socket

SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly"]

def resolve_redirects(url: str, timeout=3) -> str:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        return response.url
    except Exception:
        return url

def get_domain_age(domain: str) -> int:
    """Check domain age using RDAP (modern WHOIS replacement)."""
    try:
        parts = domain.split('.')
        if len(parts) > 2:
            root_domain = '.'.join(parts[-2:])
        else:
            root_domain = domain
            
        if root_domain.replace('.', '').isdigit():
            return -1
            
        # Use RDAP instead of old WHOIS
        rdap_url = f"https://rdap.org/domain/{root_domain}"
        response = requests.get(rdap_url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            # Find the registration event
            for event in data.get("events", []):
                if event.get("eventAction") == "registration":
                    # Date format is typically 2021-05-18T04:00:00Z
                    date_str = event.get("eventDate").split("T")[0]
                    creation_date = datetime.strptime(date_str, "%Y-%m-%d")
                    age = (datetime.now() - creation_date).days
                    return age
        return -1
    except Exception as e:
        print(f"RDAP Lookup failed for {domain}: {e}")
        return -1

def analyze_real_world(url: str) -> dict:
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    domain = parsed.netloc.split(":")[0].lower()
    
    final_url = url
    is_shortener = any(s in domain for s in SHORTENERS)
    if is_shortener:
        final_url = resolve_redirects(url)
    
    domain_age = get_domain_age(urlparse(final_url).netloc)
    
    ext = tldextract.extract(final_url)
    tld = ext.suffix.lower()
    root_domain = ext.domain.lower()
    
    # FIXED: Removed the dots from the TLD list
    SAFE_TLDS = ["com", "org", "net", "edu", "gov", "ai", "io", "dev", "app", "co", "in", "me", "tech"]
    
    is_safe_tld = tld in SAFE_TLDS
    has_weird_chars = any(c in domain for c in ["0x", "xn--", "\u200b"])
    
    from phishing_detector.features.build import get_min_brand_distance, BRANDS
    brand_dist = get_min_brand_distance(root_domain)
    
    risk_score = 0.0
    reasons = []
    
    # WHOIS rules
    if domain_age != -1 and domain_age < 30:
        risk_score += 1.0
        reasons.append(f"New domain ({domain_age} days old)")
    elif domain_age > 365:
        risk_score -= 0.5
        reasons.append(f"Established domain")
        
    # Structural rules
    if is_safe_tld and not has_weird_chars:
        # Check if it's a typosquatted brand (e.g., paypa1) vs a safe word
        if brand_dist <= 1 and root_domain not in BRANDS:
            risk_score += 0.8
            reasons.append("Typosquatting brand detected")
        else:
            risk_score -= 0.5
            reasons.append("Standard safe TLD")
    elif not is_safe_tld:
        risk_score += 0.5
        reasons.append(f"Uncommon TLD ({tld})")
        
    return {
        "final_url": final_url,
        "domain_age_days": domain_age,
        "is_shortener": is_shortener,
        "risk_score_adjustment": risk_score,
        "reasons": reasons
    }