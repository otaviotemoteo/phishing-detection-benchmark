"""
URL feature engineering for classical ML (Planejamento §4.4).

Turns a raw URL string into a small set of numeric, interpretable features so the
classical models can run on the raw-URL datasets (Mendeley). Deep Learning models
(Phase 4) skip this entirely and consume the raw character sequence instead.

Kept deliberately simple and lexical (no network calls): every feature is derived
from the URL string itself, so extraction is fast and fully reproducible.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import pandas as pd


def _host(url: str) -> str:
    """Best-effort hostname from a URL string (strips credentials and port)."""
    try:
        netloc = urlparse(url if "://" in url else "http://" + url).netloc
    except ValueError:
        return ""
    return netloc.split("@")[-1].split(":")[0]


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_url_features(urls: pd.Series) -> pd.DataFrame:
    """Extract numeric lexical features from raw URLs.

    Args:
        urls: Series of URL strings.

    Returns:
        A DataFrame (aligned to ``urls.index``) of numeric feature columns:
        lengths, structural counts, and binary character/scheme flags.
    """
    u = urls.fillna("").astype(str)
    host = u.map(_host)

    feats = pd.DataFrame(index=u.index)
    feats["url_length"] = u.str.len()
    feats["domain_length"] = host.str.len()
    feats["n_dots"] = u.str.count(r"\.")
    feats["n_subdomains"] = host.str.count(r"\.")
    feats["path_depth"] = u.str.count("/")
    feats["n_query_params"] = u.str.count(r"[?&=]")
    feats["n_digits"] = u.str.count(r"\d")
    feats["n_special"] = u.str.count(r"[^A-Za-z0-9]")
    feats["has_at"] = u.str.contains("@").astype(int)
    feats["has_hyphen"] = u.str.contains("-").astype(int)
    feats["has_underscore"] = u.str.contains("_").astype(int)
    feats["has_https"] = u.str.lower().str.startswith("https").astype(int)
    feats["has_ip"] = host.map(_is_ip).astype(int)
    return feats
