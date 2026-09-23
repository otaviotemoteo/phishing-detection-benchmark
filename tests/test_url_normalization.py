"""
URL scheme normalization invariants (D-010).

`normalize_url_scheme` is the control that separates the cross-dataset finding
from a formatting artifact: Mendeley URLs carry ``http(s)://`` almost always and
the Kaggle corpus rarely, so without stripping it the lexical features encode a
collection habit. Because those features feed the published numbers, these tests
pin the function's behaviour exactly as it is today rather than as it might
ideally be. One deliberate gap is recorded below.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.feature_engineering import normalize_url_scheme


def norm(*urls: str) -> list[str]:
    """Normalize a handful of URLs and return them as a plain list."""
    return list(normalize_url_scheme(pd.Series(list(urls))))


def test_leading_http_and_https_are_removed():
    assert norm("http://example.com") == ["example.com"]
    assert norm("https://example.com") == ["example.com"]


def test_a_scheme_inside_the_url_is_left_alone():
    # Only a leading scheme is a formatting difference between corpora. A scheme
    # inside a redirect parameter is part of the URL's content and removing it
    # would change what the lexical features measure.
    assert norm("site.com/redirect?url=http://evil.example") == [
        "site.com/redirect?url=http://evil.example"
    ]
    assert norm("shop.example/httpsecure/login") == ["shop.example/httpsecure/login"]


def test_only_one_leading_scheme_is_removed_per_url():
    # The anchored pattern matches once, so a doubled scheme keeps its second half.
    assert norm("http://http://doubled.example") == ["http://doubled.example"]


def test_other_schemes_are_preserved():
    assert norm("ftp://files.example") == ["ftp://files.example"]


def test_normalization_is_idempotent():
    urls = [
        "http://example.com",
        "https://example.com/path?q=1",
        "example.com",
        "site.com/redirect?url=http://evil.example",
    ]
    once = normalize_url_scheme(pd.Series(urls))
    twice = normalize_url_scheme(once)

    assert list(once) == list(twice)


def test_urls_without_a_scheme_are_unchanged():
    assert norm("example.com/login", "") == ["example.com/login", ""]


def test_uppercase_schemes_are_not_stripped():
    """Recorded behaviour, not endorsed behaviour.

    The pattern is case sensitive, so ``HTTP://`` survives normalization while
    ``http://`` does not. Changing that would change the lexical features and
    therefore the published metrics, so the behaviour is pinned here and the gap
    is reported instead. If a future run relaxes the pattern, this test fails on
    purpose, and the numbers must be regenerated together with it.
    """
    assert norm("HTTP://example.com") == ["HTTP://example.com"]
    assert norm("HttPs://example.com") == ["HttPs://example.com"]


def test_index_and_length_are_preserved():
    series = pd.Series(
        ["http://a.example", "b.example", "https://c.example"], index=[10, 20, 30]
    )
    result = normalize_url_scheme(series)

    assert list(result.index) == [10, 20, 30]
    assert len(result) == len(series)


def test_the_runner_uses_this_exact_function():
    """The cross-dataset runner must not carry its own copy of the rule."""
    pytest.importorskip("torch", reason="the cross-dataset runner imports PyTorch")
    from src.experiments import runner_cross

    assert runner_cross._normalize_urls is normalize_url_scheme
