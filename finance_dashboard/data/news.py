import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import streamlit as st
from newsapi import NewsApiClient

from finance_dashboard.config import get_news_api_key

NewsStatus = Literal["ok", "empty", "rate_limited", "invalid_key", "no_key", "error"]


@dataclass(frozen=True)
class NewsResult:
    articles: list
    status: NewsStatus
    message: str | None = None


def _cache_scope(api_key: str | None) -> str:
    if not api_key:
        return "shared"
    return "user:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("ratelimited", "rate limit", "429", "too many requests", "developer accounts")
    )


def _is_invalid_key(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("apikeyinvalid", "invalid api key", "401", "unauthorized"))


def _resolve_api_key(explicit_key: str | None) -> str | None:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    return get_news_api_key()


@st.cache_data(show_spinner="Fetching headlines…")
def fetch_company_news(
    query: str,
    days_back: int,
    cache_scope: str,
    language: str = "en",
    page_size: int = 10,
    api_key: str | None = None,
) -> NewsResult:
    key = _resolve_api_key(api_key)
    if not key:
        return NewsResult([], "no_key", "No NewsAPI key configured.")

    today = datetime.today()
    from_date = today - timedelta(days=days_back)

    try:
        client = NewsApiClient(api_key=key)
        response = client.get_everything(
            q=query,
            from_param=from_date.strftime("%Y-%m-%d"),
            to=today.strftime("%Y-%m-%d"),
            language=language,
            sort_by="relevancy",
            page_size=page_size,
        )
        articles = response.get("articles", [])
        if not articles:
            return NewsResult([], "empty")
        return NewsResult(articles, "ok")
    except Exception as exc:
        if _is_rate_limited(exc):
            return NewsResult([], "rate_limited", "Daily request limit reached.")
        if _is_invalid_key(exc):
            return NewsResult([], "invalid_key", "Invalid API key.")
        return NewsResult([], "error", str(exc))


def fetch_news_for_company(query: str, days_back: int, user_api_key: str | None = None) -> NewsResult:
    key = user_api_key or get_news_api_key()
    scope = _cache_scope(key)
    return fetch_company_news(query, days_back, scope, api_key=key)
