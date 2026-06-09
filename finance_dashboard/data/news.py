from datetime import datetime, timedelta

import streamlit as st
from newsapi import NewsApiClient

from finance_dashboard.config import NEWS_API_KEY

_news_client: NewsApiClient | None = None


def get_news_client() -> NewsApiClient:
    global _news_client
    if _news_client is None:
        if not NEWS_API_KEY:
            raise RuntimeError("NEWS_API_KEY is not configured")
        _news_client = NewsApiClient(api_key=NEWS_API_KEY)
    return _news_client


@st.cache_data(show_spinner="Fetching headlines…")
def fetch_company_news(query: str, days_back: int = 30, language: str = "en", page_size: int = 10):
    today = datetime.today()
    from_date = today - timedelta(days=days_back)

    try:
        articles = get_news_client().get_everything(
            q=query,
            from_param=from_date.strftime("%Y-%m-%d"),
            to=today.strftime("%Y-%m-%d"),
            language=language,
            sort_by="relevancy",
            page_size=page_size,
        )
        return articles.get("articles", [])
    except Exception as exc:
        st.error(f"News API error: {exc}")
        return []
