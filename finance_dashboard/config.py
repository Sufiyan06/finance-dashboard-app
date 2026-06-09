import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
COMPANIES_CSV = ROOT_DIR / "us_companies.csv"
CACHE_DIR = ROOT_DIR / ".cache"

APP_TITLE = "Equity Analytics"
APP_TAGLINE = "Multi-asset research workspace for price, risk, fundamentals, and macro signals."


def get_news_api_key() -> str | None:
    """Local dev: .env via os.getenv. Streamlit Cloud: st.secrets."""
    key = os.getenv("NEWS_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("NEWS_API_KEY")
    except Exception:
        return None

WEATHER_ARCHIVE_MIN_YEAR = 1940
MIDWEST_LATITUDE = 41.8781
MIDWEST_LONGITUDE = -93.0977

TAB_LABELS = [
    "Overview",
    "Price & MAs",
    "Candlestick & Volume",
    "Returns & Volatility",
    "Distribution",
    "Correlations",
    "Pairwise Scatter",
    "Fundamentals",
    "News & Events",
    "Portfolio Simulation",
    "Alternative Signals",
]
