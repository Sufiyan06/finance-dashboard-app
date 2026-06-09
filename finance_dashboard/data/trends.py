import pandas as pd
import streamlit as st
from pytrends.request import TrendReq

_pytrends: TrendReq | None = None


def _get_pytrends() -> TrendReq:
    global _pytrends
    if _pytrends is None:
        _pytrends = TrendReq(hl="en-US", tz=360)
    return _pytrends


@st.cache_data(show_spinner="Loading Google Trends…")
def fetch_trends(keyword: str, start_date: str, end_date: str) -> pd.Series:
    try:
        timeframe = f"{start_date} {end_date}"
        client = _get_pytrends()
        client.build_payload([keyword], timeframe=timeframe, geo="")
        data = client.interest_over_time()
        if not data.empty:
            return data[keyword]
    except Exception as exc:
        st.warning(f"Could not fetch trends for {keyword}: {exc}")
    return pd.Series(dtype=float)
