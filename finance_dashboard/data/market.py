import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from finance_dashboard.analytics.market_frames import get_field_frame, normalize_multiindex
from finance_dashboard.models import MarketSnapshot


@st.cache_data(show_spinner="Loading market data…")
def load_market_data(tickers_list: tuple[str, ...], start, end):
    if not tickers_list:
        return None, None

    raw = yf.download(
        list(tickers_list),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    data_multi = normalize_multiindex(raw, list(tickers_list))
    close_df = get_field_frame(data_multi, "Close", list(tickers_list)).dropna(how="all")
    data_multi = data_multi.loc[close_df.index]
    return data_multi, close_df


def build_market_snapshot(data_multi: pd.DataFrame, close_df: pd.DataFrame) -> MarketSnapshot:
    close_df = close_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    daily_returns = close_df.pct_change().dropna()
    cum_returns = (1 + daily_returns).cumprod() - 1
    volatility_252 = daily_returns.std() * np.sqrt(252)

    return MarketSnapshot(
        data_multi=data_multi,
        close_df=close_df,
        daily_returns=daily_returns,
        cum_returns=cum_returns,
        volatility_252=volatility_252,
    )
