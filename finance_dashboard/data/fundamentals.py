import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner="Fetching fundamentals…")
def fetch_realtime_fundamentals(tickers_list: tuple[str, ...], ticker_name_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for symbol in tickers_list:
        try:
            info = yf.Ticker(symbol).info
            rows.append(
                {
                    "Symbol": symbol,
                    "Name": info.get("shortName", ticker_name_map.get(symbol, symbol)),
                    "Sector": info.get("sector", "N/A"),
                    "Industry": info.get("industry", "N/A"),
                    "Country": info.get("country", "N/A"),
                    "Market Cap (Real-Time)": info.get("marketCap", np.nan),
                }
            )
        except Exception:
            rows.append(
                {
                    "Symbol": symbol,
                    "Name": ticker_name_map.get(symbol, symbol),
                    "Sector": "N/A",
                    "Industry": "N/A",
                    "Country": "N/A",
                    "Market Cap (Real-Time)": np.nan,
                }
            )
    return pd.DataFrame(rows)
