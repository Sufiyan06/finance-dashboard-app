import streamlit as st

from finance_dashboard.data.fundamentals import fetch_realtime_fundamentals
from finance_dashboard.models import SidebarSelection
from finance_dashboard.utils.formatting import format_market_cap


def render_fundamentals_tab(selection: SidebarSelection) -> None:
    st.subheader("Fundamentals snapshot")
    if not selection.selected_companies:
        st.info("Select at least one company.")
        return

    tickers = list(selection.tickers)
    fundamentals_df = fetch_realtime_fundamentals(tuple(tickers), selection.ticker_to_company)
    fundamentals_df["__order"] = fundamentals_df["Symbol"].apply(lambda symbol: tickers.index(symbol) if symbol in tickers else 9999)
    fundamentals_df = fundamentals_df.sort_values("__order").drop(columns="__order")
    fundamentals_df["Market Cap (Real-Time)"] = fundamentals_df["Market Cap (Real-Time)"].apply(format_market_cap)

    st.dataframe(fundamentals_df, use_container_width=True, hide_index=True)
