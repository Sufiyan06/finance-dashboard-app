import streamlit as st

from finance_dashboard.config import APP_TAGLINE, APP_TITLE, NEWS_API_KEY, TAB_LABELS
from finance_dashboard.data.companies import load_company_universe
from finance_dashboard.data.market import build_market_snapshot, load_market_data
from finance_dashboard.theme import inject_global_styles, render_page_header
from finance_dashboard.ui.sidebar import render_sidebar
from finance_dashboard.ui.tabs import TAB_RENDERERS


def run() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

    inject_global_styles()

    if not NEWS_API_KEY:
        st.error("NEWS_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")
        st.stop()

    company_to_ticker, ticker_to_company = load_company_universe()
    selection = render_sidebar(company_to_ticker, ticker_to_company)

    render_page_header(APP_TITLE, APP_TAGLINE)

    if not selection.tickers:
        st.info("Select one or more companies from the sidebar to begin.")
        st.stop()

    data_multi, close_df = load_market_data(selection.tickers, selection.start_date, selection.end_date)
    if data_multi is None or close_df is None or close_df.empty:
        st.warning("No market data returned for the selected universe and date range.")
        st.stop()

    market = build_market_snapshot(data_multi, close_df)
    tabs = st.tabs(TAB_LABELS)

    render_args = [
        (market,),
        (selection, market),
        (selection, market),
        (market,),
        (selection, market),
        (market,),
        (market,),
        (selection,),
        (selection,),
        (selection, market),
        (selection,),
    ]

    for tab, renderer, args in zip(tabs, TAB_RENDERERS, render_args):
        with tab:
            renderer(*args)
