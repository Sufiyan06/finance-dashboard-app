from datetime import datetime

import streamlit as st

from finance_dashboard.models import SidebarSelection


def render_sidebar(company_to_ticker: dict[str, str], ticker_to_company: dict[str, str]) -> SidebarSelection:
    st.sidebar.header("Universe")
    selected_companies = st.sidebar.multiselect(
        "Companies",
        options=sorted(company_to_ticker.keys()),
        placeholder="Search by company name",
    )

    tickers = tuple(company_to_ticker[c] for c in selected_companies if c in company_to_ticker)
    start_date = st.sidebar.date_input("Start date", datetime(2018, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.today())

    primary_ticker = None
    if tickers:
        primary_ticker = st.sidebar.selectbox("Focus ticker", options=list(tickers), index=0)

    _render_education_panel()

    return SidebarSelection(
        selected_companies=tuple(selected_companies),
        tickers=tickers,
        primary_ticker=primary_ticker,
        start_date=start_date,
        end_date=end_date,
        company_to_ticker=company_to_ticker,
        ticker_to_company=ticker_to_company,
    )


def _render_education_panel() -> None:
    with st.sidebar.expander("Indicator reference"):
        st.markdown("**RSI (14)**")
        st.caption("Above 70: overbought · Below 30: oversold")

        st.markdown("**MACD (12, 26, 9)**")
        st.caption("Line crossing above signal: bullish · Below signal: bearish")

        st.markdown("**Knowledge check**")
        quiz_answer = st.radio(
            "RSI at 80 typically indicates:",
            ["Oversold", "Neutral", "Overbought"],
            key="education_quiz",
            label_visibility="collapsed",
        )
        if st.button("Submit", key="education_quiz_btn", use_container_width=True):
            if quiz_answer == "Overbought":
                st.success("Correct — RSI above 70 is commonly treated as overbought.")
            else:
                st.error("RSI above 70 is overbought; below 30 is oversold.")
