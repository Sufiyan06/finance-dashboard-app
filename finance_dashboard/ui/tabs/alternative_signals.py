from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from finance_dashboard.config import MIDWEST_LATITUDE, MIDWEST_LONGITUDE
from finance_dashboard.data.trends import fetch_trends
from finance_dashboard.data.weather import fetch_weather
from finance_dashboard.models import SidebarSelection


def render_alternative_signals_tab(selection: SidebarSelection) -> None:
    st.subheader("Alternative data")
    if not selection.selected_companies:
        st.info("Select companies from the sidebar.")
        return

    _render_trends(selection)
    st.divider()
    _render_weather(selection)


def _render_trends(selection: SidebarSelection) -> None:
    st.markdown("##### Google Trends — search interest")
    trends_data = pd.DataFrame()

    for company in selection.selected_companies:
        ticker = selection.company_to_ticker.get(company, company)
        series = fetch_trends(company, selection.start_date.strftime("%Y-%m-%d"), selection.end_date.strftime("%Y-%m-%d"))
        if not series.empty:
            trends_data[ticker] = series

    if trends_data.empty:
        st.info("No trend data available for the selected companies.")
    else:
        st.line_chart(trends_data)


def _render_weather(selection: SidebarSelection) -> None:
    st.markdown("##### Weather archive — US Midwest proxy")
    st.caption("Historical temperature, precipitation, wind, and solar radiation for ag/energy context.")

    archive_max = datetime.today().date() - timedelta(days=1)
    if selection.end_date > archive_max:
        st.caption(f"Archive data is available through {archive_max}.")

    weather_df = fetch_weather(
        MIDWEST_LATITUDE,
        MIDWEST_LONGITUDE,
        selection.start_date.strftime("%Y-%m-%d"),
        selection.end_date.strftime("%Y-%m-%d"),
    )

    if weather_df.empty:
        st.info("No weather data available for the selected date range.")
        return

    indexed = weather_df.set_index("date")
    st.line_chart(indexed[["precipitation", "temp_max", "temp_min"]])
    st.line_chart(indexed[["solar_radiation", "wind"]])
    st.caption("Compare these macro inputs against commodity-linked or weather-sensitive holdings in the price tabs.")
