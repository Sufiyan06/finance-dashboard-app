import numpy as np
import plotly.express as px
import streamlit as st

from finance_dashboard.models import MarketSnapshot
from finance_dashboard.theme import apply_chart_defaults
from finance_dashboard.utils.formatting import to_percent


def render_overview_tab(market: MarketSnapshot) -> None:
    st.subheader("Market overview")
    close_df = market.close_df

    fig_line = apply_chart_defaults(
        px.line(
            close_df,
            x=close_df.index,
            y=close_df.columns,
            title="Adjusted close (USD)",
            labels={"value": "Price (USD)", "variable": "Ticker"},
            render_mode="svg",
        )
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("##### Performance summary")
    cols = st.columns(min(5, len(close_df.columns)))
    for index, ticker in enumerate(close_df.columns):
        price_change = (close_df[ticker].iloc[-1] / close_df[ticker].iloc[0] - 1) * 100
        vol = market.daily_returns[ticker].std() * 100 if ticker in market.daily_returns.columns else np.nan
        with cols[index % len(cols)]:
            st.metric(label=f"{ticker} total return", value=to_percent(price_change))
            st.metric(label=f"{ticker} daily vol.", value=to_percent(vol))

    st.markdown("##### Cumulative return")
    fig_cum = apply_chart_defaults(
        px.line(
            market.cum_returns,
            labels={"value": "Cumulative return", "variable": "Ticker"},
            title="Cumulative return since start date",
            render_mode="svg",
        )
    )
    st.plotly_chart(fig_cum, use_container_width=True)
