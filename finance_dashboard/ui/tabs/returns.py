import plotly.express as px
import pandas as pd
import streamlit as st

from finance_dashboard.models import MarketSnapshot
from finance_dashboard.theme import apply_chart_defaults


def render_returns_tab(market: MarketSnapshot) -> None:
    st.subheader("Return profile")
    daily_returns = market.daily_returns

    if not daily_returns.empty:
        fig_ret = apply_chart_defaults(
            px.line(
                daily_returns * 100,
                x=daily_returns.index,
                y=daily_returns.columns,
                labels={"value": "Daily return (%)", "variable": "Ticker"},
                title="Daily returns",
                render_mode="svg",
            )
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown("##### Summary statistics")
    stats = pd.DataFrame(
        {
            "Total return (%)": (market.close_df.iloc[-1] / market.close_df.iloc[0] - 1) * 100,
            "Annualized volatility (%)": market.volatility_252 * 100,
        }
    ).round(2)
    st.dataframe(stats, use_container_width=True)
