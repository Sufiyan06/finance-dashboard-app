import plotly.express as px
import streamlit as st

from finance_dashboard.models import MarketSnapshot
from finance_dashboard.theme import apply_chart_defaults


def render_correlations_tab(market: MarketSnapshot) -> None:
    if len(market.close_df.columns) <= 1:
        st.info("Select at least two tickers to view return correlations.")
        return

    st.subheader("Correlation matrix")
    corr = market.daily_returns.corr()
    fig_heat = apply_chart_defaults(
        px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Daily return correlations",
            aspect="auto",
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)
