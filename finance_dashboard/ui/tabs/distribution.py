import plotly.express as px
import streamlit as st

from finance_dashboard.models import MarketSnapshot, SidebarSelection
from finance_dashboard.theme import apply_chart_defaults


def render_distribution_tab(selection: SidebarSelection, market: MarketSnapshot) -> None:
    st.subheader("Distribution analysis")
    close_df = market.close_df

    st.markdown("**Price distribution**")
    if len(close_df.columns) > 1:
        reset = close_df.reset_index()
        date_col = reset.columns[0]
        melted = reset.melt(id_vars=[date_col], var_name="Ticker", value_name="Price")
        fig_box = apply_chart_defaults(px.box(melted, x="Ticker", y="Price", color="Ticker", title="Close price distribution"))
    else:
        fig_box = apply_chart_defaults(px.box(close_df, y=close_df.columns[0], title=f"{close_df.columns[0]} price distribution"))
    st.plotly_chart(fig_box, use_container_width=True)

    primary = selection.primary_ticker
    if primary and primary in market.daily_returns.columns:
        st.markdown(f"**Daily return distribution — {primary}**")
        fig_hist = apply_chart_defaults(
            px.histogram(
                market.daily_returns[primary] * 100,
                nbins=50,
                title=f"{primary} — daily return histogram (%)",
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)
