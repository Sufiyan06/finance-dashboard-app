import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from finance_dashboard.models import MarketSnapshot
from finance_dashboard.theme import apply_chart_defaults


def render_pairwise_tab(market: MarketSnapshot) -> None:
    close_df = market.close_df
    if len(close_df.columns) <= 1:
        st.info("Select at least two tickers to view pairwise relationships.")
        return

    st.subheader("Pairwise price relationships")
    tickers = list(close_df.columns[:4])
    if len(close_df.columns) > 4:
        st.caption("Showing the first four selected tickers.")

    pairs = [(i, j) for i in range(len(tickers)) for j in range(i + 1, len(tickers))]
    n_cols = 2
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{tickers[i]} vs {tickers[j]}" for i, j in pairs],
    )

    for idx, (i, j) in enumerate(pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        x_ticker, y_ticker = tickers[i], tickers[j]
        pair_df = close_df[[x_ticker, y_ticker]].dropna()
        fig.add_trace(
            go.Scatter(x=pair_df[x_ticker], y=pair_df[y_ticker], mode="markers", name=f"{x_ticker} vs {y_ticker}"),
            row=row,
            col=col,
        )

    fig.update_layout(height=300 * n_rows, showlegend=False, title_text="Close price scatter matrix")
    st.plotly_chart(apply_chart_defaults(fig), use_container_width=True)
