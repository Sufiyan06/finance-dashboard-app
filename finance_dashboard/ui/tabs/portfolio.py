import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from finance_dashboard.analytics.market_frames import extract_close_series
from finance_dashboard.models import MarketSnapshot, SidebarSelection
from finance_dashboard.theme import apply_chart_defaults


def render_portfolio_tab(selection: SidebarSelection, market: MarketSnapshot) -> None:
    st.subheader("Portfolio simulation")
    tickers = list(selection.tickers)
    portfolio_symbols = st.multiselect(
        "Holdings",
        options=tickers,
        default=tickers[:2] if len(tickers) >= 2 else tickers,
    )

    if not portfolio_symbols:
        st.info("Select at least one holding.")
        return

    portfolio_data = market.close_df[portfolio_symbols]
    port_daily_returns = portfolio_data.pct_change().dropna()

    st.markdown("##### Allocation")
    weights = []
    for symbol in portfolio_symbols:
        weight = st.number_input(
            f"{symbol} weight",
            min_value=0.0,
            max_value=1.0,
            value=1.0 / len(portfolio_symbols),
            step=0.05,
        )
        weights.append(weight)

    weights = np.array(weights)
    if weights.sum() != 1.0:
        weights = weights / weights.sum()

    portfolio_returns = (port_daily_returns * weights).sum(axis=1)
    cum_portfolio = (1 + portfolio_returns).cumprod()

    benchmark = yf.download("^GSPC", start=selection.start_date, end=selection.end_date, auto_adjust=True, progress=False)
    benchmark_returns = extract_close_series(benchmark).pct_change().dropna()
    cum_benchmark = (1 + benchmark_returns).cumprod()

    common_index = cum_portfolio.index.intersection(cum_benchmark.index)
    results = pd.DataFrame(
        {
            "Portfolio": cum_portfolio.loc[common_index].squeeze(),
            "S&P 500": cum_benchmark.loc[common_index].squeeze(),
        }
    )
    fig_port = apply_chart_defaults(
        px.line(results, title="Portfolio vs S&P 500 (growth of $1)", render_mode="svg")
    )
    st.plotly_chart(fig_port, use_container_width=True)

    sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-12) * np.sqrt(252)
    col1, col2, col3 = st.columns(3)
    col1.metric("Annualized return", f"{portfolio_returns.mean() * 252:.2%}")
    col2.metric("Annualized volatility", f"{portfolio_returns.std() * np.sqrt(252):.2%}")
    col3.metric("Sharpe ratio", f"{sharpe:.2f}")
