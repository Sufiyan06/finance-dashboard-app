import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finance_dashboard.analytics.indicators import compute_macd_alerts, compute_rsi_alerts, macd, rsi
from finance_dashboard.models import MarketSnapshot, SidebarSelection
from finance_dashboard.theme import apply_chart_defaults


def render_price_indicators_tab(selection: SidebarSelection, market: MarketSnapshot) -> None:
    st.subheader("Technical analysis")
    primary = selection.primary_ticker
    if primary is None:
        st.info("Select a focus ticker in the sidebar.")
        return

    close_df = market.close_df
    series = close_df[primary].dropna()
    ma_df = pd.DataFrame(
        {
            primary: series,
            "MA20": series.rolling(20).mean(),
            "MA50": series.rolling(50).mean(),
            "MA200": series.rolling(200).mean(),
        }
    )

    fig_ma = apply_chart_defaults(
        px.line(
            ma_df,
            x=ma_df.index,
            y=ma_df.columns,
            title=f"{primary} — price and moving averages (20 / 50 / 200)",
            render_mode="svg",
        )
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    st.markdown("##### Momentum indicators")
    rsi14 = rsi(series, 14)
    macd_line, signal_line, macd_hist = macd(series)

    fig_rsi = apply_chart_defaults(
        px.line(rsi14.to_frame("RSI"), x=rsi14.index, y="RSI", title=f"{primary} — RSI (14)", render_mode="svg")
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#94a3b8")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#94a3b8")
    st.plotly_chart(fig_rsi, use_container_width=True)

    macd_df = pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": macd_hist})
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=macd_df.index, y=macd_df["Signal"], name="Signal"))
    fig_macd.add_trace(go.Bar(x=macd_df.index, y=macd_df["Histogram"], name="Histogram", opacity=0.5))
    fig_macd.update_layout(title=f"{primary} — MACD (12, 26, 9)")
    st.plotly_chart(apply_chart_defaults(fig_macd), use_container_width=True)

    _render_signal_watchlist(selection, close_df)


def _render_signal_watchlist(selection: SidebarSelection, close_df: pd.DataFrame) -> None:
    st.markdown("##### Signal watchlist")
    st.caption("Latest RSI regime and MACD crossover state per ticker.")

    tickers = list(selection.tickers)
    if not tickers:
        st.info("Select companies in the sidebar to enable monitoring.")
        return

    monitored = st.multiselect("Tickers to monitor", options=tickers, default=tickers[:4])
    alert_rows = []
    for ticker in monitored:
        series = close_df[ticker].dropna()
        _, overbought, oversold = compute_rsi_alerts(series)
        _, _, bullish, bearish = compute_macd_alerts(series)

        if overbought.iloc[-1]:
            rsi_signal = "Overbought"
        elif oversold.iloc[-1]:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

        if bullish.iloc[-1]:
            macd_signal = "Bullish crossover"
        elif bearish.iloc[-1]:
            macd_signal = "Bearish crossover"
        else:
            macd_signal = "Neutral"

        alert_rows.append(
            {
                "Ticker": ticker,
                "RSI": rsi_signal,
                "MACD": macd_signal,
                "Last close": round(series.iloc[-1], 2),
                "As of": series.index[-1].date(),
            }
        )

    if not alert_rows:
        st.info("Select at least one ticker.")
        return

    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)

    st.markdown("##### Recent signal map (30 sessions)")
    for ticker in monitored:
        series = close_df[ticker].dropna().tail(30)
        rsi_series, overbought, oversold = compute_rsi_alerts(series)
        _, _, bullish, bearish = compute_macd_alerts(series)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, mode="lines", name="RSI", yaxis="y2"))
        fig.add_trace(
            go.Scatter(
                x=series.index[overbought],
                y=series[overbought],
                mode="markers",
                name="RSI > 70",
                marker=dict(color="#c0392b", size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=series.index[oversold],
                y=series[oversold],
                mode="markers",
                name="RSI < 30",
                marker=dict(color="#1e6f4f", size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=series.index[bullish],
                y=series[bullish],
                mode="markers",
                name="MACD bullish",
                marker=dict(color="#1e3a5f", symbol="triangle-up", size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=series.index[bearish],
                y=series[bearish],
                mode="markers",
                name="MACD bearish",
                marker=dict(color="#b45309", symbol="triangle-down", size=10),
            )
        )
        fig.update_layout(
            title=f"{ticker} — price with RSI and MACD markers",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(apply_chart_defaults(fig), use_container_width=True)
