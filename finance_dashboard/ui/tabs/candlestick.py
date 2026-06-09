import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from finance_dashboard.models import MarketSnapshot, SidebarSelection
from finance_dashboard.theme import apply_chart_defaults


def _ohlcv_series(data_multi, field: str, ticker: str):
    try:
        return data_multi[(field, ticker)]
    except KeyError:
        return data_multi[(ticker, field)]


def render_candlestick_tab(selection: SidebarSelection, market: MarketSnapshot) -> None:
    st.subheader("Price action")
    primary = selection.primary_ticker
    if primary is None:
        st.info("Select a focus ticker in the sidebar.")
        return

    open_s = _ohlcv_series(market.data_multi, "Open", primary).dropna()
    high_s = _ohlcv_series(market.data_multi, "High", primary).dropna()
    low_s = _ohlcv_series(market.data_multi, "Low", primary).dropna()
    close_s = _ohlcv_series(market.data_multi, "Close", primary).dropna()
    vol_s = _ohlcv_series(market.data_multi, "Volume", primary).fillna(0)

    index = close_s.index
    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=index,
                open=open_s.reindex(index),
                high=high_s.reindex(index),
                low=low_s.reindex(index),
                close=close_s.reindex(index),
                name=primary,
            )
        ]
    )
    fig_candle.update_layout(
        title=f"{primary} — OHLC",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(apply_chart_defaults(fig_candle), use_container_width=True)

    st.markdown("##### Volume")
    fig_vol = apply_chart_defaults(
        px.bar(
            vol_s.to_frame("Volume").reindex(index),
            x=index,
            y="Volume",
            title=f"{primary} — daily volume",
        )
    )
    st.plotly_chart(fig_vol, use_container_width=True)
