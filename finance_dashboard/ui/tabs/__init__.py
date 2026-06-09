from finance_dashboard.ui.tabs.alternative_signals import render_alternative_signals_tab
from finance_dashboard.ui.tabs.candlestick import render_candlestick_tab
from finance_dashboard.ui.tabs.correlations import render_correlations_tab
from finance_dashboard.ui.tabs.distribution import render_distribution_tab
from finance_dashboard.ui.tabs.fundamentals import render_fundamentals_tab
from finance_dashboard.ui.tabs.news import render_news_tab
from finance_dashboard.ui.tabs.overview import render_overview_tab
from finance_dashboard.ui.tabs.pairwise import render_pairwise_tab
from finance_dashboard.ui.tabs.portfolio import render_portfolio_tab
from finance_dashboard.ui.tabs.price_indicators import render_price_indicators_tab
from finance_dashboard.ui.tabs.returns import render_returns_tab

TAB_RENDERERS = [
    render_overview_tab,
    render_price_indicators_tab,
    render_candlestick_tab,
    render_returns_tab,
    render_distribution_tab,
    render_correlations_tab,
    render_pairwise_tab,
    render_fundamentals_tab,
    render_news_tab,
    render_portfolio_tab,
    render_alternative_signals_tab,
]
