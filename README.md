# Equity Analytics

A modular **Streamlit** research dashboard for US equities — price action, technical indicators, fundamentals, news, portfolio simulation, and alternative data overlays.

Add `docs/screenshot.png` before publishing (capture from the Overview tab).

## Architecture

```
finance_dashboard.py            # Application entry point
finance_dashboard/
  config.py                     # Paths, env vars, tab labels
  models.py                     # Typed state objects (selection, market snapshot)
  theme.py                      # Inter font, chart template, layout CSS
  runner.py                     # Streamlit orchestration
  data/                         # External data access (cached)
    companies.py                # Universe loader (us_companies.csv)
    market.py                   # Yahoo Finance OHLCV
    news.py                     # NewsAPI headlines
    trends.py                   # Google Trends
    weather.py                  # Open-Meteo archive
    fundamentals.py             # Real-time Yahoo fundamentals
  analytics/                    # Pure calculation layer
    indicators.py               # RSI, MACD, signal logic
    market_frames.py            # yfinance MultiIndex normalization
  ui/
    sidebar.py                  # Universe + date controls
    tabs/                       # One module per dashboard tab
.streamlit/config.toml          # Theme and server defaults
```

Separation keeps I/O, analytics, and presentation independent — the same pattern used in production analytics apps where data pipelines and UI evolve on different schedules.

## Features

| Tab | Description |
|-----|-------------|
| Overview | Adjusted close trends, KPIs, cumulative return |
| Price & MAs | Moving averages, RSI, MACD, signal watchlist |
| Candlestick & Volume | OHLCV deep-dive on focus ticker |
| Returns & Volatility | Daily returns and summary statistics |
| Distribution | Price box plots and return histograms |
| Correlations | Return correlation heatmap |
| Pairwise Scatter | Cross-sectional price relationships |
| Fundamentals | Sector, industry, market cap (live) |
| News & Events | Headlines via NewsAPI |
| Portfolio Simulation | Weighted portfolio vs S&P 500 |
| Alternative Signals | Google Trends + Midwest weather archive |

## Prerequisites

- Python 3.10+
- [NewsAPI](https://newsapi.org/register) key (required)

## Data: `us_companies.csv`

Place `us_companies.csv` in the project root (bundled sample: ~7,000 US symbols). Required columns: `Symbol`, `Name`. Optional: `Sector`, `Industry`, `Market Cap`, `Country`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add NEWS_API_KEY
```

## Run

```bash
streamlit run finance_dashboard.py
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWS_API_KEY` | Yes | NewsAPI key ([free tier](https://newsapi.org/register)) |

Copy `.env.example` to `.env` locally. For [Streamlit Cloud](https://streamlit.io/cloud), add `NEWS_API_KEY` under app secrets — never commit `.env`.

## Roadmap

- SEC filing ingestion
- Strategy backtesting on indicator signals
- Persistent watchlists (Postgres/SQLite)
- Streamlit Cloud deployment with secrets

## Stack

Streamlit · yfinance · Plotly · NewsAPI · pytrends · Open-Meteo
