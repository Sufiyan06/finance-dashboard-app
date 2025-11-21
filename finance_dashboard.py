import streamlit as st
from newsapi import NewsApiClient
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pytrends.request import TrendReq
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
from openai import OpenAI


# ============= Page config =============
st.set_page_config(page_title="📊 Stock Analytics Pro", layout="wide")

# ============= Load company list (your CSV) =============
# Expected columns from your example: Symbol, Name, Sector, Industry, Market Cap (and others)
df_companies = pd.read_csv("us_companies.csv")
df_companies.columns = df_companies.columns.str.strip()
if "Name" in df_companies.columns:
    df_companies["Name"] = df_companies["Name"].astype(str).str.strip()
if "Symbol" in df_companies.columns:
    df_companies["Symbol"] = df_companies["Symbol"].astype(str).str.strip()

# Build mappings
company_to_ticker = dict(zip(df_companies.get("Name", pd.Series([], dtype=str)),
                             df_companies.get("Symbol", pd.Series([], dtype=str))))
ticker_to_company = {v: k for k, v in company_to_ticker.items()}

# Some optional metadata helpers
meta_cols = ["Symbol", "Name", "Sector", "Industry", "Market Cap", "Country"]
company_meta = df_companies[meta_cols].drop_duplicates() if set(meta_cols).issubset(df_companies.columns) else None

NEWS_API_KEY = "7d317d89b0e747b08834ec5ddde2cafd"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

@st.cache_data(show_spinner=True)
def fetch_trends(keyword, start_date, end_date):
    """
    Fetch Google Trends interest over time for a keyword
    """
    try:
        timeframe = f"{start_date} {end_date}"
        pytrends.build_payload([keyword], timeframe=timeframe, geo="")  # 🌍 Worldwide
        data = pytrends.interest_over_time()
        if not data.empty:
            return data[keyword]
    except Exception as e:
        st.warning(f"Could not fetch trends for {keyword}: {e}")
    return pd.Series()


# Setup cache & retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
client = openmeteo_requests.Client(session=retry_session)

def fetch_weather(latitude, longitude, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "shortwave_radiation_sum"],
        "timezone": "auto"
    }
    responses = client.weather_api(url, params=params)
    data = responses[0].Daily()

    df = pd.DataFrame({
        "date": pd.date_range(start=start_date, end=end_date, freq="D"),
        "temp_max": data.Variables(0).ValuesAsNumpy(),
        "temp_min": data.Variables(1).ValuesAsNumpy(),
        "precipitation": data.Variables(2).ValuesAsNumpy(),
        "wind": data.Variables(3).ValuesAsNumpy(),
        "solar_radiation": data.Variables(4).ValuesAsNumpy()
    })
    return df


@st.cache_data(show_spinner=True)
def fetch_company_news(query, days_back=30, language="en", page_size=10):
    from datetime import timedelta
    
    today = datetime.today()
    from_date = today - timedelta(days=days_back)

    try:
        articles = newsapi.get_everything(
            q=query,
            from_param=from_date.strftime("%Y-%m-%d"),
            to=today.strftime("%Y-%m-%d"),
            language=language,
            sort_by="relevancy",
            page_size=page_size
        )
        return articles.get("articles", [])
    except Exception as e:
        st.error(f"News API error: {e}")
        return []




# ============= Sidebar controls =============
st.sidebar.header("Dashboard Controls")

selected_companies = st.sidebar.multiselect(
    "Select companies (type to search)",
    options=sorted(company_to_ticker.keys())
)

tickers = [company_to_ticker[c] for c in selected_companies if c in company_to_ticker]

start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# If we have tickers, choose a primary ticker for candlestick/indicators
primary_ticker = None
if tickers:
    primary_ticker = st.sidebar.selectbox("Primary ticker for deep-dive", options=tickers, index=0) 

# ============= Helpers =============
def to_percent(x, digits=2):
    return f"{x:.{digits}f}%"

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def normalize_multiindex(df: pd.DataFrame, tickers_list):
    """
    Ensure multi-ticker download always yields a MultiIndex [Field, Ticker].
    For single ticker, convert single-level columns to MultiIndex.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Already MultiIndex: sort for consistency
        df = df.sort_index(axis=1)
        return df
    else:
        # Single ticker case: make MultiIndex with fields on level 0 and ticker on level 1
        new_cols = pd.MultiIndex.from_product([df.columns, tickers_list])
        df.columns = new_cols
        return df

def get_field_frame(data_multi: pd.DataFrame, field: str, tickers_list):
    """
    Returns a DataFrame indexed by date, columns = tickers, values = requested field.
    Works for both single and multi ticker downloads.
    """
    if isinstance(data_multi.columns, pd.MultiIndex):
        # Expect level 0 = field (e.g., 'Close'), level 1 = ticker
        # Some yfinance versions use the reverse order; handle both.
        if field in data_multi.columns.get_level_values(0):
            out = data_multi[field]
        else:
            # swap if needed
            swapped = data_multi.copy()
            swapped.columns = swapped.columns.swaplevel(0, 1)
            out = swapped[field]
        # Ensure all requested tickers are present as columns if they exist
        cols = [t for t in tickers_list if t in out.columns]
        out = out[cols] if cols else out
        if isinstance(out, pd.Series):  # single ticker degenerates to Series sometimes
            out = out.to_frame(cols[0] if cols else field)
        return out
    else:
        # Should not happen after normalize, but just in case
        return data_multi[[field]].rename(columns={field: tickers_list[0]})
    
def compute_rsi_alerts(series: pd.Series, period=14):
    rsi_series = rsi(series, period)
    overbought = rsi_series > 70
    oversold = rsi_series < 30
    return rsi_series, overbought, oversold

def compute_macd_alerts(series: pd.Series):
    macd_line, signal_line, _ = macd(series)
    # Bullish crossover: MACD crosses above signal
    bullish = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
    bearish = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)
    return macd_line, signal_line, bullish, bearish

# ============= Data loader =============
@st.cache_data(show_spinner=True)
def load_market_data(tickers_list, start, end):
    """
    Returns:
      - data_multi: MultiIndex columns [Field, Ticker]
      - close_df: Close prices wide (columns=tickers)
    """
    if not tickers_list:
        return None, None

    raw = yf.download(
        tickers_list,
        start=start,
        end=end,
        auto_adjust=False,     # keep raw OHLCV
        group_by="ticker",
        progress=False
    )

    data_multi = normalize_multiindex(raw, tickers_list)
    close_df = get_field_frame(data_multi, "Close", tickers_list).dropna(how="all")
    # align on dates with valid close data
    data_multi = data_multi.loc[close_df.index]
    return data_multi, close_df

# ============= Main =============
st.title("📈 Stock Analytics Pro — Multi-Feature Dashboard")
st.caption("Powered by Yahoo Finance (OHLCV) + derived indicators (MA/RSI/MACD) + comparisons.")

if not tickers:
    st.info("Start by selecting one or more companies from the sidebar.")
    st.stop()

data_multi, close_df = load_market_data(tickers, start_date, end_date)
if data_multi is None or close_df is None or close_df.empty:
    st.warning("No data returned for the chosen period/tickers.")
    st.stop()

# Housekeeping: drop all-null columns and rows
close_df = close_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

# Compute derived datasets
daily_returns = close_df.pct_change().dropna()
cum_returns = (1 + daily_returns).cumprod() - 1  # cumulative return from first valid day
volatility_252 = daily_returns.std() * np.sqrt(252)  # annualized volatility

# Tabs to keep UI tidy
tabs = st.tabs([
    "Overview", "Price & MAs", "Candlestick & Volume", "Returns & Volatility",
    "Distribution", "Correlations", "Pairwise Scatter", "Fundamentals", "News & Events", "Portfolio Simulation", "Alternative Signals"
])



# ================= Education Mode =================
st.sidebar.header("🎓 Education Mode")
education_mode = st.sidebar.checkbox("Enable Education Mode")

if education_mode:
    st.subheader("📘 Learn Technical Indicators")

    with st.expander("Relative Strength Index (RSI)"):
        st.write("RSI (Relative Strength Index) is a momentum indicator measuring the speed and change of price movements.")
        st.markdown("- RSI > 70 → Stock is usually considered **Overbought** (may pull back).")
        st.markdown("- RSI < 30 → Stock is usually considered **Oversold** (may bounce up).")

    with st.expander("Moving Average Convergence Divergence (MACD)"):
        st.write("MACD is a trend-following momentum indicator that shows the relationship between two moving averages.")
        st.markdown("- **MACD Line crosses above Signal Line → Bullish trend change.**")
        st.markdown("- **MACD Line crosses below Signal Line → Bearish trend change.**")

    # ---------------- Gamified Quiz ----------------
    st.subheader("🎮 Quiz Mode")
    quiz_question = st.radio(
        "Question: If RSI = 80, what does it usually mean?",
        ["Stock is oversold", "Stock is neutral", "Stock is overbought"]
    )

    if st.button("Check Answer"):
        if quiz_question == "Stock is overbought":
            st.success("✅ Correct! RSI > 70 usually means the stock is overbought.")
        else:
            st.error("❌ Not quite. Remember: RSI > 70 → Overbought, RSI < 30 → Oversold.")

# ================= Overview =================
with tabs[0]:
    st.subheader("Overview — Price Trends")
    fig_line = px.line(
        close_df,
        x=close_df.index,
        y=close_df.columns,
        title="Close Price (USD)",
        labels={"value": "Price (USD)", "variable": "Ticker"},
        render_mode="svg"
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    

    # KPIs
    st.markdown("#### Key Performance Metrics")
    cols = st.columns(min(5, len(close_df.columns)))
    for i, t in enumerate(close_df.columns):
        price_change = (close_df[t].iloc[-1] / close_df[t].iloc[0] - 1) * 100
        vol = daily_returns[t].std() * 100 if t in daily_returns.columns else np.nan
        with cols[i % len(cols)]:
            st.metric(label=f"{t} % Change", value=to_percent(price_change))
            st.metric(label=f"{t} Daily Volatility", value=to_percent(vol))

    # Cumulative return comparison
    st.markdown("#### Cumulative Return (from first selected date)")
    fig_cum = px.line(
    cum_returns,
    labels={"value": "Cumulative Return", "variable": "Ticker"},
    title="Cumulative Return",
    render_mode="svg"
)
    st.plotly_chart(fig_cum, use_container_width=True)

# ================= Price & MAs =================
with tabs[1]:
    st.subheader("Price with Moving Averages")
    if primary_ticker is None:
        st.info("Select a primary ticker in the sidebar.")
    else:
        series = close_df[primary_ticker].dropna()
        ma20 = series.rolling(20).mean()
        ma50 = series.rolling(50).mean()
        ma200 = series.rolling(200).mean()

        ma_df = pd.DataFrame({
            primary_ticker: series,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200
        })

        fig_ma = px.line(
            ma_df,
            x=ma_df.index,
            y=ma_df.columns,
            title=f"{primary_ticker}: Price & Moving Averages (20/50/200)",
            render_mode="svg"
        )
        st.plotly_chart(fig_ma, use_container_width=True)

        # RSI + MACD
        st.markdown("#### RSI(14) & MACD(12,26,9)")
        rsi14 = rsi(series, 14)
        macd_line, signal_line, macd_hist = macd(series)

        # RSI chart
        fig_rsi = px.line(
            rsi14.to_frame("RSI"),
            x=rsi14.index,
            y="RSI",
            title=f"{primary_ticker}: RSI(14)",
            render_mode="svg"
        )
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.add_hline(y=30, line_dash="dash")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD chart
        macd_df = pd.DataFrame({
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": macd_hist
        })
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=macd_df.index, y=macd_df["Signal"], name="Signal"))
        fig_macd.add_trace(go.Bar(x=macd_df.index, y=macd_df["Histogram"], name="Hist"))
        fig_macd.update_layout(title=f"{primary_ticker}: MACD(12,26,9)")
        st.plotly_chart(fig_macd, use_container_width=True)
        
                # ================= Smart Alerts =================
        st.markdown("#### ⚡ Smart Alerts / Watchlist Signals")

        if not tickers:
            st.info("Select companies in the sidebar to enable alerts.")
        else:
            st.markdown(
                "Automatically highlights overbought/oversold conditions and MACD crossovers "
                "for selected tickers."
            )

            selected_alerts = st.multiselect(
                "Select tickers to monitor for signals",
                options=tickers,
                default=tickers[:4]  # default: first 4
            )

            alert_rows = []
            for t in selected_alerts:
                series = close_df[t].dropna()

                rsi_series, overbought, oversold = compute_rsi_alerts(series)
                macd_line, signal_line, bullish, bearish = compute_macd_alerts(series)

                latest_date = series.index[-1]

                # Check latest signals
                rsi_signal = "Overbought ⚠️" if overbought.iloc[-1] else ("Oversold ✅" if oversold.iloc[-1] else "Neutral")
                macd_signal = "Bullish ↑" if bullish.iloc[-1] else ("Bearish ↓" if bearish.iloc[-1] else "Neutral")

                alert_rows.append({
                    "Ticker": t,
                    "RSI Signal": rsi_signal,
                    "MACD Signal": macd_signal,
                    "Last Close": series.iloc[-1],
                    "Date": latest_date.date()
                })

            if alert_rows:
                alert_df = pd.DataFrame(alert_rows)
                st.dataframe(alert_df)

                st.markdown("#### Visual Alerts (Recent 30 days)")
                for t in selected_alerts:
                    series = close_df[t].dropna().tail(30)
                    rsi_series, overbought, oversold = compute_rsi_alerts(series)
                    macd_line, signal_line, bullish, bearish = compute_macd_alerts(series)

                    fig_alert = go.Figure()
                    fig_alert.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Close'))
                    fig_alert.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, mode='lines', name='RSI', yaxis='y2'))

                    # Highlight RSI overbought/oversold points
                    fig_alert.add_trace(go.Scatter(
                        x=series.index[overbought], y=series[overbought], mode='markers', name='RSI>70', marker=dict(color='red', size=10)
                    ))
                    fig_alert.add_trace(go.Scatter(
                        x=series.index[oversold], y=series[oversold], mode='markers', name='RSI<30', marker=dict(color='green', size=10)
                    ))

                    # MACD crossovers
                    fig_alert.add_trace(go.Scatter(
                        x=series.index[bullish], y=series[bullish], mode='markers', name='MACD Bullish', marker=dict(color='blue', symbol='triangle-up', size=12)
                    ))
                    fig_alert.add_trace(go.Scatter(
                        x=series.index[bearish], y=series[bearish], mode='markers', name='MACD Bearish', marker=dict(color='orange', symbol='triangle-down', size=12)
                    ))

                    # Dual-axis for RSI
                    fig_alert.update_layout(
                        title=f"{t}: Close Price + RSI Alerts + MACD Crossovers",
                        yaxis=dict(title="Price"),
                        yaxis2=dict(title="RSI", overlaying='y', side='right'),
                        legend=dict(orientation="h")
                    )

                    st.plotly_chart(fig_alert, use_container_width=True)
            else:
                st.info("Select at least 1 ticker to show alerts.")
        



# ================= Candlestick & Volume =================
with tabs[2]:
    st.subheader("Candlestick & Volume")
    if primary_ticker is None:
        st.info("Select a primary ticker in the sidebar.")
    else:
        # Extract OHLCV for the primary ticker from the multiindex frame
        def fld(f):  # helper to get a series column safely
            try:
                # Standard case: first level is field
                return data_multi[(f, primary_ticker)]
            except KeyError:
                # Swaplevel case
                return data_multi[(primary_ticker, f)]

        open_s = fld("Open").dropna()
        high_s = fld("High").dropna()
        low_s = fld("Low").dropna()
        close_s = fld("Close").dropna()
        vol_s = fld("Volume").fillna(0)

        common_index = close_s.index
        fig_candle = go.Figure(
            data=[go.Candlestick(
                x=common_index,
                open=open_s.reindex(common_index),
                high=high_s.reindex(common_index),
                low=low_s.reindex(common_index),
                close=close_s.reindex(common_index),
                name=primary_ticker
            )]
        )
        fig_candle.update_layout(title=f"Candlestick: {primary_ticker}", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig_candle, use_container_width=True)

        st.markdown("#### Volume")
        fig_vol = px.bar(
            vol_s.to_frame("Volume").reindex(common_index),
            x=common_index,
            y="Volume",
            title=f"Volume: {primary_ticker}"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ================= Returns & Volatility =================
with tabs[3]:
    st.subheader("Daily Returns (%)")
    if not daily_returns.empty:
        fig_ret = px.line(
            daily_returns * 100,
            x=daily_returns.index,
            y=daily_returns.columns,
            labels={"value": "Daily Return (%)", "variable": "Ticker"},
            title="Daily Returns (%)",
            render_mode="svg"
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown("#### Summary Stats")
    stats = pd.DataFrame({
        "Total Return (%)": (close_df.iloc[-1] / close_df.iloc[0] - 1) * 100,
        "Annualized Volatility (%)": volatility_252 * 100
    }).round(2)
    st.dataframe(stats)

# ================= Distribution =================
with tabs[4]:
    st.subheader("Distribution of Prices & Returns")
    # Box plot of prices
    st.markdown("**Price Distribution (Box Plot)**")
    if len(close_df.columns) > 1:
        melt_prices = close_df.reset_index().melt(id_vars=["Date"], var_name="Ticker", value_name="Price")
        fig_box = px.box(melt_prices, x="Ticker", y="Price", color="Ticker", title="Price Distribution")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        fig_box = px.box(close_df, y=close_df.columns[0], title=f"{close_df.columns[0]} Price Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

    # Histogram of daily returns for primary ticker
    if primary_ticker and primary_ticker in daily_returns.columns:
        st.markdown(f"**Daily Return Histogram — {primary_ticker}**")
        fig_hist = px.histogram(
            daily_returns[primary_ticker] * 100,
            nbins=50,
            title=f"Histogram of Daily Returns (%) — {primary_ticker}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ================= Correlations =================
with tabs[5]:
    if len(close_df.columns) > 1:
        st.subheader("Correlation Heatmap (Close Returns)")
        corr = daily_returns.corr()
        fig_heat = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix (Daily Returns)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Select 2+ tickers to view correlations.")

# ================= Pairwise Scatter =================
with tabs[6]:
    if len(close_df.columns) > 1:
        st.subheader("Pairwise Scatter with Trendline (Close Prices)")

        # Limit to first 4 selected tickers
        tickers_list = close_df.columns[:4]
        
        if len(close_df.columns) > 4:
            st.info("More than 4 stocks selected — displaying scatter plots for the first 4 only.")
    
        pairs = [(i, j) for i in range(len(tickers_list)) for j in range(i+1, len(tickers_list))]

        n_pairs = len(pairs)
        n_cols = 2  # 2 charts per row
        n_rows = (n_pairs + n_cols - 1) // n_cols

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{tickers_list[i]} vs {tickers_list[j]}" for i,j in pairs]
        )

        for idx, (i,j) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            x_t, y_t = tickers_list[i], tickers_list[j]
            pair_df = close_df[[x_t, y_t]].dropna()

            fig.add_trace(
                go.Scatter(x=pair_df[x_t], y=pair_df[y_t], mode='markers', name=f"{x_t} vs {y_t}"),
                row=row, col=col
            )

        fig.update_layout(
            height=300*n_rows,
            showlegend=False,
            title_text="Pairwise Scatter (Close Prices)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select 2+ tickers to view pairwise scatter.")



# ================= Fundamentals (Real-Time) =================
with tabs[7]:
    st.subheader("Company Fundamentals (Real-Time)")

    if selected_companies:
        @st.cache_data(show_spinner=True)
        def fetch_realtime_fundamentals(tickers_list):
            """
            Fetch real-time company fundamentals from Yahoo Finance
            Returns a DataFrame with columns:
            Symbol | Name | Sector | Industry | Country | Market Cap
            """
            rows = []
            for t in tickers_list:
                try:
                    ticker_obj = yf.Ticker(t)
                    info = ticker_obj.info

                    rows.append({
                        "Symbol": t,
                        "Name": info.get("shortName", ticker_to_company.get(t, t)),
                        "Sector": info.get("sector", "N/A"),
                        "Industry": info.get("industry", "N/A"),
                        "Country": info.get("country", "N/A"),
                        "Market Cap (Real-Time)": info.get("marketCap", np.nan)
                    })
                except Exception as e:
                    # Graceful fallback
                    rows.append({
                        "Symbol": t,
                        "Name": ticker_to_company.get(t, t),
                        "Sector": "N/A",
                        "Industry": "N/A",
                        "Country": "N/A",
                        "Market Cap (Real-Time)": np.nan
                    })
            return pd.DataFrame(rows)

        fundamentals_df = fetch_realtime_fundamentals(tickers)

        # Keep display order same as selected tickers
        fundamentals_df["__order"] = fundamentals_df["Symbol"].apply(lambda x: tickers.index(x) if x in tickers else 9999)
        fundamentals_df = fundamentals_df.sort_values("__order").drop(columns="__order")

        # Format market cap in billions/millions for readability
        def format_market_cap(x):
            if pd.isna(x):
                return "N/A"
            elif x >= 1e12:
                return f"${x/1e12:.2f}T"
            elif x >= 1e9:
                return f"${x/1e9:.2f}B"
            elif x >= 1e6:
                return f"${x/1e6:.2f}M"
            else:
                return f"${x:.0f}"

        fundamentals_df["Market Cap (Real-Time)"] = fundamentals_df["Market Cap (Real-Time)"].apply(format_market_cap)

        st.dataframe(fundamentals_df)
    else:
        st.info("Select at least 1 company to view real-time fundamentals.")


# ================= News & Events =================
with tabs[8]:
    st.subheader("📰 Latest News & Events")
    if primary_ticker:
        company_name = ticker_to_company.get(primary_ticker, primary_ticker)
        st.markdown(f"Showing news for **{company_name} ({primary_ticker})**")
        
        news_days = st.sidebar.slider("News lookback (days)", 7, 30, 30)
        articles = fetch_company_news(company_name, days_back=news_days)

        if not articles:
            st.info("No recent news found for this company.")
        else:
            for art in articles:
                st.markdown(f"### [{art['title']}]({art['url']})")
                if art.get("urlToImage"):
                    st.image(art["urlToImage"], width=400)
                st.caption(f"**Source:** {art['source']['name']} | {art['publishedAt'][:10]}")
                st.write(art.get("description", ""))
                st.markdown("---")
    else:
        st.info("Select a primary ticker to fetch related news.")
        
        
# ================= Portfolio Simulation =================
with tabs[9]:
    st.subheader("📊 Portfolio Simulation")

    portfolio_symbols = st.multiselect(
        "Select stocks for your portfolio",
        options=tickers,
        default=tickers[:2] if len(tickers) >= 2 else tickers
    )

    if portfolio_symbols:
        # Fetch closing prices for selected portfolio symbols
        portfolio_data = close_df[portfolio_symbols]

        # Calculate daily returns
        port_daily_returns = portfolio_data.pct_change().dropna()

        # User can assign weights
        st.markdown("#### Assign Weights")
        weights = []
        for symbol in portfolio_symbols:
            w = st.number_input(
                f"Weight for {symbol}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(portfolio_symbols),
                step=0.05
            )
            weights.append(w)

        weights = np.array(weights)
        if weights.sum() != 1.0:
            weights = weights / weights.sum()  # normalize

        # Portfolio returns
        portfolio_returns = (port_daily_returns * weights).sum(axis=1)
        cum_portfolio = (1 + portfolio_returns).cumprod()

        # Benchmark: S&P 500 (^GSPC)
        benchmark = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True, progress=False)

        # Pick 'Adj Close' if exists, else 'Close'
        if "Adj Close" in benchmark.columns:
            benchmark_series = benchmark["Adj Close"].dropna()
        else:
            benchmark_series = benchmark["Close"].dropna()

        benchmark_returns = benchmark_series.pct_change().dropna()
        cum_benchmark = (1 + benchmark_returns).cumprod()

        # Align dates with portfolio
        common_index = cum_portfolio.index.intersection(cum_benchmark.index)
        cum_portfolio = cum_portfolio.loc[common_index]
        cum_benchmark = cum_benchmark.loc[common_index]

        # Ensure both are 1D Series for DataFrame
        cum_portfolio = cum_portfolio.squeeze() if hasattr(cum_portfolio, 'squeeze') else cum_portfolio
        cum_benchmark = cum_benchmark.squeeze() if hasattr(cum_benchmark, 'squeeze') else cum_benchmark

        # Plot cumulative returns
        results = pd.DataFrame({
            "Portfolio": cum_portfolio,
            "S&P 500": cum_benchmark
        })
        fig_port = px.line(results, title="Portfolio vs. S&P 500", render_mode="svg")
        st.plotly_chart(fig_port, use_container_width=True)

        # Show stats
        sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std() + 1e-12) * np.sqrt(252)
        st.markdown("#### Portfolio Stats")
        st.write(f"- Annualized Return: {portfolio_returns.mean() * 252:.2%}")
        st.write(f"- Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
        st.write(f"- Sharpe Ratio: {sharpe_ratio:.2f}")

    else:
        st.info("Select at least 1 stock to simulate a portfolio.")

# ================= Google Trends Tab =================
with tabs[10]:
    st.subheader("📈 Google Trends — Retail Interest")
    
    if not selected_companies:
        st.info("Select companies from the sidebar first.")
    else:
        trends_data = pd.DataFrame()

        for company in selected_companies:
            keyword = company.split()[0]  # just first word
            trend_series = fetch_trends(keyword, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if not trend_series.empty:
                trends_data[keyword] = trend_series

        if not trends_data.empty:
            st.line_chart(trends_data)
        else:
            st.info("No trend data available for the selected companies.")

    st.subheader("🌦️ Weather Impact — Agriculture & Energy Stocks")

    if not selected_companies:
        st.info("Select companies from the sidebar first.")
    else:
        # Example: US Midwest for agriculture
        weather_df = fetch_weather(41.8781, -93.0977, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        st.line_chart(weather_df.set_index("date")[["precipitation", "temp_max", "temp_min"]])
        st.line_chart(weather_df.set_index("date")[["solar_radiation", "wind"]])
        
        st.info("Overlay this data with agriculture/energy stock performance to analyze correlations.")



