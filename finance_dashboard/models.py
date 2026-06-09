from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SidebarSelection:
    selected_companies: tuple[str, ...]
    tickers: tuple[str, ...]
    primary_ticker: Optional[str]
    start_date: date
    end_date: date
    company_to_ticker: dict[str, str]
    ticker_to_company: dict[str, str]


@dataclass(frozen=True)
class MarketSnapshot:
    data_multi: pd.DataFrame
    close_df: pd.DataFrame
    daily_returns: pd.DataFrame
    cum_returns: pd.DataFrame
    volatility_252: pd.Series
