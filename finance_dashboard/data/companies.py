from pathlib import Path

import pandas as pd

from finance_dashboard.config import COMPANIES_CSV

def load_company_universe(csv_path: Path | None = None) -> tuple[dict[str, str], dict[str, str]]:
    path = csv_path or COMPANIES_CSV
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.strip()

    company_to_ticker = dict(
        zip(df.get("Name", pd.Series([], dtype=str)), df.get("Symbol", pd.Series([], dtype=str)))
    )
    ticker_to_company = {symbol: name for name, symbol in company_to_ticker.items()}
    return company_to_ticker, ticker_to_company
