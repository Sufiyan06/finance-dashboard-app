import pandas as pd


def normalize_multiindex(df: pd.DataFrame, tickers_list: list[str]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        return df.sort_index(axis=1)

    new_cols = pd.MultiIndex.from_product([df.columns, tickers_list])
    df = df.copy()
    df.columns = new_cols
    return df


def get_field_frame(data_multi: pd.DataFrame, field: str, tickers_list: list[str]) -> pd.DataFrame:
    if isinstance(data_multi.columns, pd.MultiIndex):
        if field in data_multi.columns.get_level_values(0):
            out = data_multi[field]
        else:
            swapped = data_multi.copy()
            swapped.columns = swapped.columns.swaplevel(0, 1)
            out = swapped[field]

        cols = [t for t in tickers_list if t in out.columns]
        out = out[cols] if cols else out
        if isinstance(out, pd.Series):
            out = out.to_frame(cols[0] if cols else field)
        return out

    return data_multi[[field]].rename(columns={field: tickers_list[0]})


def extract_close_series(frame: pd.DataFrame) -> pd.Series:
    """Return a 1D close-price series from a yfinance download (handles MultiIndex columns)."""
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = frame.columns.get_level_values(0)
        if "Adj Close" in level0:
            prices = frame["Adj Close"]
        elif "Close" in level0:
            prices = frame["Close"]
        else:
            prices = frame.iloc[:, 0]
    elif "Adj Close" in frame.columns:
        prices = frame["Adj Close"]
    else:
        prices = frame["Close"]

    series = prices.squeeze()
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.dropna()
