import pandas as pd


def to_percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def format_market_cap(value) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:.0f}"
