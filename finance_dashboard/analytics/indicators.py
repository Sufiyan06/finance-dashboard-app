import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_rsi_alerts(series: pd.Series, period: int = 14):
    rsi_series = rsi(series, period)
    overbought = rsi_series > 70
    oversold = rsi_series < 30
    return rsi_series, overbought, oversold


def compute_macd_alerts(series: pd.Series):
    macd_line, signal_line, _ = macd(series)
    bullish = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
    bearish = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)
    return macd_line, signal_line, bullish, bearish
