import pandas as pd

# ============================================================
# Technical indicators
# ============================================================
# 1.  SMA  — Simple Moving Average
# 2.  EMA  — Exponential Moving Average
# 3.  Daily Return
# 4.  RSI  — Relative Strength Index  (Wilder EWM — corrected)
# 5.  MACD — Moving Average Convergence Divergence + Signal + Histogram
# 6.  Rolling Volatility
# 7.  Bollinger Bands
# 8.  Volume Change
# 9.  Price vs EMA  (relative, stationary feature)
# 10. Price vs SMA  (relative, stationary feature)
# 11. N-day Return
# ============================================================


def moving_averageSMA(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average of Close price."""
    data[f"SMA_{window}"] = data['Close'].rolling(window=window).mean()
    return data


def moving_averageEMA(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """Exponential moving average of Close price."""
    data[f"EMA_{window}"] = data['Close'].ewm(span=window, adjust=False).mean()
    return data


def daily_return(data: pd.DataFrame) -> pd.DataFrame:
    """Percentage daily return (expressed as %)."""
    data["Daily_Return"] = data['Close'].pct_change() * 100
    return data


def rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    The standard RSI (Welles Wilder) uses EWM with com = window-1,
    not a simple rolling mean.  Using a simple rolling mean produces
    slightly wrong values compared to most charting platforms.
    """
    delta = data['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing: alpha = 1 / window  →  com = window - 1
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    data[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return data


def macd(data: pd.DataFrame) -> pd.DataFrame:
    """
    MACD line, Signal line, and Histogram.

    MACD      = EMA(12) − EMA(26)
    Signal    = EMA(9) of MACD
    Histogram = MACD − Signal  (shows momentum shifts)
    """
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()

    data["MACD"]          = ema_12 - ema_26
    data["Signal_Line"]   = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Histogram"] = data["MACD"] - data["Signal_Line"]
    return data


def rolling_volatility(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling standard deviation of daily returns (proxy for volatility)."""
    data[f"Volatility_{window}"] = data["Daily_Return"].rolling(window=window).std()
    return data


def bollinger(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Bollinger Bands: middle, upper (+2σ) and lower (-2σ) bands."""
    middle = data['Close'].rolling(window=window).mean()
    std    = data['Close'].rolling(window=window).std()

    data[f"Middle_band_{window}"] = middle
    data[f"Upper_band_{window}"]  = middle + 2 * std
    data[f"Lower_band_{window}"]  = middle - 2 * std
    return data


def volume_change(data: pd.DataFrame) -> pd.DataFrame:
    """Percentage change in trading volume."""
    data["Volume_change"] = data['Volume'].pct_change()
    return data


def price_vs_ema(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Close / EMA — a stationary, relative feature.

    Values above 1 mean price is above its EMA (bullish momentum).
    Using this instead of the raw EMA avoids non-stationarity problems
    (the raw EMA is proportional to the absolute price level).
    """
    data[f"Price_vs_EMA_{window}"] = data['Close'] / data[f'EMA_{window}']
    return data


def price_vs_sma(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Close / SMA — same idea as price_vs_ema but for the simple average.
    """
    data[f"Price_vs_SMA_{window}"] = data['Close'] / data[f'SMA_{window}']
    return data


def return_n(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """N-day percentage return."""
    data[f"Return_{window}"] = data['Close'].pct_change(window)
    return data


# ------------------------------------------------------------------
# Convenience wrapper — adds all indicators at once
# ------------------------------------------------------------------

def add_all_ind(data: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach all technical indicators to the DataFrame."""

    # Moving averages (needed before price_vs_* calls)
    data = moving_averageSMA(data, 20)
    data = moving_averageSMA(data, 100)
    data = moving_averageEMA(data, 20)
    data = moving_averageEMA(data, 50)
    data = moving_averageEMA(data, 100)

    # Returns & momentum
    data = daily_return(data)
    data = return_n(data, 5)

    # Oscillators
    data = rsi(data, 14)
    data = macd(data)

    # Volatility & bands
    data = rolling_volatility(data, 20)
    data = bollinger(data, 20)

    # Volume
    data = volume_change(data)

    # Relative / stationary price features
    data = price_vs_ema(data, 20)
    data = price_vs_ema(data, 50)
    data = price_vs_sma(data, 20)

    return data
