TICKER = "AAPL"    # primary stock — used for validation, test set, and backtest
PERIOD = "10y"

# All stocks used for training. The model learns general market patterns
# from multiple stocks rather than overfitting to one ticker's behaviour.
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

ML_FEATURES = [
    "RSI_14",
    "MACD",
    "Signal_Line",
    "MACD_Histogram",
    "Daily_Return",
    "Volatility_20",
    "Price_vs_EMA_20",
    "Price_vs_SMA_20",
    "Return_5",
    "Volume_change",
]
