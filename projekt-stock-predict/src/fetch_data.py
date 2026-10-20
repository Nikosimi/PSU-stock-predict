import yfinance as yf
import pandas as pd


def fetch_data(ticker, period='10y', interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker   : str   — stock ticker symbol, e.g. 'AAPL'
    period   : str   — lookback window, e.g. '1y', '5y', '10y'
    interval : str   — bar size, e.g. '1d', '1wk'

    Returns
    -------
    pd.DataFrame with OHLCV columns, or None on failure.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval,
                           progress=False, auto_adjust=True)

        if data.empty:
            print(f"[ERROR] No data returned for '{ticker}'. "
                  "Check the ticker symbol or your internet connection.")
            return None

        # yfinance sometimes returns a MultiIndex when downloading a single
        # ticker — flatten it so we always get plain column names.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        print(f"[INFO] Fetched {len(data)} rows for '{ticker}' "
              f"(period={period}, interval={interval})")
        return data

    except Exception as e:
        print(f"[ERROR] Failed to fetch data for '{ticker}': {e}")
        return None
