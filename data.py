"""Market data via yfinance (free, no API key). Supports forex (EURUSD=X),
crypto (BTC-USD), stocks/ETFs (AAPL, SPY) and more."""
import pandas as pd
import yfinance as yf

REQUIRED = ["open", "high", "low", "close"]


def fetch(symbol="EURUSD=X", period="730d", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    df = df[[c for c in REQUIRED if c in df.columns] + ["volume"]].dropna(subset=REQUIRED)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # remove duplicate index entries (yfinance quirks)
    df = df[~df.index.duplicated(keep="last")]

    # The NNFX method trades on CLOSED daily bars only, so drop the
    # in-progress current-day bar when it exists.
    if interval in ("1d", "1wk", "1mo") and len(df):
        today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        last = df.index[-1]
        last = last.tz_localize(None) if last.tz is not None else last
        if last >= today:
            df = df.iloc[:-1]
    return df
