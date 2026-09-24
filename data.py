"""Market data fetching.

Primary: yfinance, hardened with a browser-impersonating TLS session
(curl_cffi) plus retries - this fixes most 429 'Too Many Requests' errors,
which are extremely common when yfinance runs on shared cloud-host IPs
(Render, Heroku, Railway all report this constantly: Yahoo Finance rate
limits/blocks datacenter IP ranges much harder than home IPs).

Fallback / recommended-for-Render: Twelve Data (https://twelvedata.com),
a real forex/stock data API with a free tier (no credit card). If
TWELVEDATA_API_KEY is set, it is used automatically when yfinance fails,
or as the primary source if DATA_PROVIDER=twelvedata. It does not suffer
from the cloud-IP blocking problem since it's a proper API, not scraping.
"""
import time

import pandas as pd
import requests

REQUIRED = ["open", "high", "low", "close"]


def fetch(symbol="GBPJPY=X", period="730d", interval="1d",
          provider="auto", twelvedata_key=None):
    """Returns a DataFrame indexed by date with open/high/low/close/volume.
    Raises RuntimeError with a clear, user-facing message on failure.
    """
    errors = []

    if provider in ("twelvedata",):
        return _fetch_twelvedata(symbol, twelvedata_key, period, interval)

    if provider in ("auto", "yfinance"):
        try:
            return _fetch_yfinance(symbol, period, interval)
        except Exception as e:  # noqa: BLE001
            errors.append(f"yfinance: {e}")
            if provider == "yfinance" or not twelvedata_key:
                raise RuntimeError(
                    "Data fetch failed. " + " | ".join(errors) +
                    (". Yahoo Finance frequently rate-limits cloud hosts "
                     "(Render/Heroku) - set TWELVEDATA_API_KEY (free, no "
                     "card, twelvedata.com) to use a reliable fallback."
                     if not twelvedata_key else "")
                ) from e

    # auto mode fell through to the fallback
    try:
        return _fetch_twelvedata(symbol, twelvedata_key, period, interval)
    except Exception as e:  # noqa: BLE001
        errors.append(f"twelvedata: {e}")
        raise RuntimeError("Data fetch failed. " + " | ".join(errors)) from e


def _clean(df, interval):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[[c for c in REQUIRED if c in df.columns] + ["volume"]].dropna(subset=REQUIRED)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # The NNFX method trades on CLOSED bars only: drop an in-progress
    # current-day/week/month bar if present.
    if interval in ("1d", "1wk", "1mo") and len(df):
        today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        last = df.index[-1]
        last = last.tz_localize(None) if last.tz is not None else last
        if last >= today:
            df = df.iloc[:-1]
    return df


def _fetch_yfinance(symbol, period, interval, attempts=3):
    import yfinance as yf
    session = None
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
    except Exception:  # noqa: BLE001 - curl_cffi optional, fall back to plain session
        session = None

    last_err = None
    for i in range(attempts):
        try:
            kwargs = dict(period=period, interval=interval, progress=False, auto_adjust=True)
            if session is not None:
                kwargs["session"] = session
            df = yf.download(symbol, **kwargs)
            if df is None or df.empty:
                raise RuntimeError("empty response (symbol may be wrong, or Yahoo blocked this request)")
            return _clean(df, interval)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(str(last_err))


def _to_twelvedata_symbol(symbol):
    s = symbol.upper()
    if s.endswith("=X") and len(s) == 8:          # e.g. GBPJPY=X -> GBP/JPY
        base = s[:-2]
        return base[:3] + "/" + base[3:]
    if s.endswith("-USD") or "-" in s:            # e.g. BTC-USD -> BTC/USD
        return s.replace("-", "/")
    return s                                       # stocks/ETFs: SPY, AAPL, ...


def _fetch_twelvedata(symbol, api_key, period, interval):
    if not api_key:
        raise RuntimeError("no TWELVEDATA_API_KEY set")
    td_symbol = _to_twelvedata_symbol(symbol)
    td_interval = {"1d": "1day", "1wk": "1week", "1mo": "1month"}.get(interval, "1day")
    try:
        days = int(str(period).rstrip("d"))
        outputsize = min(5000, max(150, days))
    except ValueError:
        outputsize = 800

    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": td_symbol, "interval": td_interval, "outputsize": outputsize,
        "apikey": api_key, "order": "ASC",
    }, timeout=20)
    payload = r.json()
    if payload.get("status") == "error" or "values" not in payload:
        raise RuntimeError(payload.get("message", f"unexpected response for {td_symbol}"))

    rows = payload["values"]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _clean(df, interval)
