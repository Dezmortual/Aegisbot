"""Central configuration - everything overridable via environment variables."""
import os


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _b(name, default="false"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


SYMBOL = os.environ.get("SYMBOL", "GBPJPY=X")
INTERVAL = os.environ.get("INTERVAL", "1d")            # NNFX = daily charts
PERIOD = os.environ.get("PERIOD", "730d")              # history length
CAPITAL = _f("CAPITAL", 10000)
RISK_PCT = _f("RISK_PCT", 1.0)                          # % equity risked per trade
SL_ATR_MULT = _f("SL_ATR_MULT", 1.5)                    # stop distance = ATR * mult
TP_ATR_MULT = _f("TP_ATR_MULT", 4.0)                    # target distance = ATR * mult
FEE_BPS = _f("FEE_BPS", 1.0)                            # round-trip cost per side, basis points
MAX_LEVERAGE = _f("MAX_LEVERAGE", 5.0)
REFRESH_HOURS = _f("REFRESH_HOURS", 6)
CHART_BARS = int(_f("CHART_BARS", 260))

MODE = os.environ.get("MODE", "SSL-HYBRID")            # SSL-HYBRID | SSL+QQE+WAE | SSL-HYBRID+SuperTrend
FILTER = os.environ.get("FILTER", "HULL")               # NO-FILTER | SUPERTREND | HULL

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# --- data source ---
# "auto" tries yfinance first, then falls back to Twelve Data if a key is set.
# Set to "twelvedata" to use it as the primary source (recommended on Render:
# Yahoo Finance frequently rate-limits/blocks shared cloud-host IPs).
DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "auto")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")

# --- risk & exit management (tuned via 5y backtests, see README) ---
BREAKEVEN_AT_R = _f("BREAKEVEN_AT_R", 1.0)        # move stop to entry at +1R profit (0 = off)
EXIT_ON_BASELINE = _b("EXIT_ON_BASELINE", "true") # exit when close crosses back through baseline
ADX_ENTRY_MIN = _f("ADX_ENTRY_MIN", 20)           # block new entries when ADX below (0 = off)
TRAIL_ATR = _f("TRAIL_ATR", 0)                    # trailing stop at N*ATR from best price (0 = off)
ALLOW_SHORTS = _b("ALLOW_SHORTS", "true")

STRATEGY_PARAMS = {
    "sslsignals": MODE,
    "strend": FILTER,
    "squeezesignalfilter": _b("SQUEEZE_FILTER", "true"),
    "adxsignalfilter": _b("ADX_FILTER"),
}
