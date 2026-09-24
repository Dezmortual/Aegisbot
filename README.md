# AEGIS FX Trading Bot

An automated **paper-trading robot** built from the *SSL HYBRID ADVANCED* Pine Script
(NNFX method, by Trader Hari Krishna / Mihkel00). The complete indicator logic was
ported to Python: SSL1 baseline with Keltner channel, SSL2 continuations, SSL3 exits,
QQE MOD, Waddah Attar Explosion, ADX/DMI, Supertrend, Hull filter, Bollinger squeeze
and the 10-component strength table.

- **100% free stack** - Python + Flask, GitHub, Render free tier, yfinance data
- **Professional dark dashboard** - live signal, strength meter, candlestick chart
  with baseline channel & trade markers, equity curve, checklist and trade log
- **Self-healing** - the bot deterministically replays the strategy over full price
  history on every refresh, so it never loses state, even after Render spins it down
- **Telegram alerts** (optional) - get a message on every new BUY/SELL signal

> This is a paper-trading / research bot. It does **not** place real orders.
> Trading involves substantial risk. Not financial advice.

---

## 1. How it trades (NNFX rules)

| Event | Rule |
|---|---|
| **Entry** | `signal_up` / `signal_dn` per the selected mode: **SSL-HYBRID** (SSL3 exit-line cross), **SSL+QQE+WAE** (WAE explosion + QQE + baseline agreement), or **SSL-HYBRID+SuperTrend** |
| **Filter** | Optional trend filter: **NO-FILTER**, **SUPERTREND** or **HULL** (110 HMA direction) |
| **Exit** | Close crossing the SSL3 exit MA, an opposite signal (position flips), or ATR stop/target |
| **Sizing** | Risk `%` of equity per trade; stop distance = ATR × `SL_ATR_MULT` |
| **Continuations** | SSL2 + ATR continuation alerts appear in the signal feed (logged, not traded) |
| **Strength table** | Sum of 10 component signals (PSAR, baseline, QQE dot, WAE, ADX, EMA, Stoch, RSI, CCI, Supertrend), range −11…+11 |

Recommended for **Daily charts** (NNFX method), which is the default.

## 2. Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 - the dashboard populates after the first data fetch.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "AEGIS FX trading bot"
git branch -M main
git remote add origin https://github.com/<your-username>/aegis-fx-bot.git
git push -u origin main
```

(Create the empty repo first on [github.com/new](https://github.com/new) - pick **Private**.)

## 4. Deploy to Render (free)

**Option A - Blueprint (easiest):**
1. Go to [dashboard.render.com](https://dashboard.render.com) → **New → Blueprint**
2. Select your `aegis-fx-bot` repo. The `render.yaml` in this repo is detected automatically.
3. Click **Apply**. Done - free web service, no credit card needed.

**Option B - Manual:**
1. **New → Web Service** → pick your repo
2. Runtime: **Python 3** · Build: `pip install -r requirements.txt`
3. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8`
4. Instance type: **Free**

### Free tier notes

- Render free web services sleep after 15 min without traffic. The bot includes a
  **keep-alive thread** that pings its own public URL every 10 minutes (uses
  `RENDER_EXTERNAL_URL`, which Render sets automatically), so your scheduler keeps running.
- Even if it does sleep: on wake it re-simulates the full history from yfinance, so no
  trades or signals are ever missed.

## 5. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SYMBOL` | `GBPJPY=X` | Any yfinance symbol: forex `EURUSD=X`, crypto `BTC-USD`, stocks `AAPL` |
| `INTERVAL` | `1d` | Bar size (NNFX: daily) |
| `PERIOD` | `730d` | History to load |
| `CAPITAL` | `10000` | Starting paper equity ($) |
| `RISK_PCT` | `1.0` | % of equity risked per trade |
| `SL_ATR_MULT` | `1.5` | Stop-loss distance in ATRs |
| `TP_ATR_MULT` | `4.0` | Take-profit distance in ATRs |
| `BREAKEVEN_AT_R` | `1.0` | Move stop to entry once profit reaches +N·risk (0 = off) |
| `EXIT_ON_BASELINE` | `true` | Also exit when close crosses back through the baseline |
| `ADX_ENTRY_MIN` | `20` | Block new entries when ADX < this (0 = off) |
| `TRAIL_ATR` | `0` | Optional trailing stop at N·ATR from best price (0 = off) |
| `ALLOW_SHORTS` | `true` | Set `false` for long-only trading |
| `MAX_LEVERAGE` | `5.0` | Cap on position notional |
| `FEE_BPS` | `1.0` | Simulated cost per side, basis points |
| `MODE` | `SSL-HYBRID` | `SSL-HYBRID` · `SSL+QQE+WAE` · `SSL-HYBRID+SuperTrend` |
| `FILTER` | `HULL` | `NO-FILTER` · `SUPERTREND` · `HULL` |
| `SQUEEZE_FILTER` | `true` | Drop signals during Bollinger squeeze |
| `ADX_FILTER` | `false` | Drop signals when ADX < threshold (sideways) |
| `REFRESH_HOURS` | `6` | How often to recompute / check for new signals |
| `TELEGRAM_BOT_TOKEN` | – | Optional: Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | – | Optional: your Telegram chat id |

### Telegram alerts (optional)

1. Create a bot with [@BotFather](https://t.me/BotFather) → copy the token
2. Message your bot once, then get your chat id from `api.telegram.org/bot<TOKEN>/getUpdates`
3. Set both env vars in Render → Environment

## 6. Tuning for profitability (what we found)

The bot ships with settings chosen by sweeping ~200 backtest configurations over
5 years of daily data. The default signals alone are a **chop-losing trend system** -
raw SSL-HYBRID on EURUSD 2y lost ~12%. Three regime filters turn it around:

| Mechanism | Effect |
|---|---|
| **ADX entry gate (≥20)** | No new trades in sideways regimes. Biggest single improvement: EURUSD 2y losses cut ~75% |
| **Bollinger-squeeze filter** | Skip entries while BB is inside the Keltner channel (volatility dry-up) |
| **Hull trend filter (HMA 110)** | Only trade in the direction of the long-term trend |
| **Breakeven stop (+1R)** | Once a trade is +1R in profit, the stop moves to entry: losers become scratches |
| **Baseline exit** | Exit when close crosses back through the baseline, faster than waiting for SSL3 |

**Tuned defaults (GBPJPY=X daily, 5y):** +4.2% net, profit factor 2.0, max drawdown 2.8%,
15 trades (33% win rate — classic trend-following: many small scratches, few big winners).

**Honest caveats:**
- The edge is **pair-specific**. The same config is still negative on EURUSD/GBPUSD/SPY:
  SSL-Hybrid needs a volatile, trending instrument (GBPJPY, indices, crypto in trends).
  Test `SYMBOL` before trusting any pair.
- 15 trades over 5 years is a thin sample. Treat results as directional, not proven.
- Daily backtests fill at signal-bar close and ignore slippage/weekend gaps; live
  results will differ.

## 7. Project structure

```
app.py           Flask app, scheduler, keep-alive, API routes
config.py        Env-based configuration
indicators.py    Pine → Python ports (HMA, JMA, QQE, WAE, ADX, Supertrend, PSAR, …)
ssl_hybrid.py    Signal engine: baseline entries, continuations, exits, filters, strength
data.py          yfinance market data
trader.py        Paper-trading simulation (entries, ATR stops, flips, stats)
dashboard.html   The dashboard (Lightweight Charts + Chart.js, no build step)
render.yaml      Render Blueprint (free plan)
```

Everything sits flat in one folder - no subfolders - so you can upload the files
one by one into a GitHub repo straight from your phone. Upload **all 13 files to
the repo root** (don't put any inside folders).

## Credits

Original Pine Script: *SSL HYBRID ADVANCED* by Trader Hari Krishna (Mihkel00),
QQE MOD, Waddah Attar Explosion, and the NNFX community. Strategy components
© their respective TradingView authors.

MIT licensed - see `LICENSE`.
