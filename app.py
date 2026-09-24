"""
AEGIS FX - SSL Hybrid NNFX Trading Bot
Flask app + scheduler. Deployable free to Render (see render.yaml / README.md).

Routes:
  /               professional dashboard
  /api/status     signal, position, checklist, stats
  /api/candles    price history + indicator overlays + trade markers
  /api/trades     trade log
  /api/equity     equity curve
  /api/signals    recent signal feed
"""
import json
import os
import threading
import time
from datetime import datetime, timezone

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, make_response, render_template

import config
import data as data_mod
import trader
import ssl_hybrid
from ssl_hybrid import CHECKLIST_LABELS

app = Flask(__name__, template_folder=".")

STATE = {
    "ready": False,
    "error": None,
    "updated_at": None,
    "symbol": config.SYMBOL,
    "status": {},
    "chart": [],
    "trades": [],
    "equity": [],
    "signals": [],
}
LOCK = threading.Lock()
_last_alert_key = None

_RUNTIME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_state.json")

def _load_runtime():
    try:
        with open(_RUNTIME_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - first boot or ephemeral disk reset
        return {"trading_enabled": True, "paused_after": None}

def _save_runtime(state):
    try:
        with open(_RUNTIME_FILE, "w") as f:
            json.dump(state, f)
    except Exception:  # noqa: BLE001
        pass

RUNTIME = _load_runtime()

# Instantiate the strategy once from config
_strat = ssl_hybrid.SSLHybrid(config.STRATEGY_PARAMS)

# Brand logo (SSL Hybrid strategy emblem)
LOGO_URL = "https://media.base44.com/images/public/6ab0c98c5f34f466a41c41ce/0a2a3dcb9_generated_image.png"


# ------------------------------------------------------------------ engine run
def refresh():
    global _last_alert_key
    try:
        df = data_mod.fetch(config.SYMBOL, config.PERIOD, config.INTERVAL,
                            provider=config.DATA_PROVIDER,
                            twelvedata_key=config.TWELVEDATA_API_KEY)
        if len(df) < 120:
            raise RuntimeError(f"Only {len(df)} bars fetched - not enough history")
        sdf = _strat.compute(df)
        result = trader.simulate(sdf, {
            "capital": config.CAPITAL,
            "risk_pct": config.RISK_PCT,
            "sl_atr_mult": config.SL_ATR_MULT,
            "tp_atr_mult": config.TP_ATR_MULT,
            "fee_bps": config.FEE_BPS,
            "max_leverage": config.MAX_LEVERAGE,
            "breakeven_at_r": config.BREAKEVEN_AT_R,
            "exit_on_baseline": config.EXIT_ON_BASELINE,
            "adx_entry_min": config.ADX_ENTRY_MIN,
            "trail_atr": config.TRAIL_ATR,
            "allow_shorts": config.ALLOW_SHORTS,
            "paused_after": None if RUNTIME["trading_enabled"] else RUNTIME.get("paused_after"),
        })

        # ---- chart payload (last N bars)
        tail = sdf.tail(config.CHART_BARS)
        chart = []
        for dt, row in tail.iterrows():
            chart.append({
                "time": int(dt.timestamp()),
                "open": round(row["open"], 6), "high": round(row["high"], 6),
                "low": round(row["low"], 6), "close": round(row["close"], 6),
                "baseline": _r6(row["baseline"]), "upperk": _r6(row["upperk"]),
                "lowerk": _r6(row["lowerk"]), "ssl_exit": _r6(row["ssl_exit"]),
            })

        # ---- status payload from the last closed bar
        last = sdf.iloc[-1]
        last_bar_time = sdf.index[-1]
        checklist = [{"name": label, "value": int(last[col])} for col, label in CHECKLIST_LABELS]

        last_sig = next((s for s in result["signals"]
                         if s["type"] in ("BUY", "SELL")), None)
        status = {
            "symbol": config.SYMBOL,
            "interval": config.INTERVAL,
            "mode": config.MODE,
            "filter": config.FILTER,
            "price": round(last["close"], 6),
            "bar_time": last_bar_time.isoformat(),
            "signal": ("BUY" if bool(last["signal_up"]) else
                       "SELL" if bool(last["signal_dn"]) else "FLAT"),
            "strength": int(last["strength"]),
            "strength_treshold": _strat.p["strength_treshold"],
            "squeeze": bool(last["squeeze"]),
            "adx": round(float(last["adx"]), 2) if last["adx"] == last["adx"] else None,
            "atr": _r6(last["atr"]),
            "checklist": checklist,
            "position": result["position"],
            "stats": result["stats"],
            "last_signal": last_sig,
            "trading_enabled": bool(RUNTIME["trading_enabled"]),
            "logo": LOGO_URL,
        }

        with LOCK:
            STATE.update(ready=True, error=None,
                         updated_at=datetime.now(timezone.utc).isoformat(),
                         status=status, chart=chart,
                         trades=result["trades"],
                         equity=result["equity_curve"],
                         signals=result["signals"])

        # ---- Telegram alert for the newest bar signal (fires once per bar)
        alert_sig = None
        if bool(last["signal_up"]):
            alert_sig = "BUY"
        elif bool(last["signal_dn"]):
            alert_sig = "SELL"
        if alert_sig:
            key = f"{last_bar_time.date().isoformat()}_{alert_sig}"
            if key != _last_alert_key:
                _last_alert_key = key
                _notify(f"AEGIS FX {config.SYMBOL} [{config.MODE}]: {alert_sig} signal "
                        f"@ {round(last['close'], 6)} (strength {int(last['strength'])}/11)")
        return True
    except Exception as e:  # noqa: BLE001 - surface any failure to the dashboard
        with LOCK:
            STATE["error"] = str(e)
            STATE["updated_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[aegis-fx] refresh failed: {e}", flush=True)
        return False


def _r6(x):
    try:
        return round(float(x), 6)
    except (TypeError, ValueError):
        return None


def _notify(text):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"[aegis-fx] telegram notify failed: {e}", flush=True)


# ------------------------------------------------------------------ keep-alive
def _keepalive():
    """Pings the public URL so Render's free tier doesn't spin the service down."""
    url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("PUBLIC_URL")
    if not url:
        return
    while True:
        try:
            requests.get(url, timeout=10)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(600)


# ------------------------------------------------------------------ routes
@app.route("/")
def dashboard():
    resp = make_response(render_template("dashboard.html"))
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/api/status")
def api_status():
    with LOCK:
        if STATE["ready"]:
            payload = dict(STATE["status"])
            payload["ready"] = True
            return jsonify(payload)
        return jsonify({"ready": False, "error": STATE["error"],
                        "updated_at": STATE["updated_at"],
                        "provider": config.DATA_PROVIDER,
                        "symbol": config.SYMBOL})


@app.route("/api/refresh", methods=["POST", "GET"])
def api_refresh():
    """Manual trigger - lets the dashboard's Retry button force an immediate refresh
    instead of waiting for the scheduled job."""
    ok = refresh()
    with LOCK:
        return jsonify({"ok": ok, "ready": STATE["ready"], "error": STATE["error"]})


@app.route("/api/candles")
def api_candles():
    with LOCK:
        return jsonify(STATE["chart"])


@app.route("/api/trades")
def api_trades():
    with LOCK:
        return jsonify(STATE["trades"])


@app.route("/api/equity")
def api_equity():
    with LOCK:
        return jsonify(STATE["equity"])


@app.route("/api/signals")
def api_signals():
    with LOCK:
        return jsonify(STATE["signals"])


@app.route("/api/trading/toggle", methods=["POST"])
def api_trading_toggle():
    """Start/stop button: pause blocks NEW entries; open positions are still
    managed to their exit. Recomputes the paper account so the dashboard
    reflects it immediately."""
    enabled = not RUNTIME["trading_enabled"]
    RUNTIME["trading_enabled"] = enabled
    RUNTIME["paused_after"] = None if enabled else datetime.now(timezone.utc).isoformat()
    _save_runtime(RUNTIME)
    refresh()
    with LOCK:
        return jsonify({"ok": True, "trading_enabled": enabled})


@app.route("/healthz")
def healthz():
    return "ok"


# ------------------------------------------------------------------ startup
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(refresh, "interval", hours=max(1, int(config.REFRESH_HOURS)))
scheduler.start()

def _startup_retry():
    """First refresh often fails on cloud hosts due to Yahoo rate-limiting a
    fresh IP; retry quickly (not once every REFRESH_HOURS) until it succeeds."""
    delay = 15
    while True:
        with LOCK:
            already_ready = STATE["ready"]
        if already_ready:
            return
        if refresh():
            return
        time.sleep(delay)
        delay = min(delay * 2, 300)


threading.Thread(target=_startup_retry, daemon=True).start()
threading.Thread(target=_keepalive, daemon=True).start()

if __name__ == "__main__":
    refresh()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
