"""
Paper-trading engine.

Deterministically replays the strategy over the full price history on every
refresh, so the bot self-heals after Render free-tier spin-downs: whatever
happened while asleep is recomputed from actual market data.

Rules (NNFX style):
  Entry  : signal_up / signal_dn from the SSL Hybrid engine (mode + filter aware)
  Exit   : close crossing the SSL3 exit line, an opposite signal (flip), or
           ATR-based stop-loss / take-profit
  Sizing : risk % of equity per trade, stop distance = ATR * SL mult
"""
import numpy as np
import pandas as pd


def simulate(sdf, cfg):
    p = cfg
    df = sdf
    # first bar where the strategy is fully warmed up
    warm_cols = ["baseline", "upperk", "lowerk", "ssl_exit", "atr", "strength"]
    valid = df[warm_cols].dropna()
    start = df.index.get_loc(valid.index[0]) + 1 if len(valid) else len(df)

    capital = float(p["capital"])
    risk_pct = float(p["risk_pct"])
    sl_mult = float(p["sl_atr_mult"])
    tp_mult = float(p["tp_atr_mult"])
    fee_bps = float(p["fee_bps"])
    max_leverage = float(p["max_leverage"])
    breakeven_at_r = float(p.get("breakeven_at_r", 0))     # move stop to entry at +N R (0 = off)
    trail_atr = float(p.get("trail_atr", 0))               # trail stop at N*ATR from best price (0 = off)
    adx_entry_min = float(p.get("adx_entry_min", 0))      # block new entries when ADX < this (0 = off)
    exit_on_baseline = bool(p.get("exit_on_baseline", False))
    allow_shorts = bool(p.get("allow_shorts", True))
    paused_after = p.get("paused_after")  # ISO date str: block NEW entries at/after it

    paused_ts = pd.to_datetime(paused_after) if paused_after else None
    if paused_ts is not None and paused_ts.tzinfo is not None:
        paused_ts = paused_ts.tz_localize(None)  # bar index is tz-naive

    position = None
    trades = []
    signals = []
    equity = capital
    curve = []          # (date, equity marked-to-market)
    peak = capital
    max_dd = 0.0

    dates = df.index

    def entries_allowed(i):
        if paused_ts is None:
            return True
        return dates[i] < paused_ts

    def close_position(i, price, reason):
        nonlocal position, equity
        pos = position
        fee = pos["notional"] * fee_bps / 10000
        pnl = pos["qty"] * (price - pos["entry"]) * (1 if pos["side"] == "long" else -1) - pos["entry_fee"] - fee
        equity += pnl
        trades.append({
            "side": pos["side"],
            "entry_time": pos["time"].isoformat(),
            "entry_price": round(pos["entry"], 6),
            "exit_time": dates[i].isoformat(),
            "exit_price": round(price, 6),
            "qty": round(pos["qty"], 6),
            "stop": round(pos["stop"], 6),
            "target": round(pos["target"], 6),
            "reason": reason,
            "pnl": round(pnl, 2),
            "pnl_pct": round(100 * pnl / pos["equity_at_entry"], 3),
            "equity": round(equity, 2),
        })
        signals.append({"time": dates[i].isoformat(), "type": f"EXIT_{pos['side'].upper()}",
                        "price": round(price, 6), "reason": reason})
        position = None

    def open_position(i, side):
        nonlocal position
        row = df.iloc[i]
        entry = row["close"]
        atr = row["atr"]
        if not atr or np.isnan(atr) or atr <= 0:
            return
        stop_dist = atr * sl_mult
        risk_amt = equity * risk_pct / 100.0
        qty = risk_amt / stop_dist
        max_qty = (equity * max_leverage) / entry
        qty = min(qty, max_qty)
        if qty <= 0:
            return
        notional = qty * entry
        position = {
            "side": side, "time": dates[i], "entry": entry, "qty": qty,
            "notional": notional, "entry_fee": notional * fee_bps / 10000,
            "stop": entry - stop_dist if side == "long" else entry + stop_dist,
            "target": entry + atr * tp_mult if side == "long" else entry - atr * tp_mult,
            "risk_dist": stop_dist, "be_moved": False, "best": entry, "atr0": atr,
            "equity_at_entry": equity,
        }
        signals.append({"time": dates[i].isoformat(),
                        "type": "BUY" if side == "long" else "SELL",
                        "price": round(entry, 6),
                        "strength": int(row["strength"]) if not np.isnan(row["strength"]) else 0})

    for i in range(start, len(df)):
        row = df.iloc[i]

        # ---- manage open position
        if position:
            # track best price for the trailing stop
            if position["side"] == "long":
                position["best"] = max(position["best"], row["high"])
            else:
                position["best"] = min(position["best"], row["low"])
            if trail_atr > 0:
                trail_stop = (position["best"] - trail_atr * position["atr0"]) if position["side"] == "long" \
                    else (position["best"] + trail_atr * position["atr0"])
                if position["side"] == "long":
                    position["stop"] = max(position["stop"], trail_stop)
                else:
                    position["stop"] = min(position["stop"], trail_stop)
            # breakeven stop: once profit >= breakeven_at_r * risk, protect the entry
            if breakeven_at_r > 0 and not position["be_moved"]:
                gain = (row["close"] - position["entry"]) * (1 if position["side"] == "long" else -1)
                if gain >= breakeven_at_r * position["risk_dist"]:
                    position["stop"] = position["entry"] if position["side"] == "long" \
                        else position["entry"]
                    position["be_moved"] = True
            # intrabar SL / TP
            if position["side"] == "long":
                if row["low"] <= position["stop"]:
                    close_position(i, position["stop"], "STOP" if not position["be_moved"] else "BREAKEVEN")
                elif row["high"] >= position["target"]:
                    close_position(i, position["target"], "TARGET")
            else:
                if row["high"] >= position["stop"]:
                    close_position(i, position["stop"], "STOP" if not position["be_moved"] else "BREAKEVEN")
                elif row["low"] <= position["target"]:
                    close_position(i, position["target"], "TARGET")
            # signal exits at close
            if position:
                if position["side"] == "long" and bool(row["exit_long"]):
                    close_position(i, row["close"], "SSL3 EXIT")
                elif position["side"] == "short" and bool(row["exit_short"]):
                    close_position(i, row["close"], "SSL3 EXIT")
                # optional baseline violation exit (faster than SSL3)
                if exit_on_baseline and position:
                    if position["side"] == "long" and bool(row["exit_long_base"]):
                        close_position(i, row["close"], "BASELINE EXIT")
                    elif position["side"] == "short" and bool(row["exit_short_base"]):
                        close_position(i, row["close"], "BASELINE EXIT")
            # flip on opposite signal
            if position:
                if position["side"] == "long" and bool(row["signal_dn"]):
                    close_position(i, row["close"], "FLIP" if (allow_shorts and entries_allowed(i)) else "OPPOSITE SIGNAL")
                    if allow_shorts and entries_allowed(i):
                        open_position(i, "short")
                elif position["side"] == "short" and bool(row["signal_up"]):
                    if allow_shorts or entries_allowed(i):
                        close_position(i, row["close"], "FLIP" if allow_shorts else "OPPOSITE SIGNAL")
                    if allow_shorts and entries_allowed(i):
                        open_position(i, "long")
                elif position["side"] == "long" and bool(row["signal_dn"]) and not allow_shorts:
                    close_position(i, row["close"], "OPPOSITE SIGNAL")

        # ---- entries
        if position is None and entries_allowed(i):
            adx_ok = True
            if adx_entry_min > 0:
                adx_ok = row["adx"] == row["adx"] and row["adx"] >= adx_entry_min
            if adx_ok and bool(row["signal_up"]):
                open_position(i, "long")
            elif adx_ok and bool(row["signal_dn"]) and allow_shorts:
                open_position(i, "short")

        # ---- continuation alerts (not traded, surfaced on the dashboard)
        if bool(row["cont_buy"]):
            signals.append({"time": dates[i].isoformat(), "type": "CONT_BUY",
                            "price": round(row["close"], 6)})
        elif bool(row["cont_sell"]):
            signals.append({"time": dates[i].isoformat(), "type": "CONT_SELL",
                            "price": round(row["close"], 6)})

        # ---- mark to market
        mtm = equity
        if position:
            mtm += position["qty"] * (row["close"] - position["entry"]) * \
                (1 if position["side"] == "long" else -1)
        curve.append((dates[i].isoformat(), round(mtm, 2)))
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak * 100 if peak > 0 else 0)

    # ---- stats
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    stats = {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(100 * len(wins) / len(trades), 1) if trades else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (None if gross_win == 0 else 999),
        "total_pnl": round(equity - capital, 2),
        "total_pnl_pct": round(100 * (equity - capital) / capital, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "equity": round(equity, 2),
    }

    # ---- open position snapshot
    open_pos = None
    if position:
        last = df.iloc[-1]
        upnl = position["qty"] * (last["close"] - position["entry"]) * \
            (1 if position["side"] == "long" else -1)
        open_pos = {
            "side": position["side"],
            "entry_time": position["time"].isoformat(),
            "entry_price": round(position["entry"], 6),
            "qty": round(position["qty"], 6),
            "stop": round(position["stop"], 6),
            "target": round(position["target"], 6),
            "unrealized_pnl": round(upnl, 2),
        }

    return {"trades": trades[::-1], "equity_curve": curve, "signals": signals[::-1][:120],
            "stats": stats, "position": open_pos}
