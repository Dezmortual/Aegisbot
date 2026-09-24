"""
SSL HYBRID ADVANCED (NNFX) strategy engine.
Python port of the Pine Script v6 indicator by Trader Hari Krishna / Mihkel00.

Produces, per closed candle:
  - Baseline (SSL1) + Keltner channel entries
  - SSL2 continuation signals
  - SSL3 exit signals
  - QQE MOD + WAE confirmation signals
  - ADX / Squeeze / Supertrend / HULL filters
  - The 10-component "Strength" table
"""
import numpy as np
import pandas as pd

import indicators as ind

DEFAULTS = {
    # signal mode & filter (mirrors the Pine inputs)
    "sslsignals": "SSL-HYBRID",          # SSL-HYBRID | SSL+QQE+WAE | SSL-HYBRID+SuperTrend
    "strend": "NO-FILTER",               # NO-FILTER | SUPERTREND | HULL
    "squeezesignalfilter": False,
    "adxsignalfilter": False,
    # ATR
    "atrlen": 14,
    "atr_mult": 1.5,
    "atr_smoothing": "WMA",
    # SSL1 / Baseline
    "maType": "HMA",
    "len": 55,
    "multy": 0.2,                        # baseline Keltner multiplier
    # SSL2 / Continuation
    "SSL2Type": "JMA",
    "len2": 5,
    "atr_crit": 0.9,                     # continuation ATR criteria
    # SSL3 / Exit
    "SSL3Type": "HMA",
    "len3": 15,
    # Jurik / VAMA extras
    "jurik_phase": 3,
    "jurik_power": 1,
    "volatility_lookback": 10,
    # QQE MOD
    "rsi_period": 6, "rsi_sf": 5, "qqe": 3.0, "qqe2": 1.61, "threshold": 3.0,
    "qqe_bb_len": 50, "qqe_bb_mult": 0.35,
    # WAE
    "wae_sensitivity": 150, "wae_fast": 20, "wae_slow": 40, "wae_bb_len": 20, "wae_bb_mult": 2.0,
    # ADX
    "adx_len": 14, "adx_sm": 14, "adx_treshold": 20,
    # Supertrend filter
    "st_mult": 3.0, "st_len": 10,
    # HULL filter
    "hull_len": 110,
    # EMA colors
    "ema1": 20, "ema2": 50,
    # Oscillators
    "rsi_len": 7, "rsi_sm": 18,
    "cci_len": 40,
    "stoch_len": 14, "stoch_k": 3, "stoch_d": 3, "stoch_ob": 80, "stoch_os": 20,
    "psar_start": 0.02, "psar_inc": 0.02, "psar_max": 0.2,
    # Squeeze (BB vs Keltner)
    "sq_matype": "SMA", "k_length": 20, "k_dev": 2.0, "bb_length": 20, "bb_dev": 2.0,
    # Strength table
    "strength_treshold": 8,
}


class SSLHybrid:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update({k: v for k, v in params.items() if k in DEFAULTS or True})
        self.p = p

    # ------------------------------------------------------------------ compute
    def compute(self, df):
        p = self.p
        out = df.copy()
        close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

        ma_kw = {k: p[k] for k in ("jurik_phase", "jurik_power", "volatility_lookback")}

        # --- ATR bands
        atr_slen = ind.atr(df, p["atrlen"], p["atr_smoothing"])
        out["atr"] = atr_slen
        out["upper_band"] = atr_slen * p["atr_mult"] + close
        out["lower_band"] = close - atr_slen * p["atr_mult"]

        # --- SSL1 / Baseline
        ema_high = ind.ma(p["maType"], high, p["len"], **ma_kw)
        ema_low = ind.ma(p["maType"], low, p["len"], **ma_kw)
        hlv = self._hlv(close, ema_high, ema_low)
        out["ssl_down"] = pd.Series(np.where(hlv < 0, ema_high, ema_low), index=df.index)

        # Baseline Keltner channel (on the baseline MA of close)
        BBMC = ind.ma(p["maType"], close, p["len"], **ma_kw)
        keltma = BBMC
        rng = ind.true_range(df) if True else high - low
        rangema = ind.ema(rng, p["len"])
        out["baseline"] = BBMC
        out["upperk"] = keltma + rangema * p["multy"]
        out["lowerk"] = keltma - rangema * p["multy"]

        # --- SSL2 / Continuation
        ma_high = ind.ma(p["SSL2Type"], high, p["len2"], **ma_kw)
        ma_low = ind.ma(p["SSL2Type"], low, p["len2"], **ma_kw)
        hlv2 = self._hlv(close, ma_high, ma_low)
        out["ssl_down2"] = pd.Series(np.where(hlv2 < 0, ma_high, ma_low), index=df.index)

        # --- SSL3 / Exit
        exit_high = ind.ma(p["SSL3Type"], high, p["len3"], **ma_kw)
        exit_low = ind.ma(p["SSL3Type"], low, p["len3"], **ma_kw)
        hlv3 = self._hlv(close, exit_high, exit_low)
        out["ssl_exit"] = pd.Series(np.where(hlv3 < 0, exit_high, exit_low), index=df.index)

        # Continuation within ATR criteria
        upper_half = atr_slen * p["atr_crit"] + close
        lower_half = close - atr_slen * p["atr_crit"]
        buy_inatr = lower_half < out["ssl_down2"]
        sell_inatr = upper_half > out["ssl_down2"]
        buy_cont = (close > BBMC) & (close > out["ssl_down2"])
        sell_cont = (close < BBMC) & (close < out["ssl_down2"])
        out["cont_buy"] = buy_inatr & buy_cont
        out["cont_sell"] = sell_inatr & sell_cont

        # --- QQE MOD
        rsi_ma, fast_tl, fast_tl2 = ind.qqe_mod(
            df, p["rsi_period"], p["rsi_sf"], p["qqe"], None, p["qqe2"])
        out["rsi_ma"] = rsi_ma
        rsi_ma2 = ind.ema(ind.rsi(close, p["rsi_period"]), p["rsi_sf"])
        basis = ind.sma(fast_tl - 50, p["qqe_bb_len"])
        dev = p["qqe_bb_mult"] * (fast_tl - 50).rolling(p["qqe_bb_len"]).std(ddof=0)
        upper_q = basis + dev
        lower_q = basis - dev
        out["qqe_green1"] = (rsi_ma2 - 50) > p["threshold"]
        out["qqe_green2"] = (rsi_ma - 50) > upper_q
        out["qqe_red1"] = (rsi_ma2 - 50) < -p["threshold"]
        out["qqe_red2"] = (rsi_ma - 50) < lower_q
        qqe_green = out["qqe_green1"] & out["qqe_green2"]
        qqe_red = out["qqe_red1"] & out["qqe_red2"]
        out["qqe_buy"] = qqe_green & ~qqe_green.shift(1).fillna(False)
        out["qqe_sell"] = qqe_red & ~qqe_red.shift(1).fillna(False)

        # --- WAE
        t_up, t_dn, e1 = ind.waddah_attar(df, p["wae_sensitivity"], p["wae_fast"],
                                          p["wae_slow"], p["wae_bb_len"], p["wae_bb_mult"])
        out["wae_buy"] = (t_up > 0) & (t_up > e1)
        out["wae_sell"] = (t_dn > 0) & (t_dn > e1)

        # --- ADX / DMI
        plus_di, minus_di, adx = ind.dmi(df, p["adx_len"], p["adx_sm"])
        out["adx"] = adx
        out["plus_di"] = plus_di
        out["minus_di"] = minus_di

        # --- Supertrend
        st_line, direction = ind.supertrend(df, p["st_mult"], p["st_len"])
        out["st_dir"] = direction  # < 0 = uptrend (Pine convention)
        st_change = direction != direction.shift(1)
        out["st_up_flip"] = (direction < 0) & st_change
        out["st_dn_flip"] = (direction > 0) & st_change

        # --- HULL filter line
        hull = ind.hma(close, p["hull_len"])
        out["hull"] = hull
        out["hma_direction"] = np.where(close > hull, 1, np.where(close < hull, -1, 0))

        # --- Bollinger vs Keltner squeeze
        kma = ind.ma(p["sq_matype"], close, p["k_length"])
        k_upper = kma + p["k_dev"] * ind.rma(ind.true_range(df), p["k_length"])
        k_lower = kma - p["k_dev"] * ind.rma(ind.true_range(df), p["k_length"])
        bb_u, bb_l = ind.bollinger(close, p["bb_length"], p["bb_dev"])
        out["squeeze"] = (bb_u <= k_upper) & (bb_l >= k_lower)

        # --- EMAs, PSAR, oscillators
        ma1 = ind.ema(close, p["ema1"])
        ma2 = ind.ema(close, p["ema2"])
        out["ema1"] = ma1
        out["ema2"] = ma2
        sar, sar_bull = ind.psar(df, p["psar_start"], p["psar_inc"], p["psar_max"])
        out["psarn"] = np.where(sar < close, 1, -1)

        r = ind.rsi(close, p["rsi_len"])
        mr = ind.ema(ind.ema(ind.ema(r, p["rsi_sm"]), p["rsi_sm"]), p["rsi_sm"])
        out["rsi_sig"] = np.where(r > mr, 1, np.where(r < mr, -1, 0))

        hlc3 = (high + low + close) / 3
        c = ind.cci(hlc3, p["cci_len"])
        out["cci_sig"] = np.where(c > 100, 1, np.where(c < -100, -1, 0))

        k_, d_ = ind.stoch(df, p["stoch_len"], p["stoch_k"], p["stoch_d"])
        co = ind.crossover(k_, d_)
        cu = ind.crossunder(k_, d_)
        sup = (k_ > p["stoch_ob"]) | (cu & (k_ > p["stoch_ob"]))
        sdn = (k_ < p["stoch_os"]) | (co & (k_ < p["stoch_os"]))
        stobg = np.where(sup, 1, np.where(sdn, -1, np.where(co, 2, np.where(cu, -2, 0))))
        out["sto_sig"] = stobg

        # --- checklist components (numeric, mirrors the Pine "colors")
        out["chk_ssl"] = np.where((close > out["upperk"]) & (close > BBMC), 1,
                                  np.where((close < out["lowerk"]) & (close < BBMC), -1, 0))
        out["chk_hma"] = np.where(close > out["upperk"], 1,
                                  np.where(close < out["lowerk"], -1, 0))
        out["chk_candle"] = out["psarn"]
        out["chk_qqe"] = np.where(out["qqe_green1"] & out["qqe_green2"], 1,
                                  np.where(out["qqe_red1"] & out["qqe_red2"], -1, 0))
        out["chk_qqedot"] = np.where(out["qqe_green1"] & out["qqe_green2"] & out["wae_buy"], 1,
                                    np.where(out["qqe_red1"] & out["qqe_red2"] & out["wae_sell"], -1, 0))
        out["chk_st"] = np.where(direction < 0, 1, np.where(direction > 0, -1, 0))
        out["chk_adx"] = np.where((adx > p["adx_treshold"]) & (plus_di > minus_di), 1,
                                  np.where((adx > p["adx_treshold"]) & (minus_di > plus_di), -1, 0))
        out["chk_wae"] = np.where(out["wae_buy"], 1, np.where(out["wae_sell"], -1, 0))
        out["chk_ema"] = np.where(ind.crossover(ma1, ma2) | (close > ma2), 1,
                                  np.where(ind.crossunder(ma1, ma2) | (close < ma2), -1, 0))

        # --- Strength table (-11..11)
        out["strength"] = (out["psarn"] + out["chk_ssl"] + out["chk_qqedot"] + out["chk_wae"]
                           + out["chk_adx"] + out["chk_ema"] + out["sto_sig"]
                           + out["rsi_sig"] + out["cci_sig"] + out["chk_st"])

        # --- Entries / exits
        out["exit_long"] = ind.crossover(out["ssl_exit"], close)    # close fell under exit MA
        out["exit_short"] = ind.crossover(close, out["ssl_exit"])   # close rose above exit MA
        # optional extra exit: close crossing back through the baseline
        out["exit_long_base"] = ind.crossunder(close, BBMC)
        out["exit_short_base"] = ind.crossover(close, BBMC)

        ssl_buy = (close > out["upperk"]) & (close > BBMC)
        ssl_sell = (close < out["lowerk"]) & (close < BBMC)
        ssl_up = ind.crossover(close, out["ssl_exit"])
        ssl_dn = ind.crossover(out["ssl_exit"], close)

        sideways = ((p["squeezesignalfilter"] & out["squeeze"]) |
                    (p["adxsignalfilter"] & (adx < p["adx_treshold"])))

        out["base_buy"] = ind.crossover(close, out["upperk"]) & ~sideways
        out["base_sell"] = ind.crossover(out["lowerk"], close) & ~sideways

        mode_up = {
            "SSL-HYBRID": ssl_up,
            "SSL+QQE+WAE": out["wae_buy"] & out["qqe_buy"] & ssl_buy,
            "SSL-HYBRID+SuperTrend": ssl_up | out["st_up_flip"],
        }
        mode_dn = {
            "SSL-HYBRID": ssl_dn,
            "SSL+QQE+WAE": out["wae_sell"] & out["qqe_sell"] & ssl_sell,
            "SSL-HYBRID+SuperTrend": ssl_dn | out["st_dn_flip"],
        }
        signal_up = mode_up.get(p["sslsignals"], ssl_up).fillna(False)
        signal_dn = mode_dn.get(p["sslsignals"], ssl_dn).fillna(False)

        if p["strend"] == "SUPERTREND":
            signal_up = signal_up & (direction < 0)
            signal_dn = signal_dn & (direction > 0)
        elif p["strend"] == "HULL":
            signal_up = signal_up & (out["hma_direction"] > 0)
            signal_dn = signal_dn & (out["hma_direction"] < 0)

        out["signal_up"] = signal_up & ~sideways
        out["signal_dn"] = signal_dn & ~sideways
        return out

    @staticmethod
    def _hlv(close, up_line, dn_line):
        """SSL Heikin-style state: 1 above upper line, -1 below lower line, else previous."""
        vals = close.values
        up_v, dn_v = up_line.values, dn_line.values
        out = np.zeros(len(vals))
        state = 0
        for i in range(len(vals)):
            c = vals[i]
            if np.isnan(c) or np.isnan(up_v[i]) or np.isnan(dn_v[i]):
                out[i] = state
                continue
            if c > up_v[i]:
                state = 1
            elif c < dn_v[i]:
                state = -1
            out[i] = state
        return pd.Series(out, index=close.index)


CHECKLIST_LABELS = [
    ("chk_ssl", "SSL / Baseline Signal"),
    ("chk_hma", "Baseline Channel (HMA)"),
    ("chk_candle", "Candle Color (PSAR)"),
    ("chk_qqe", "QQE Color"),
    ("chk_qqedot", "QQE + WAE Dot"),
    ("chk_st", "Supertrend"),
    ("chk_adx", "ADX / DMI"),
    ("chk_wae", "WAE Explosion"),
    ("chk_ema", "EMA 20/50"),
    ("sto_sig", "Stochastic"),
    ("rsi_sig", "RSI (smoothed)"),
    ("cci_sig", "CCI"),
]
