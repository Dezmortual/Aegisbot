"""
Indicator library - Python ports of the Pine Script v6 functions used by
SSL HYBRID ADVANCED (NNFX). Pure pandas/numpy, no compiled dependencies.
"""
import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- moving averages
def sma(s, n):
    return s.rolling(int(n)).mean()


def ema(s, n):
    return s.ewm(span=int(n), adjust=False).mean()


def rma(s, n):
    return s.ewm(alpha=1.0 / int(n), adjust=False).mean()


def wma(s, n):
    n = int(n)
    if n < 1:
        n = 1
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def dema(s, n):
    e = ema(s, n)
    return 2 * e - ema(e, n)


def tema(s, n):
    e = ema(s, n)
    return 3 * (e - ema(e, n)) + ema(ema(e, n), n)


def tma(s, n):
    n = int(n)
    return sma(sma(s, math.ceil(n / 2)), math.floor(n / 2) + 1)


def hma(s, n):
    n = int(n)
    half = max(1, n // 2)
    sq = max(1, int(round(math.sqrt(n))))
    return wma(2 * wma(s, half) - wma(s, n), sq)


def lsma(s, n):
    n = int(n)

    def _lr(x):
        t = np.arange(n, dtype=float)
        p = np.polyfit(t, x, 1)
        return p[0] * (n - 1) + p[1]

    return s.rolling(n).apply(_lr, raw=True)


def zlsma(s, n):
    l = lsma(s, n)
    l2 = lsma(l, n)
    return l + (l - l2)


def jma(s, n, phase=3.0, power=1.0):
    """Jurik Moving Average (port of everget's JMA)."""
    n = int(n)
    src = s.values.astype(float)
    out = np.full_like(src, np.nan)
    if phase < -100:
        pr = 0.5
    elif phase > 100:
        pr = 2.5
    else:
        pr = phase / 100 + 1.5
    beta = 0.45 * (n - 1) / (0.45 * (n - 1) + 2)
    alpha = beta ** power
    e0 = e1 = e2 = j = 0.0
    started = False
    for i, v in enumerate(src):
        if np.isnan(v):
            continue
        if not started:
            e0, e1, e2, j = v, 0.0, 0.0, v
            started = True
            out[i] = j
            continue
        e0 = (1 - alpha) * v + alpha * e0
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + pr * e1 - j) * (1 - alpha) ** 2 + alpha ** 2 * e2
        j = e2 + j
        out[i] = j
    return pd.Series(out, index=s.index)


def mcginley(s, n):
    n = int(n)
    src = s.values.astype(float)
    out = np.full_like(src, np.nan)
    mg = np.nan
    for i, v in enumerate(src):
        if np.isnan(v):
            continue
        if np.isnan(mg):
            mg = ema(pd.Series(src[: i + 1]), n).iloc[-1]
        else:
            denom = n * (v / mg) ** 4 if mg != 0 else 0
            mg = mg + (v - mg) / denom if denom else mg
        out[i] = mg
    return pd.Series(out, index=s.index)


def vama(s, n, vol_lookback=10):
    mid = ema(s, n)
    dev = s - mid
    vol_up = dev.rolling(int(vol_lookback)).max()
    vol_dn = dev.rolling(int(vol_lookback)).min()
    return mid + (vol_up + vol_dn) / 2


def kijun2(s, n, kidiv=1):
    n = int(n)
    k = (s.rolling(n).min() + s.rolling(n).max()) / 2
    m = max(1, int(n / max(1, kidiv)))
    conv = (s.rolling(m).min() + s.rolling(m).max()) / 2
    return (k + conv) / 2


def mf(s, n, beta=0.8, feedback=False, z=0.5):
    """Modular Filter."""
    n = int(n)
    alpha = 2.0 / (n + 1)
    vals = s.values.astype(float)
    out = np.zeros_like(vals)
    ts = b = c = 0.0
    os_ = 0
    prev_ts = np.nan
    for i, src in enumerate(vals):
        if np.isnan(src):
            continue
        a = z * src + (1 - z) * (prev_ts if not np.isnan(prev_ts) else src) if feedback else src
        b = a if a > alpha * a + (1 - alpha) * (b if b else a) else alpha * a + (1 - alpha) * (b if b else a)
        c = a if a < alpha * a + (1 - alpha) * (c if c else a) else alpha * a + (1 - alpha) * (c if c else a)
        os_ = 1 if a == b else (0 if a == c else os_)
        upper = beta * b + (1 - beta) * c
        lower = beta * c + (1 - beta) * b
        ts = os_ * upper + (1 - os_) * lower
        prev_ts = ts
        out[i] = ts
    return pd.Series(out, index=s.index)


def _ssf2(src, length):
    PI = 2 * math.asin(1)
    arg = math.sqrt(2) * PI / length
    a1 = math.exp(-arg)
    b1 = 2 * a1 * math.cos(arg)
    c2, c3 = b1, -(a1 ** 2)
    c1 = 1 - c2 - c3
    out = np.zeros_like(src)
    for i in range(len(src)):
        p1 = out[i - 1] if i >= 1 else 0
        p2 = out[i - 2] if i >= 2 else 0
        out[i] = c1 * src[i] + c2 * p1 + c3 * p2
    return out


def _ssf3(src, length):
    PI = 2 * math.asin(1)
    arg = PI / length
    a1 = math.exp(-arg)
    b1 = 2 * a1 * math.cos(1.738 * arg)
    c1 = a1 ** 2
    coef2 = b1 + c1
    coef3 = -(c1 + b1 * c1)
    coef4 = c1 ** 2
    coef1 = 1 - coef2 - coef3 - coef4
    out = np.zeros_like(src)
    for i in range(len(src)):
        p1 = out[i - 1] if i >= 1 else 0
        p2 = out[i - 2] if i >= 2 else 0
        p3 = out[i - 3] if i >= 3 else 0
        out[i] = coef1 * src[i] + coef2 * p1 + coef3 * p2 + coef4 * p3
    return out


def edsma(s, n, ssf_length=20, ssf_poles=2):
    n = int(n)
    src = s.values.astype(float)
    zeros = np.nan_to_num(src - np.roll(src, 2))
    zeros[:2] = 0
    avg_zeros = (zeros + np.roll(zeros, 1)) / 2
    avg_zeros[:1] = 0
    ssf = _ssf2(avg_zeros, ssf_length) if ssf_poles == 2 else _ssf3(avg_zeros, ssf_length)
    ssf = pd.Series(ssf, index=s.index)
    stdev = ssf.rolling(n).std(ddof=0)
    scaled = np.where(stdev != 0, ssf / stdev, 0)
    alpha = pd.Series(5 * np.abs(scaled) / n, index=s.index)
    alpha = alpha.clip(0, 1)
    out = np.zeros(len(s))
    prev = 0.0
    for i in range(len(s)):
        out[i] = alpha.iloc[i] * src[i] + (1 - alpha.iloc[i]) * prev
        prev = out[i]
    return pd.Series(out, index=s.index)


def ma(ma_type, src, length, **kw):
    """Dispatcher mirroring the Pine `ma(type, src, len)` function."""
    t = str(ma_type).upper()
    if t == "SMA":
        return sma(src, length)
    if t == "EMA":
        return ema(src, length)
    if t == "WMA":
        return wma(src, length)
    if t == "RMA":
        return rma(src, length)
    if t == "DEMA":
        return dema(src, length)
    if t == "TEMA":
        return tema(src, length)
    if t == "TMA":
        return tma(src, length)
    if t == "HMA":
        return hma(src, length)
    if t == "LSMA":
        return lsma(src, length)
    if t == "ZLSMA":
        return zlsma(src, length)
    if t == "JMA":
        return jma(src, length, kw.get("jurik_phase", 3), kw.get("jurik_power", 1))
    if t == "MCGINLEY":
        return mcginley(src, length)
    if t == "VAMA":
        return vama(src, length, kw.get("volatility_lookback", 10))
    if t == "KIJUN V2":
        return kijun2(src, length, kw.get("kidiv", 1))
    if t == "MF":
        return mf(src, length, kw.get("beta", 0.8), kw.get("feedback", False), kw.get("z", 0.5))
    if t == "EDSMA":
        return edsma(src, length, kw.get("ssf_length", 20), kw.get("ssf_poles", 2))
    return sma(src, length)


# ---------------------------------------------------------------- volatility etc
def true_range(df):
    pc = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - pc).abs()
    c = (df["low"] - pc).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df, n=14, smoothing="WMA"):
    f = {"RMA": rma, "SMA": sma, "EMA": ema, "WMA": wma}.get(str(smoothing).upper(), wma)
    return f(true_range(df), n)


def crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def cross(a, b):
    return crossover(a, b) | crossunder(a, b)


def bollinger(src, n=20, mult=2.0):
    basis = sma(src, n)
    dev = mult * src.rolling(n).std(ddof=0)
    return basis + dev, basis - dev


def rsi(src, n=14):
    d = src.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(d.clip(upper=0).abs(), n)
    ratio = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + ratio)
    out = out.mask((dn == 0) & (up > 0), 100.0)   # only gains -> overbought
    out = out.mask((up == 0) & (dn > 0), 0.0)     # only losses -> oversold
    return out.fillna(50.0)


def macd(src, fast=12, slow=26, signal=9):
    m = ema(src, fast) - ema(src, slow)
    s = ema(m, signal)
    return m, s, m - s


def dmi(df, di_len=14, adx_sm=14):
    """Wilder's DMI/ADX. Returns (plus_di, minus_di, adx)."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr_r = rma(true_range(df), di_len).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, di_len) / tr_r
    minus_di = 100 * rma(minus_dm, di_len) / tr_r
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return plus_di, minus_di, rma(dx, adx_sm)


def supertrend(df, mult=3.0, n=10):
    """Returns (st_line, direction) where direction < 0 means uptrend (as in Pine)."""
    a = rma(true_range(df), n)
    hl2 = (df["high"] + df["low"]) / 2
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    fu_v, fl_v = ub.to_numpy(copy=True), lb.to_numpy(copy=True)
    close = df["close"].to_numpy(copy=True)
    for i in range(1, len(fu_v)):
        fu_v[i] = fu_v[i] if (fu_v[i] < fu_v[i - 1] or close[i - 1] > fu_v[i - 1]) else fu_v[i - 1]
        fl_v[i] = fl_v[i] if (fl_v[i] > fl_v[i - 1] or close[i - 1] < fl_v[i - 1]) else fl_v[i - 1]
    direction = np.zeros(len(fu_v))
    st = np.zeros(len(fu_v))
    direction[0] = 1
    st[0] = fu_v[0]
    for i in range(1, len(fu_v)):
        if close[i] > fu_v[i]:
            direction[i] = -1
        elif close[i] < fl_v[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
        st[i] = fl_v[i] if direction[i] < 0 else fu_v[i]
    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


def psar(df, start=0.02, inc=0.02, mx=0.2):
    """Parabolic SAR. Returns (sar, is_bullish)."""
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(df)
    sar = np.zeros(n)
    bull = np.zeros(n, dtype=bool)
    if n == 0:
        return pd.Series(sar, index=df.index), pd.Series(bull, index=df.index)
    trend_up = high[0] >= low[0]
    ep = high[0] if trend_up else low[0]
    af = start
    sar[0] = low[0] if trend_up else high[0]
    bull[0] = trend_up
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if trend_up:
            s = min(s, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < s:
                trend_up = False
                s = ep
                ep = low[i]
                af = start
            elif high[i] > ep:
                ep = high[i]
                af = min(af + inc, mx)
        else:
            s = max(s, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > s:
                trend_up = True
                s = ep
                ep = high[i]
                af = start
            elif low[i] < ep:
                ep = low[i]
                af = min(af + inc, mx)
        sar[i] = s
        bull[i] = trend_up
    return pd.Series(sar, index=df.index), pd.Series(bull, index=df.index)


def stoch(df, n=14, k_sm=3, d_sm=3):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    k = sma(100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan), k_sm)
    d = sma(k, d_sm)
    return k, d


def cci(src, n=40):
    m = sma(src, n)

    def _mad(x):
        return np.mean(np.abs(x - x.mean()))

    mad = src.rolling(n).apply(_mad, raw=True)
    return (src - m) / (0.015 * mad.replace(0, np.nan))


def qqe_mod(df, rsi_period=6, sf=5, qqe=3.0, wilders=None, qqe2=1.61):
    """QQE MOD (double RSI QQE, port from the script). Returns (rsi_ma, fast_tl1, fast_tl2)."""
    wilders = wilders or (rsi_period * 2 - 1)

    def _qqe(factor):
        r_ = rsi(df["close"], rsi_period)
        rm = ema(r_, sf)
        atr_rsi = (rm.shift(1) - rm).abs()
        ma_atr = ema(atr_rsi, wilders)
        dar = ema(ma_atr, wilders) * factor
        rs_index = rm
        lb = np.zeros(len(rm))
        sb = np.zeros(len(rm))
        rs_v = rs_index.values
        dar_v = dar.values
        lb_v, sb_v = lb, sb
        for i in range(1, len(rs_v)):
            nlb = rs_v[i] - dar_v[i]
            nsb = rs_v[i] + dar_v[i]
            if not np.isnan(rs_v[i - 1]) and rs_v[i - 1] > lb_v[i - 1] and rs_v[i] > lb_v[i - 1]:
                lb_v[i] = max(lb_v[i - 1], nlb)
            else:
                lb_v[i] = nlb
            if not np.isnan(rs_v[i - 1]) and rs_v[i - 1] < sb_v[i - 1] and rs_v[i] < sb_v[i - 1]:
                sb_v[i] = min(sb_v[i - 1], nsb)
            else:
                sb_v[i] = nsb
        trend_ = np.zeros(len(rs_v))
        trend_[0] = 1
        for i in range(1, len(rs_v)):
            # cross(RSIndex, shortband[1]) -> 1 ; cross(longband[1], RSIndex) -> -1
            up_x = rs_v[i] > sb_v[i - 1] and rs_v[i - 1] <= sb_v[i - 1]
            dn_x = rs_v[i] < lb_v[i - 1] and rs_v[i - 1] >= lb_v[i - 1]
            if up_x:
                trend_[i] = 1
            elif dn_x:
                trend_[i] = -1
            else:
                trend_[i] = trend_[i - 1]
        fast_tl = np.where(np.array(trend_) == 1, lb_v, sb_v)
        return rm, pd.Series(fast_tl, index=df.index)

    rm1, tl1 = _qqe(qqe)
    rm2, tl2 = _qqe(qqe2)
    return rm1, tl1, rm2


def waddah_attar(df, sensitivity=150, fast=20, slow=40, bb_len=20, bb_mult=2.0):
    """WAE. Returns (trend_up, trend_down, explosion_line)."""
    m = macd(df["close"], fast, slow)[0]
    t1 = (m - m.shift(1)) * sensitivity
    bu, bl = bollinger(df["close"], bb_len, bb_mult)
    e1 = bu - bl
    up = t1.clip(lower=0)
    dn = (-t1).clip(lower=0)
    return up, dn, e1
