import os
import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path("data")

# ─── Базовые ТА индикаторы ────────────────────────────────────────────────────
def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    for w in [8, 21, 50, 200]:
        df[f"ema_{w}"] = ta.trend.ema_indicator(c, window=w)
    df["ema_8_21_cross"]  = df["ema_8"]  - df["ema_21"]
    df["ema_21_50_cross"] = df["ema_21"] - df["ema_50"]
    df["price_vs_ema21"]  = (c - df["ema_21"]) / df["ema_21"]
    df["price_vs_ema50"]  = (c - df["ema_50"]) / df["ema_50"]

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["macd_hist_delta"] = df["macd_hist"].diff()   # ускорение гистограммы

    df["adx"] = ta.trend.adx(h, l, c, window=14)
    df["dmp"] = ta.trend.adx_pos(h, l, c, window=14)
    df["dmn"] = ta.trend.adx_neg(h, l, c, window=14)
    df["dmp_dmn_cross"] = df["dmp"] - df["dmn"]

    df["rsi_14"]       = ta.momentum.rsi(c, window=14)
    df["rsi_7"]        = ta.momentum.rsi(c, window=7)
    df["rsi_delta"]    = df["rsi_14"].diff()          # динамика RSI
    df["rsi_14_vs_50"] = df["rsi_14"] - 50           # центрируем

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

    df["cci_20"]   = ta.trend.cci(h, l, c, window=20)
    df["willr_14"] = ta.momentum.williams_r(h, l, c, lbp=14)

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width_delta"] = df["bb_width"].diff()     # сжатие/расширение

    df["atr_14"]     = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_ratio"]  = df["atr_14"] / c              # нормированный ATR

    df["obv"]     = ta.volume.on_balance_volume(c, v)
    df["mfi_14"]  = ta.volume.money_flow_index(h, l, c, v, window=14)
    df["cmf_20"]  = ta.volume.chaikin_money_flow(h, l, c, v, window=20)
    df["vwap"]    = ta.volume.volume_weighted_average_price(h, l, c, v, window=14)
    df["price_vs_vwap"] = (c - df["vwap"]) / df["vwap"]

    return df

# ─── Структура свечи ──────────────────────────────────────────────────────────
def add_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    df["body"]         = c - o
    df["body_abs"]     = df["body"].abs()
    df["upper_wick"]   = h - df[["open", "close"]].max(axis=1)
    df["lower_wick"]   = df[["open", "close"]].min(axis=1) - l
    df["candle_range"] = h - l
    df["body_ratio"]   = df["body_abs"] / df["candle_range"].replace(0, np.nan)
    df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"].replace(0, np.nan)
    df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"].replace(0, np.nan)

    df["is_bullish"]  = (c > o).astype(int)
    df["is_doji"]     = (df["body_ratio"] < 0.1).astype(int)
    df["is_hammer"]   = ((df["lower_wick"] > 2 * df["body_abs"]) & (df["upper_wick"] < df["body_abs"])).astype(int)
    df["is_shooting"] = ((df["upper_wick"] > 2 * df["body_abs"]) & (df["lower_wick"] < df["body_abs"])).astype(int)

    df["vol_ema_20"]  = v.ewm(span=20).mean()
    df["vol_ratio"]   = v / df["vol_ema_20"]
    # Опционально: если taker_buy_base нет в фрейме, не будем падать
    if "taker_buy_base" in df.columns:
        df["taker_ratio"] = df["taker_buy_base"] / v.replace(0, np.nan)
        df["taker_delta"] = df["taker_ratio"].diff()

    return df

# ─── Лаговые фичи и серии ─────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df["volume"]

    for n in [1, 2, 3, 5, 10, 20, 60]:
        df[f"ret_{n}"]     = c.pct_change(n)
        df[f"vol_ret_{n}"] = v.pct_change(n)

    for lag in [1, 2, 3]:
        df[f"rsi_lag_{lag}"]       = df["rsi_14"].shift(lag)
        df[f"macd_hist_lag_{lag}"] = df["macd_hist"].shift(lag)
        df[f"ret_lag_{lag}"]       = df["ret_1"].shift(lag)
        df[f"is_bullish_lag_{lag}"]= df["is_bullish"].shift(lag)

    bullish = df["is_bullish"]
    streak = []
    cur_streak = 0
    for b in bullish:
        if b == 1:
            cur_streak = max(0, cur_streak) + 1
        else:
            cur_streak = min(0, cur_streak) - 1
        streak.append(cur_streak)
    df["candle_streak"] = streak

    prev_body_abs = df["body_abs"].shift(1)
    prev_is_bull  = df["is_bullish"].shift(1)
    df["bullish_engulf"] = ((df["is_bullish"] == 1) & (prev_is_bull == 0) &
                            (df["body_abs"] > prev_body_abs)).astype(int)
    df["bearish_engulf"] = ((df["is_bullish"] == 0) & (prev_is_bull == 1) &
                            (df["body_abs"] > prev_body_abs)).astype(int)

    for w in [10, 20, 60]:
        df[f"roll_std_{w}"]   = c.pct_change().rolling(w).std()
        df[f"roll_skew_{w}"]  = c.pct_change().rolling(w).skew()
        df[f"roll_bull_{w}"]  = bullish.rolling(w).mean()

    return df

# ─── Режим рынка ──────────────────────────────────────────────────────────────
def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    df["regime_trend"] = np.sign(c - df["ema_50"])

    atr_ma = df["atr_14"].rolling(100).mean()
    df["regime_vol_high"] = (df["atr_14"] > atr_ma).astype(int)

    df["regime_sideways"] = (df["adx"] < 20).astype(int)

    df["bb_breakout_up"]   = (c > df["bb_upper"]).astype(int)
    df["bb_breakout_down"] = (c < df["bb_lower"]).astype(int)

    high_20 = df["high"].rolling(20).max()
    low_20  = df["low"].rolling(20).min()
    rng_20  = (high_20 - low_20).replace(0, np.nan)
    df["pos_in_range_20"] = (c - low_20) / rng_20

    return df

# ─── Циклические фичи времени ─────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    hour = idx.hour
    dow = idx.dayofweek
    df["hour"] = hour
    df["day_of_week"] = dow
    df["is_weekend"] = (dow >= 5).astype(int)
    df["hour_sin"]      = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"]     = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]     = np.cos(2 * np.pi * idx.month / 12)

    h = idx.hour
    df["session_asia"]   = ((h >= 0)  & (h < 8)).astype(int)
    df["session_europe"] = ((h >= 8)  & (h < 16)).astype(int)
    df["session_us"]     = ((h >= 14) & (h < 22)).astype(int)
    df["session_overlap"]= ((h >= 14) & (h < 16)).astype(int)
    df["is_eu_open"]     = ((h >= 7)  & (h <= 10)).astype(int)
    df["is_us_open"]     = ((h >= 13) & (h <= 16)).astype(int)
    df["is_overlap"]     = ((h >= 13) & (h <= 15)).astype(int)

    return df


def _completed_htf_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Build completed higher-timeframe candles aligned to 5m timestamps.

    The source index is the 5m candle open timestamp. Using label/right and
    closed/right means a 15m candle stamped 10:00 only becomes available to
    the 5m row stamped 10:00, after that 5m candle has closed in our setup.
    Earlier 5m rows inside that 15m window get the previous completed HTF bar.
    """
    return (
        df.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def add_multitimeframe_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    btc = df_5m.copy()
    if btc.index.tz is not None:
        btc.index = btc.index.tz_convert("UTC").tz_localize(None)

    df_15 = _completed_htf_ohlcv(btc, "15min")
    df_15["mtf_ema8_15m"] = df_15["close"].ewm(span=8, adjust=False).mean()
    df_15["mtf_ema21_15m"] = df_15["close"].ewm(span=21, adjust=False).mean()
    df_15["mtf_rsi_15m"] = ta.momentum.rsi(df_15["close"], window=14)
    df_15["mtf_atr_15m"] = ta.volatility.average_true_range(
        df_15["high"], df_15["low"], df_15["close"], window=14
    )
    df_15["mtf_trend_15m"] = (df_15["mtf_ema8_15m"] > df_15["mtf_ema21_15m"]).astype(int)
    df_15["mtf_trend_str_15m"] = (df_15["mtf_ema8_15m"] - df_15["mtf_ema21_15m"]) / df_15["close"]
    df_15["mtf_rsi_norm_15m"] = (df_15["mtf_rsi_15m"] - 50) / 50
    df_15["mtf_vol_regime_15m"] = (
        df_15["mtf_atr_15m"] > df_15["mtf_atr_15m"].rolling(20).mean()
    ).astype(int)

    df_1h = _completed_htf_ohlcv(btc, "1h")
    df_1h["mtf_ema12_1h"] = df_1h["close"].ewm(span=12, adjust=False).mean()
    df_1h["mtf_ema26_1h"] = df_1h["close"].ewm(span=26, adjust=False).mean()
    df_1h["mtf_rsi_1h"] = ta.momentum.rsi(df_1h["close"], window=14)
    df_1h["mtf_trend_1h"] = (df_1h["mtf_ema12_1h"] > df_1h["mtf_ema26_1h"]).astype(int)
    df_1h["mtf_rsi_norm_1h"] = (df_1h["mtf_rsi_1h"] - 50) / 50
    df_1h["mtf_momentum_1h"] = df_1h["close"].pct_change(4)

    df_4h = _completed_htf_ohlcv(btc, "4h")
    df_4h["mtf_trend_4h"] = (df_4h["close"] > df_4h["close"].shift(2)).astype(int)
    df_4h["mtf_momentum_4h"] = df_4h["close"].pct_change(2)

    mtf_cols_15 = ["mtf_trend_15m", "mtf_trend_str_15m", "mtf_rsi_norm_15m", "mtf_vol_regime_15m"]
    mtf_cols_1h = ["mtf_trend_1h", "mtf_rsi_norm_1h", "mtf_momentum_1h"]
    mtf_cols_4h = ["mtf_trend_4h", "mtf_momentum_4h"]

    out = btc.copy()
    for col in mtf_cols_15:
        out[col] = df_15[col].reindex(out.index, method="ffill")
    for col in mtf_cols_1h:
        out[col] = df_1h[col].reindex(out.index, method="ffill")
    for col in mtf_cols_4h:
        out[col] = df_4h[col].reindex(out.index, method="ffill")

    out["mtf_aligned_15m_1h"] = (out["mtf_trend_15m"] == out["mtf_trend_1h"]).astype(int)
    out["mtf_aligned_all"] = (
        (out["mtf_trend_15m"] == out["mtf_trend_1h"])
        & (out["mtf_trend_1h"] == out["mtf_trend_4h"])
    ).astype(int)
    out["mtf_counter_trend_15m"] = (out["mtf_trend_15m"] != out["mtf_trend_1h"]).astype(int)
    return out


def add_candle_clarity_features(df: pd.DataFrame, threshold_quantile: float = 0.35) -> pd.DataFrame:
    body_move = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)).abs()
    candle_range = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    threshold = body_move.expanding(min_periods=500).quantile(threshold_quantile).shift(1)
    threshold = threshold.fillna(body_move.expanding(min_periods=20).quantile(threshold_quantile).shift(1))
    df["clarity_body_move_pct"] = body_move
    df["clarity_range_pct"] = candle_range
    df["clarity_body_to_range"] = body_move / (candle_range + 1e-10)
    df["clarity_threshold_35q"] = threshold
    df["is_clear_candle"] = (body_move > threshold).astype(int)
    return df

# ─── ETH как опережающий индикатор ───────────────────────────────────────────
def add_eth_features(df_btc: pd.DataFrame, df_eth: pd.DataFrame) -> pd.DataFrame:
    def normalize_idx(df):
        if df.index.tzinfo is not None:
            return df.index.tz_convert("UTC").tz_localize(None)
        return df.index

    btc_idx = normalize_idx(df_btc)
    eth_idx = normalize_idx(df_eth)

    df_btc = df_btc.copy()
    df_btc.index = btc_idx
    df_eth = df_eth.copy()
    df_eth.index = eth_idx

    eth_ret1 = df_eth["close"].pct_change(1).rename("eth_ret_1")
    eth_ret3 = df_eth["close"].pct_change(3).rename("eth_ret_3")

    combined = pd.concat([df_btc["close"].pct_change().rename("btc"), 
                          df_eth["close"].pct_change().rename("eth")], axis=1).dropna()
    eth_btc_corr = combined["btc"].rolling(20).corr(combined["eth"]).rename("eth_btc_corr_20")

    eth_lead = df_eth["close"].pct_change().shift(1).rename("eth_ret_lead1")

    df_out = df_btc.join(eth_ret1, how="left")
    df_out = df_out.join(eth_ret3, how="left")
    df_out = df_out.join(eth_btc_corr, how="left")
    df_out = df_out.join(eth_lead, how="left")

    return df_out

# ─── Futures Metrics (OI, long/short ratio, taker vol) ───────────────────────
def add_futures_metrics(df_btc: pd.DataFrame, df_fm: pd.DataFrame) -> pd.DataFrame:
    """Добавляет OI, long/short ratio топ-трейдеров и taker vol ratio из futures metrics."""
    fm = df_fm.copy()
    if fm.index.tzinfo is not None:
        fm.index = fm.index.tz_convert("UTC").tz_localize(None)

    btc = df_btc.copy()
    if btc.index.tzinfo is not None:
        btc.index = btc.index.tz_convert("UTC").tz_localize(None)

    # Ресэмплируем с 5m на 5m через ffill (данные уже на 5m)
    fm_aligned = fm.reindex(btc.index, method="ffill")

    oi   = fm_aligned["sum_open_interest"].rename("oi")
    oiv  = fm_aligned["sum_open_interest_value"].rename("oi_value")
    tls  = fm_aligned["count_toptrader_long_short_ratio"].rename("top_ls_count")
    tlsr = fm_aligned["sum_toptrader_long_short_ratio"].rename("top_ls_ratio")
    lsr  = fm_aligned["count_long_short_ratio"].rename("ls_ratio")
    tvr  = fm_aligned["sum_taker_long_short_vol_ratio"].rename("taker_vol_ratio")

    # Производные: динамика OI
    oi_chg_1  = oi.pct_change(1).rename("oi_chg_1")
    oi_chg_6  = oi.pct_change(6).rename("oi_chg_6")   # 30m
    oi_chg_12 = oi.pct_change(12).rename("oi_chg_12")  # 1h
    oi_zscore = ((oi - oi.rolling(288).mean()) / (oi.rolling(288).std() + 1e-10)).rename("oi_zscore_24h")

    # Динамика taker vol ratio (>1 = покупатели доминируют)
    tvr_ma6  = tvr.rolling(6).mean().rename("taker_vol_ratio_ma6")
    tvr_delta = tvr.diff(1).rename("taker_vol_ratio_delta")

    # Long/short ratio относительно скользящего среднего
    lsr_vs_ma = (lsr / (lsr.rolling(48).mean() + 1e-10)).rename("ls_ratio_vs_ma48")

    cols = [oi, oiv, tls, tlsr, lsr, tvr,
            oi_chg_1, oi_chg_6, oi_chg_12, oi_zscore,
            tvr_ma6, tvr_delta, lsr_vs_ma]

    result = df_btc.copy()
    result.index = btc.index
    for col in cols:
        result = result.join(col, how="left")
    return result


def add_1m_orderflow_features(df_btc_5m: pd.DataFrame, df_btc_1m: pd.DataFrame) -> pd.DataFrame:
    """Добавляет intrabar order-flow фичи из 1m свечей в 5m Polymarket dataset.

    Это не full aggTrades, но Binance klines уже содержат taker_buy_base/quote,
    что даёт приближение aggressive buy/sell flow внутри каждого 5m окна.
    """
    if df_btc_1m is None or df_btc_1m.empty:
        return df_btc_5m

    btc5 = df_btc_5m.copy()
    one = df_btc_1m.copy()

    if btc5.index.tz is not None:
        btc5.index = btc5.index.tz_convert("UTC").tz_localize(None)
    if one.index.tz is not None:
        one.index = one.index.tz_convert("UTC").tz_localize(None)

    required = {"volume", "taker_buy_base", "taker_buy_quote", "close"}
    missing = required - set(one.columns)
    if missing:
        print(f"1m order-flow: нет колонок {sorted(missing)}, пропускаем")
        return btc5

    one = one.sort_index()
    one["taker_sell_base_1m"] = (one["volume"] - one["taker_buy_base"]).clip(lower=0)
    one["of_buy_ratio_1m"] = one["taker_buy_base"] / one["volume"].replace(0, np.nan)
    one["of_imbalance_1m"] = (
        (one["taker_buy_base"] - one["taker_sell_base_1m"])
        / (one["volume"] + 1e-10)
    )
    one["ret_1m"] = one["close"].pct_change()
    one["signed_taker_vol_1m"] = one["of_imbalance_1m"] * one["volume"]
    one["abs_ret_1m"] = one["ret_1m"].abs()

    agg = pd.DataFrame(index=btc5.index)
    grouped = one.resample("5min", label="left", closed="left")
    agg["of_taker_buy_5m"] = grouped["taker_buy_base"].sum()
    agg["of_taker_sell_5m"] = grouped["taker_sell_base_1m"].sum()
    agg["of_volume_1m_sum_5m"] = grouped["volume"].sum()
    agg["of_signed_taker_vol_5m"] = grouped["signed_taker_vol_1m"].sum()
    agg["of_imbalance_5m"] = (
        (agg["of_taker_buy_5m"] - agg["of_taker_sell_5m"])
        / (agg["of_volume_1m_sum_5m"] + 1e-10)
    )
    agg["of_buy_ratio_5m"] = agg["of_taker_buy_5m"] / (agg["of_volume_1m_sum_5m"] + 1e-10)
    agg["of_imbalance_mean_1m"] = grouped["of_imbalance_1m"].mean()
    agg["of_imbalance_std_1m"] = grouped["of_imbalance_1m"].std()
    agg["of_imbalance_last_1m"] = grouped["of_imbalance_1m"].last()
    agg["of_imbalance_slope_5m"] = grouped["of_imbalance_1m"].apply(
        lambda s: np.polyfit(np.arange(len(s)), s.fillna(0).values, 1)[0] if len(s) >= 2 else 0.0
    )
    agg["of_trade_intensity_5m"] = grouped["volume"].count()
    agg["of_abs_ret_sum_5m"] = grouped["abs_ret_1m"].sum()
    agg["of_ret_direction_5m"] = grouped["ret_1m"].sum()
    agg["of_flow_price_align_5m"] = agg["of_imbalance_5m"] * np.sign(agg["of_ret_direction_5m"])

    for w in [3, 6, 12]:
        agg[f"of_imbalance_ma_{w}"] = agg["of_imbalance_5m"].rolling(w).mean()
        agg[f"of_imbalance_delta_{w}"] = agg["of_imbalance_5m"] - agg[f"of_imbalance_ma_{w}"]
        agg[f"of_signed_vol_ma_{w}"] = agg["of_signed_taker_vol_5m"].rolling(w).mean()

    return btc5.join(agg, how="left")


# ─── Polymarket-Специфичные фичи ──────────────────────────────────────────────
def add_strike_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    
    # Психологические уровни (страйки) по плану
    for level in [500, 1000, 2500]:
        strike = (close / level).round() * level
        df[f"dist_strike_{level}"] = (close - strike) / close             # нормализованное расстояние
        df[f"dist_abs_{level}"] = df[f"dist_strike_{level}"].abs()        # насколько далеко от уровня
        df[f"above_strike_{level}"] = (close > strike).astype(int)        # выше или ниже
        
    return df

def add_additional_poly_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_dollar"] = df["volume"] * df["close"]
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["gap_open"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)
    return df

def add_poly_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Таргеты: y_1 (5m), y_3 (15m), y_6 (30m), y_12 (1h). P(close[t+N] > close[t])"""
    for n in [1, 3, 6, 12]:
        df[f"y_{n}"] = (df["close"].shift(-n) > df["close"]).astype(int)
    return df

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=== Feature Engineering BTC 5m for Polymarket ===\n")

    print("Загружаю 5м BTC...")
    df = pd.read_parquet(DATA_DIR / "btc_5m.parquet")
    print(f"Строк: {len(df)}, период: {df.index[0]} — {df.index[-1]}")

    print("Добавляю TA индикаторы...")
    df = add_ta_features(df)

    print("Добавляю структуру свечей...")
    df = add_candle_structure(df)

    print("Добавляю лаги и серии...")
    df = add_lag_features(df)

    print("Добавляю режим рынка...")
    df = add_market_regime(df)

    print("Добавляю циклические фичи времени...")
    df = add_time_features(df)

    print("Добавляю MTF фичи (15m/1h/4h)...")
    df = add_multitimeframe_features(df)

    print("Добавляю candle clarity фичи...")
    df = add_candle_clarity_features(df)

    # ETH
    eth_path = DATA_DIR / "eth_5m.parquet"
    if eth_path.exists():
        print("Добавляю ETH фичи...")
        df_eth = pd.read_parquet(eth_path)
        df = add_eth_features(df, df_eth)
    else:
        print("ETH данных нет, пропускаем")

    # Futures metrics (OI, long/short, taker vol)
    fm_path = DATA_DIR / "btc_futures_metrics.parquet"
    if fm_path.exists():
        print("Добавляю futures metrics (OI, long/short ratio, taker vol)...")
        df_fm = pd.read_parquet(fm_path)
        df = add_futures_metrics(df, df_fm)
    else:
        print("Futures metrics не найдены, пропускаем")

    # 1m order-flow / taker-flow фичи
    env_orderflow = os.getenv("POLY_ORDERFLOW_PATH")
    orderflow_candidates = ([Path(env_orderflow)] if env_orderflow else []) + [
        DATA_DIR / "btc_1m_full.parquet",
        Path("training/model_1m/data/btc_1m_full.parquet"),
        DATA_DIR / "btc_1m.parquet",
    ]
    orderflow_path = next((p for p in orderflow_candidates if p.exists()), None)
    if orderflow_path:
        print(f"Добавляю 1m order-flow фичи из {orderflow_path}...")
        df_1m = pd.read_parquet(orderflow_path)
        df = add_1m_orderflow_features(df, df_1m)
    else:
        print("1m order-flow данные не найдены, пропускаем")

    print("Добавляю специфичные фичи Polymarket (страйки, vol_dollar)...")
    df = add_strike_features(df)
    df = add_additional_poly_features(df)

    print("Добавляю 4 таргета (1, 3, 6, 12)...")
    df = add_poly_targets(df)

    # Убираем строки без финального таргета (последние 12) и с большим кол-вом NaN
    df = df.dropna(subset=["y_12"])
    df = df.dropna(thresh=int(len(df.columns) * 0.6))

    out_path = Path(os.getenv("POLY_FEATURE_OUT", DATA_DIR / "btc_5m_poly_features.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    NON_FEATURES = {"close_time","open","high","low","close","quote_volume",
                    "ignore","y_1","y_3","y_6","y_12","taker_buy_base","taker_buy_quote",
                    "target"} # target from old might accidentally end up here but just in case
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]

    print(f"\n✓ Готово: {out_path}")
    print(f"  Строк: {len(df)}, фич: {len(feature_cols)}")
    print(f"  Баланс y_1: {df['y_1'].value_counts(normalize=True).to_dict()}")
    print(f"  Баланс y_12: {df['y_12'].value_counts(normalize=True).to_dict()}")

if __name__ == "__main__":
    main()
