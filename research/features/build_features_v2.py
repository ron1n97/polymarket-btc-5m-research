"""
Feature engineering v2 — улучшенный датасет для обучения модели.

Добавлено по сравнению с v1:
- Лаговые таргеты и серии свечей
- Режим рынка (тренд/боковик/волатильность)
- Циклические фичи времени (sin/cos)
- ETH как опережающий индикатор
- Межсвечные паттерны (доджи, молот, поглощение)
- Фичи на основе предыдущих N свечей (rolling stats)
- Правильное разделение: никаких данных из будущего
"""

import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path("data")


# ─── Базовые ТА индикаторы ────────────────────────────────────────────────────

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # EMA
    for w in [8, 21, 50, 200]:
        df[f"ema_{w}"] = ta.trend.ema_indicator(c, window=w)
    df["ema_8_21_cross"]  = df["ema_8"]  - df["ema_21"]
    df["ema_21_50_cross"] = df["ema_21"] - df["ema_50"]
    df["price_vs_ema21"]  = (c - df["ema_21"]) / df["ema_21"]
    df["price_vs_ema50"]  = (c - df["ema_50"]) / df["ema_50"]

    # MACD
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["macd_hist_delta"] = df["macd_hist"].diff()   # ускорение гистограммы

    # ADX
    df["adx"] = ta.trend.adx(h, l, c, window=14)
    df["dmp"] = ta.trend.adx_pos(h, l, c, window=14)
    df["dmn"] = ta.trend.adx_neg(h, l, c, window=14)
    df["dmp_dmn_cross"] = df["dmp"] - df["dmn"]

    # RSI
    df["rsi_14"]       = ta.momentum.rsi(c, window=14)
    df["rsi_7"]        = ta.momentum.rsi(c, window=7)
    df["rsi_delta"]    = df["rsi_14"].diff()          # динамика RSI
    df["rsi_14_vs_50"] = df["rsi_14"] - 50           # центрируем

    # Stoch
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

    df["cci_20"]   = ta.trend.cci(h, l, c, window=20)
    df["willr_14"] = ta.momentum.williams_r(h, l, c, lbp=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width_delta"] = df["bb_width"].diff()     # сжатие/расширение

    # ATR
    df["atr_14"]     = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_ratio"]  = df["atr_14"] / c              # нормированный ATR

    # Volume
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

    # Паттерны свечей (бинарные)
    df["is_bullish"]  = (c > o).astype(int)
    df["is_doji"]     = (df["body_ratio"] < 0.1).astype(int)
    df["is_hammer"]   = ((df["lower_wick"] > 2 * df["body_abs"]) & (df["upper_wick"] < df["body_abs"])).astype(int)
    df["is_shooting"] = ((df["upper_wick"] > 2 * df["body_abs"]) & (df["lower_wick"] < df["body_abs"])).astype(int)

    # Объём
    df["vol_ema_20"]  = v.ewm(span=20).mean()
    df["vol_ratio"]   = v / df["vol_ema_20"]
    df["taker_ratio"] = df["taker_buy_base"] / v.replace(0, np.nan)
    df["taker_delta"] = df["taker_ratio"].diff()

    return df


# ─── Лаговые фичи и серии ─────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df["volume"]

    # Доходности за N свечей
    for n in [1, 2, 3, 5, 10, 20, 60]:
        df[f"ret_{n}"]     = c.pct_change(n)
        df[f"vol_ret_{n}"] = v.pct_change(n)

    # Лаговые значения RSI и MACD (был ли перекуплен/перепродан)
    for lag in [1, 2, 3]:
        df[f"rsi_lag_{lag}"]      = df["rsi_14"].shift(lag)
        df[f"macd_hist_lag_{lag}"] = df["macd_hist"].shift(lag)
        df[f"ret_lag_{lag}"]      = df["ret_1"].shift(lag)
        df[f"is_bullish_lag_{lag}"] = df["is_bullish"].shift(lag)

    # Серия: сколько бычьих/медвежьих свечей подряд
    bullish = df["is_bullish"]
    streak = []
    cur_streak = 0
    for b in bullish:
        if b == 1:
            cur_streak = max(0, cur_streak) + 1
        else:
            cur_streak = min(0, cur_streak) - 1
        streak.append(cur_streak)
    df["candle_streak"] = streak   # +3 = три бычьих подряд, -2 = две медвежьих

    # Поглощение (текущая свеча поглощает предыдущую)
    prev_body_abs = df["body_abs"].shift(1)
    prev_is_bull  = df["is_bullish"].shift(1)
    df["bullish_engulf"] = ((df["is_bullish"] == 1) & (prev_is_bull == 0) &
                            (df["body_abs"] > prev_body_abs)).astype(int)
    df["bearish_engulf"] = ((df["is_bullish"] == 0) & (prev_is_bull == 1) &
                            (df["body_abs"] > prev_body_abs)).astype(int)

    # Rolling статистики закрытий
    for w in [10, 20, 60]:
        df[f"roll_std_{w}"]   = c.pct_change().rolling(w).std()      # волатильность
        df[f"roll_skew_{w}"]  = c.pct_change().rolling(w).skew()     # асимметрия
        df[f"roll_bull_{w}"]  = bullish.rolling(w).mean()            # % бычьих свечей

    return df


# ─── Режим рынка ──────────────────────────────────────────────────────────────

def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]

    # Тренд: цена выше/ниже EMA50
    df["regime_trend"] = np.sign(c - df["ema_50"])

    # Волатильность: ATR высокий или низкий относительно своей истории
    atr_ma = df["atr_14"].rolling(100).mean()
    df["regime_vol_high"] = (df["atr_14"] > atr_ma).astype(int)

    # Боковик: ADX < 20 (слабый тренд)
    df["regime_sideways"] = (df["adx"] < 20).astype(int)

    # Пробой Боллинджера
    df["bb_breakout_up"]   = (c > df["bb_upper"]).astype(int)
    df["bb_breakout_down"] = (c < df["bb_lower"]).astype(int)

    # Расстояние от 20-дневного хая/лоя (нормализованное)
    high_20 = df["high"].rolling(20).max()
    low_20  = df["low"].rolling(20).min()
    rng_20  = (high_20 - low_20).replace(0, np.nan)
    df["pos_in_range_20"] = (c - low_20) / rng_20   # 0=на лое, 1=на хае

    return df


# ─── Циклические фичи времени ─────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    # sin/cos вместо линейных — корректно кодирует цикличность
    df["hour_sin"]      = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"]     = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]     = np.cos(2 * np.pi * idx.month / 12)

    # Азиатская/европейская/американская сессия
    h = idx.hour
    df["session_asia"]   = ((h >= 0)  & (h < 8)).astype(int)
    df["session_europe"] = ((h >= 8)  & (h < 16)).astype(int)
    df["session_us"]     = ((h >= 14) & (h < 22)).astype(int)
    df["session_overlap"] = ((h >= 14) & (h < 16)).astype(int)  # EU+US overlap

    return df


# ─── ETH как опережающий индикатор ───────────────────────────────────────────

def add_eth_features(df_btc: pd.DataFrame, df_eth: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем индекс
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

    # ETH доходность за 1 и 3 свечи
    eth_ret1 = df_eth["close"].pct_change(1).rename("eth_ret_1")
    eth_ret3 = df_eth["close"].pct_change(3).rename("eth_ret_3")

    # ETH-BTC корреляция за 20 свечей (divergence сигнал)
    btc_ret = df_btc["close"].pct_change()
    eth_ret = df_eth["close"].pct_change()

    # Rolling корреляция
    combined = pd.concat([btc_ret.rename("btc"), eth_ret.rename("eth")], axis=1).dropna()
    eth_btc_corr = combined["btc"].rolling(20).corr(combined["eth"]).rename("eth_btc_corr_20")

    # ETH опережает BTC? (ETH ret lag -1 → предсказывает BTC)
    eth_lead = df_eth["close"].pct_change().shift(1).rename("eth_ret_lead1")

    df_out = df_btc.join(eth_ret1, how="left")
    df_out = df_out.join(eth_ret3, how="left")
    df_out = df_out.join(eth_btc_corr, how="left")
    df_out = df_out.join(eth_lead, how="left")

    return df_out


# ─── 1м фичи ─────────────────────────────────────────────────────────────────

def add_1m_features(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    print("Добавляю 1м фичи...")

    def norm_idx(df):
        if df.index.tzinfo is not None:
            return df.index.tz_convert("UTC").tz_localize(None)
        return df.index

    df_5m = df_5m.copy()
    df_1m = df_1m.copy()
    df_5m.index = norm_idx(df_5m)
    df_1m.index = norm_idx(df_1m)

    results = []
    for ts in df_5m.index:
        end_ts = ts + pd.Timedelta(minutes=5)
        minutes = df_1m[(df_1m.index >= ts) & (df_1m.index < end_ts)]
        if len(minutes) < 3:
            results.append({"ts": ts})
            continue
        vol = minutes["volume"].values
        ret = minutes["close"].pct_change().values
        last = minutes.iloc[-1]
        first = minutes.iloc[0]
        vol_slope = np.polyfit(range(len(vol)), vol, 1)[0] if len(vol) > 1 else 0

        results.append({
            "ts":                    ts,
            "m1_vol_slope":          vol_slope,
            "m1_last_ret":           (last["close"] - last["open"]) / (last["open"] + 1e-9),
            "m1_last_vol_ratio":     vol[-1] / (vol.mean() + 1e-9),
            "m1_taker_ratio_last":   last["taker_buy_base"] / (last["volume"] + 1e-9),
            "m1_open_to_now":        (last["close"] - first["open"]) / (first["open"] + 1e-9),
            "m1_vol_acceleration":   (vol[-1] - vol[0]) / (vol.mean() + 1e-9),
            "m1_ret_std":            float(np.std(ret[1:])) if len(ret) > 1 else 0,
        })

    df_1m_feat = pd.DataFrame(results).set_index("ts")
    return df_5m.join(df_1m_feat, how="left")


# ─── Целевая переменная ───────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """target=1 если следующая свеча закрылась выше открытия."""
    next_close = df["close"].shift(-1)
    next_open  = df["open"].shift(-1)
    df["target"] = (next_close > next_open).astype(int)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Feature Engineering v2 ===\n")

    print("Загружаю 5м BTC...")
    df = pd.read_parquet(DATA_DIR / "btc_5m.parquet")
    print(f"Строк: {len(df)}, период: {df.index[0]} — {df.index[-1]}")

    print("TA индикаторы...")
    df = add_ta_features(df)

    print("Структура свечей...")
    df = add_candle_structure(df)

    print("Лаги и серии...")
    df = add_lag_features(df)

    print("Режим рынка...")
    df = add_market_regime(df)

    print("Время (sin/cos)...")
    df = add_time_features(df)

    # ETH
    eth_path = DATA_DIR / "eth_5m.parquet"
    if eth_path.exists():
        print("ETH фичи...")
        df_eth = pd.read_parquet(eth_path)
        df = add_eth_features(df, df_eth)
    else:
        print("ETH данных нет, пропускаем")

    # 1м фичи
    path_1m = DATA_DIR / "btc_1m.parquet"
    if path_1m.exists():
        df_1m = pd.read_parquet(path_1m)
        df = add_1m_features(df, df_1m)
    else:
        print("1м данных нет, пропускаем")

    print("Таргет...")
    df = add_target(df)

    # Убираем строки без таргета (последняя) и с большим кол-вом NaN
    df = df.dropna(subset=["target"])
    df = df.dropna(thresh=int(len(df.columns) * 0.6))

    out_path = DATA_DIR / "btc_features_v2.parquet"
    df.to_parquet(out_path)

    NON_FEATURES = {"close_time","open","high","low","close","quote_volume",
                    "ignore","target","taker_buy_base","taker_buy_quote"}
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]

    print(f"\n✓ Готово: {out_path}")
    print(f"  Строк: {len(df)}, фич: {len(feature_cols)} (было 59)")
    print(f"  Баланс: {df['target'].value_counts(normalize=True).to_dict()}")

    # Топ корреляций с таргетом
    corr = df[feature_cols + ["target"]].corr()["target"].drop("target").abs()
    print(f"\nТоп-15 корреляций с target:")
    print(corr.sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
