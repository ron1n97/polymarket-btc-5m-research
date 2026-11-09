"""
Feature engineering для предсказания направления 5м свечи BTC.
Читает сырые parquet, добавляет технические индикаторы и 1м-фичи,
сохраняет готовый датасет для обучения.
"""

import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path("data")


# ─── Технические индикаторы на 5м ────────────────────────────────────────────

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить технические индикаторы на основе OHLCV."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # Трендовые
    df["ema_8"]   = ta.trend.ema_indicator(c, window=8)
    df["ema_21"]  = ta.trend.ema_indicator(c, window=21)
    df["ema_50"]  = ta.trend.ema_indicator(c, window=50)
    df["ema_200"] = ta.trend.ema_indicator(c, window=200)

    df["ema_8_21_cross"]  = df["ema_8"]  - df["ema_21"]
    df["ema_21_50_cross"] = df["ema_21"] - df["ema_50"]

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    df["adx"] = ta.trend.adx(h, l, c, window=14)
    df["dmp"] = ta.trend.adx_pos(h, l, c, window=14)
    df["dmn"] = ta.trend.adx_neg(h, l, c, window=14)

    # Осцилляторы
    df["rsi_14"] = ta.momentum.rsi(c, window=14)
    df["rsi_7"]  = ta.momentum.rsi(c, window=7)

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["cci_20"]  = ta.trend.cci(h, l, c, window=20)
    df["willr_14"] = ta.momentum.williams_r(h, l, c, lbp=14)

    # Волатильность
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]    = bb.bollinger_pband()   # где цена внутри полос [0..1]

    df["atr_14"] = ta.volatility.average_true_range(h, l, c, window=14)

    # Объёмные
    df["obv"]   = ta.volume.on_balance_volume(c, v)
    df["vwap"]  = ta.volume.volume_weighted_average_price(h, l, c, v, window=14)
    df["mfi_14"] = ta.volume.money_flow_index(h, l, c, v, window=14)
    df["cmf_20"] = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    # Структура свечи
    df["body"]        = c - df["open"]
    df["body_abs"]    = df["body"].abs()
    df["upper_wick"]  = h - df[["open", "close"]].max(axis=1)
    df["lower_wick"]  = df[["open", "close"]].min(axis=1) - l
    df["candle_range"] = h - l
    df["body_ratio"]  = df["body_abs"] / df["candle_range"].replace(0, np.nan)

    # Изменения цены
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}"]    = c.pct_change(n)
        df[f"vol_ret_{n}"] = v.pct_change(n)

    # Отношение объёма к EMA объёма
    df["vol_ema_20"] = v.ewm(span=20).mean()
    df["vol_ratio"]  = v / df["vol_ema_20"]

    # Taker buy ratio (агрессивные покупки)
    df["taker_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)

    # Час и день недели (сезонность)
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek

    return df


# ─── 1м фичи внутри 5м свечи ─────────────────────────────────────────────────

def add_1m_features(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой 5м свечи смотрим последние 5 минутных свечей (внутри неё).
    Добавляем: momentum последней минуты, паттерн объёма, дисбаланс.
    """
    print("Обогащение 1м фичами...")

    # 1м индекс как timestamp в UTC
    df_1m = df_1m.copy()
    df_1m.index = df_1m.index.tz_localize(None) if df_1m.index.tzinfo is None else df_1m.index.tz_convert(None)
    df_5m_idx = df_5m.index.tz_localize(None) if df_5m.index.tzinfo is None else df_5m.index.tz_convert(None)

    # Для каждой 5м свечи [t, t+5m) агрегируем 1м свечи
    results = []
    for ts in df_5m_idx:
        end_ts = ts + pd.Timedelta(minutes=5)
        mask = (df_1m.index >= ts) & (df_1m.index < end_ts)
        minutes = df_1m[mask]
        if len(minutes) < 3:
            results.append({"ts": ts})
            continue

        vol = minutes["volume"].values
        ret = minutes["close"].pct_change().values

        # Объём нарастал или убывал к концу свечи?
        vol_slope = np.polyfit(range(len(vol)), vol, 1)[0] if len(vol) > 1 else 0

        # Последняя минута внутри 5м свечи
        last_min = minutes.iloc[-1]
        first_min = minutes.iloc[0]

        results.append({
            "ts": ts,
            "m1_vol_slope":        vol_slope,
            "m1_last_ret":         (last_min["close"] - last_min["open"]) / last_min["open"] if last_min["open"] else 0,
            "m1_last_vol_ratio":   vol[-1] / vol.mean() if vol.mean() else 1,
            "m1_taker_ratio_last": last_min["taker_buy_base"] / last_min["volume"] if last_min["volume"] else 0.5,
            "m1_open_to_now":      (last_min["close"] - first_min["open"]) / first_min["open"] if first_min["open"] else 0,
            "m1_vol_acceleration": (vol[-1] - vol[0]) / (vol.mean() + 1e-9),
        })

    df_1m_feat = pd.DataFrame(results).set_index("ts")
    # Совмещаем индексы
    df_5m.index = df_5m_idx
    df_out = df_5m.join(df_1m_feat, how="left")
    return df_out


# ─── Целевая переменная ───────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    target = 1, если следующая 5м свеча закрылась выше открытия (бычья).
    target = 0, если медвежья.
    """
    next_close = df["close"].shift(-1)
    next_open  = df["open"].shift(-1)
    df["target"] = (next_close > next_open).astype(int)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Загружаю 5м данные...")
    df_5m = pd.read_parquet(DATA_DIR / "btc_5m.parquet")
    print(f"5м свечей: {len(df_5m)}, период: {df_5m.index[0]} — {df_5m.index[-1]}")

    df_5m = add_ta_features(df_5m)

    # Если есть 1м данные — добавляем фичи
    path_1m = DATA_DIR / "btc_1m.parquet"
    if path_1m.exists():
        df_1m = pd.read_parquet(path_1m)
        df_5m = add_1m_features(df_5m, df_1m)
    else:
        print("1м данных нет, пропускаем 1м фичи")

    df_5m = add_target(df_5m)

    # Убираем последнюю строку (нет таргета) и NaN от индикаторов
    df_5m = df_5m.dropna(subset=["target"])
    df_5m = df_5m.dropna(thresh=int(len(df_5m.columns) * 0.7))

    # Сохраняем
    out_path = DATA_DIR / "btc_features.parquet"
    df_5m.to_parquet(out_path)
    print(f"\nГотовый датасет: {out_path}")
    print(f"Строк: {len(df_5m)}, колонок: {len(df_5m.columns)}")
    print(f"Баланс классов:\n{df_5m['target'].value_counts(normalize=True)}")
    print(f"\nФичи:\n{[c for c in df_5m.columns if c != 'target']}")


if __name__ == "__main__":
    main()
