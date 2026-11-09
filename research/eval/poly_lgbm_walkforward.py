#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"

HORIZONS = [1, 3, 6, 12]
NON_FEATURES = {
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "ignore",
    "y_1",
    "y_3",
    "y_6",
    "y_12",
    "taker_buy_base",
    "taker_buy_quote",
    "target",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURES]
    numeric = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
    return numeric


def profit_factor(pnl: np.ndarray) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def evaluate_probs(proba: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    long_mask = proba >= threshold
    short_mask = proba <= (1.0 - threshold)
    mask = long_mask | short_mask
    if not mask.any():
        return {
            "threshold": threshold,
            "trades": 0,
            "winrate": np.nan,
            "pnl_even": 0.0,
            "profit_factor": np.nan,
            "coverage": 0.0,
        }
    pred = long_mask[mask].astype(int)
    wins = pred == y[mask]
    pnl = np.where(wins, 0.5, -0.5)
    return {
        "threshold": threshold,
        "trades": int(mask.sum()),
        "winrate": float(wins.mean()),
        "pnl_even": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "coverage": float(mask.mean()),
    }


def month_splits(df: pd.DataFrame, min_train_months: int, gap_candles: int):
    periods = pd.DatetimeIndex(df.index).to_period("M")
    months = periods.unique()
    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]
        train = df[periods.isin(train_months)].copy()
        test = df[periods == test_month].copy()
        if gap_candles > 0 and len(train) > gap_candles:
            train = train.iloc[:-gap_candles]
        yield str(test_month), train, test


def fit_lgbm(train: pd.DataFrame, features: list[str], target: str) -> lgb.LGBMClassifier:
    split = max(1, int(len(train) * 0.90))
    tr = train.iloc[:split]
    val = train.iloc[split:]
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=60,
        subsample=0.75,
        subsample_freq=5,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(50, verbose=False)]
    model.fit(
        tr[features],
        tr[target],
        eval_set=[(val[features], val[target])],
        eval_metric="auc",
        callbacks=callbacks,
    )
    return model


def run_walk_forward(
    data_path: Path,
    min_train_months: int,
    gap_candles: int,
    clear_only_train: bool,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(data_path).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    features = feature_columns(df)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[f"y_{h}" for h in HORIZONS])

    rows = []
    for test_period, train, test in month_splits(df, min_train_months, gap_candles):
        if len(train) < 8_000 or len(test) < 1_000:
            continue
        train_fit = train
        if clear_only_train and "is_clear_candle" in train_fit.columns:
            clear = train_fit[train_fit["is_clear_candle"] == 1]
            if len(clear) >= 5_000:
                train_fit = clear

        for horizon in HORIZONS:
            target = f"y_{horizon}"
            if train_fit[target].nunique() < 2 or test[target].nunique() < 2:
                continue
            model = fit_lgbm(train_fit, features, target)
            proba = model.predict_proba(test[features])[:, 1]
            y = test[target].astype(int).to_numpy()
            auc = float(roc_auc_score(y, proba))

            for threshold in [0.50, 0.53, 0.54, 0.55, 0.56, 0.58]:
                metrics = evaluate_probs(proba, y, threshold)
                rows.append(
                    {
                        "dataset": data_path.name,
                        "test_period": test_period,
                        "horizon": f"y_{horizon}",
                        "train_rows": len(train_fit),
                        "test_rows": len(test),
                        "features": len(features),
                        "clear_only_train": clear_only_train,
                        "auc": auc,
                        **metrics,
                    }
                )

            if "is_clear_candle" in test.columns:
                clear_mask = test["is_clear_candle"].fillna(0).astype(int).to_numpy() == 1
                if clear_mask.sum() >= 200:
                    metrics = evaluate_probs(proba[clear_mask], y[clear_mask], 0.53)
                    rows.append(
                        {
                            "dataset": data_path.name,
                            "test_period": test_period,
                            "horizon": f"y_{horizon}",
                            "train_rows": len(train_fit),
                            "test_rows": int(clear_mask.sum()),
                            "features": len(features),
                            "clear_only_train": clear_only_train,
                            "auc": float(roc_auc_score(y[clear_mask], proba[clear_mask]))
                            if len(set(y[clear_mask])) > 1
                            else np.nan,
                            **metrics,
                            "threshold": "0.53_clear_test",
                        }
                    )

    return pd.DataFrame(rows), features


def render_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows."]
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if pd.isna(val) else f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/btc_5m_poly_features_mtf.parquet")
    parser.add_argument("--name", default="mtf")
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--gap-candles", type=int, default=100)
    parser.add_argument("--clear-only-train", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    results, features = run_walk_forward(
        data_path=args.data,
        min_train_months=args.min_train_months,
        gap_candles=args.gap_candles,
        clear_only_train=args.clear_only_train,
    )
    suffix = f"{args.name}{'_clear_train' if args.clear_only_train else ''}"
    out_csv = LOG_DIR / f"poly_lgbm_walkforward_{suffix}.csv"
    results.to_csv(out_csv, index=False)

    report = LOG_DIR / f"poly_lgbm_walkforward_{suffix}.md"
    lines = [
        "# Polymarket LGBM Walk-Forward",
        "",
        f"Dataset: `{args.data}`",
        f"Rows: `{len(pd.read_parquet(args.data))}`",
        f"Features: `{len(features)}`",
        f"Min train months: `{args.min_train_months}`",
        f"Gap candles: `{args.gap_candles}`",
        f"Clear-only train: `{args.clear_only_train}`",
        f"CSV: `{out_csv}`",
        "",
    ]

    if not results.empty:
        regular = results[~results["threshold"].astype(str).str.contains("clear_test")].copy()
        agg = regular.groupby(["horizon", "threshold"]).agg(
            folds=("test_period", "nunique"),
            trades=("trades", "sum"),
            winrate=("winrate", "mean"),
            pnl_even=("pnl_even", "sum"),
            avg_pf=("profit_factor", "mean"),
            avg_auc=("auc", "mean"),
            avg_coverage=("coverage", "mean"),
        ).reset_index()
        lines += ["## Aggregate", ""]
        lines.extend(render_table(agg.sort_values(["horizon", "threshold"])))

        lines += ["", "## Best By PnL", ""]
        lines.extend(render_table(agg.sort_values("pnl_even", ascending=False).head(20)))

        period = regular.groupby(["test_period", "horizon"]).agg(
            avg_auc=("auc", "mean"),
            pnl_053=("pnl_even", lambda x: x.iloc[0] if len(x) else np.nan),
        ).reset_index()
        lines += ["", "## Period AUC", ""]
        lines.extend(render_table(period.head(80)))

    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved report: {report}")
    if not results.empty:
        regular = results[~results["threshold"].astype(str).str.contains("clear_test")].copy()
        agg = regular.groupby(["horizon", "threshold"]).agg(
            folds=("test_period", "nunique"),
            trades=("trades", "sum"),
            winrate=("winrate", "mean"),
            pnl_even=("pnl_even", "sum"),
            avg_auc=("auc", "mean"),
            avg_coverage=("coverage", "mean"),
        ).reset_index()
        print(agg.sort_values("pnl_even", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
