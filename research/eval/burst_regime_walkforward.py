#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

NON_FEATURES = {
    "open",
    "high",
    "low",
    "close",
    "close_time",
    "quote_volume",
    "ignore",
    "y_1",
    "y_3",
    "y_6",
    "y_12",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in NON_FEATURES:
            continue
        if any(k in col.lower() for k in ("target", "future", "label", "burst_")):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def month_splits(df: pd.DataFrame, min_train_months: int):
    periods = pd.DatetimeIndex(df.index).to_period("M")
    months = periods.unique()
    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]
        train = df[periods.isin(train_months)].copy()
        test = df[periods == test_month].copy()
        yield str(test_month), train, test


def fit_model(train_x: pd.DataFrame, train_y: pd.Series) -> lgb.LGBMClassifier:
    split = max(1, int(len(train_x) * 0.9))
    x_tr, x_val = train_x.iloc[:split], train_x.iloc[split:]
    y_tr, y_val = train_y.iloc[:split], train_y.iloc[split:]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=5,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = values - peak
    return float(dd.min())


def _parse_ts(val: str):
    s = (val or "").strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        try:
            d = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    else:
        d = d.astimezone(timezone.utc)
    return d


def estimate_empirical_proxy(
    quotes_glob: str,
    resolved_glob: str,
    entry_seconds: tuple[int, ...] = (60, 120, 180, 240),
) -> dict[str, float]:
    resolved: dict[str, str] = {}
    for fp in sorted(glob.glob(resolved_glob)):
        try:
            with open(fp, newline="") as f:
                for row in csv.DictReader(f):
                    slug = row.get("slug", "")
                    actual = row.get("actual", "")
                    if slug and actual in {"Up", "Down"}:
                        resolved[slug] = actual
        except Exception:
            continue

    quotes: dict[str, list[tuple[datetime, datetime, float, float]]] = {}
    for fp in sorted(glob.glob(quotes_glob)):
        if "resolved" in fp:
            continue
        try:
            with open(fp, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("status") != "ok":
                        continue
                    slug = row.get("slug", "")
                    if slug not in resolved:
                        continue
                    ts = _parse_ts(row.get("ts", ""))
                    w = _parse_ts(row.get("window", ""))
                    if ts is None or w is None:
                        continue
                    try:
                        up = float(row.get("up_ask", "nan"))
                        dn = float(row.get("down_ask", "nan"))
                    except Exception:
                        continue
                    if not np.isfinite(up) or not np.isfinite(dn):
                        continue
                    quotes.setdefault(slug, []).append((ts, w, up, dn))
        except Exception:
            continue

    for slug in list(quotes):
        quotes[slug] = sorted(quotes[slug], key=lambda x: x[0])

    def first_after(arr, target, end):
        for ts, w, up, dn in arr:
            if ts < target:
                continue
            if ts > end:
                break
            return ts, up, dn
        return None

    up_win: list[float] = []
    up_loss: list[float] = []
    dn_win: list[float] = []
    dn_loss: list[float] = []
    for slug, actual in resolved.items():
        arr = quotes.get(slug)
        if not arr:
            continue
        w = arr[0][1]
        end = w + timedelta(minutes=5)
        for sec in entry_seconds:
            q = first_after(arr, w + timedelta(seconds=sec), end)
            if q is None:
                continue
            _, up, dn = q
            if 0 < up < 1:
                r = (1.0 / up - 1.0) if actual == "Up" else -1.0
                (up_win if actual == "Up" else up_loss).append(r)
            if 0 < dn < 1:
                r = (1.0 / dn - 1.0) if actual == "Down" else -1.0
                (dn_win if actual == "Down" else dn_loss).append(r)

    def mean_or(vals: list[float], fallback: float) -> float:
        return float(np.mean(vals)) if vals else fallback

    # Fallbacks keep old fixed-0.5 behavior if logs are sparse.
    return {
        "up_win": mean_or(up_win, 1.0),
        "up_loss": mean_or(up_loss, -1.0),
        "down_win": mean_or(dn_win, 1.0),
        "down_loss": mean_or(dn_loss, -1.0),
    }


def apply_confirmation(side: np.ndarray, confirm_bars: int) -> np.ndarray:
    if confirm_bars <= 1:
        return side
    out = np.full_like(side, "Flat", dtype=object)
    prev = "Flat"
    streak = 0
    for i, s in enumerate(side):
        if s == "Flat":
            prev = "Flat"
            streak = 0
            out[i] = "Flat"
            continue
        if s == prev:
            streak += 1
        else:
            prev = s
            streak = 1
        out[i] = s if streak >= confirm_bars else "Flat"
    return out


def build_dataset(
    df: pd.DataFrame,
    horizon: int,
    burst_pnl_thr: float,
    burst_rate_thr: float,
    proxy: dict[str, float],
    lookahead_shift: int = 1,
) -> pd.DataFrame:
    out = df.sort_index().copy()

    up_market = np.where(
        out["close"] > out["open"],
        4.0 * float(proxy["up_win"]),
        np.where(out["close"] < out["open"], 4.0 * float(proxy["up_loss"]), 0.0),
    )
    dn_market = np.where(
        out["close"] < out["open"],
        4.0 * float(proxy["down_win"]),
        np.where(out["close"] > out["open"], 4.0 * float(proxy["down_loss"]), 0.0),
    )
    out["market_pnl_up"] = up_market
    out["market_pnl_down"] = dn_market
    out["is_up"] = (out["close"] > out["open"]).astype(int)
    out["is_down"] = (out["close"] < out["open"]).astype(int)

    # Forward window (t+1 .. t+horizon)
    # Forward window aligned at t:
    # sum/mean over [t+lookahead_shift, ..., t+lookahead_shift+horizon-1]
    out["fwd_up_pnl"] = (
        out["market_pnl_up"]
        .shift(-lookahead_shift)
        .rolling(horizon, min_periods=horizon)
        .sum()
        .shift(-(horizon - 1))
    )
    out["fwd_down_pnl"] = (
        out["market_pnl_down"]
        .shift(-lookahead_shift)
        .rolling(horizon, min_periods=horizon)
        .sum()
        .shift(-(horizon - 1))
    )
    out["fwd_up_rate"] = (
        out["is_up"]
        .shift(-lookahead_shift)
        .rolling(horizon, min_periods=horizon)
        .mean()
        .shift(-(horizon - 1))
    )
    out["fwd_down_rate"] = (
        out["is_down"]
        .shift(-lookahead_shift)
        .rolling(horizon, min_periods=horizon)
        .mean()
        .shift(-(horizon - 1))
    )

    out["burst_up"] = ((out["fwd_up_pnl"] >= burst_pnl_thr) & (out["fwd_up_rate"] >= burst_rate_thr)).astype(int)
    out["burst_down"] = ((out["fwd_down_pnl"] >= burst_pnl_thr) & (out["fwd_down_rate"] >= burst_rate_thr)).astype(int)

    # Conservative anti-leak: predict from previous candle state.
    feats = feature_columns(out)
    out[feats] = out[feats].shift(1)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feats + ["burst_up", "burst_down", "market_pnl_up", "market_pnl_down"])
    return out


def run_walkforward(
    df: pd.DataFrame,
    min_train_months: int,
    p_thr_up: float,
    p_thr_down: float,
    choose_delta: float,
    confirm_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = feature_columns(df)
    monthly_rows: list[dict] = []
    decision_rows: list[pd.DataFrame] = []

    for period, train, test in month_splits(df, min_train_months):
        if len(train) < 10_000 or len(test) < 1_000:
            continue
        if train["burst_up"].nunique() < 2 or train["burst_down"].nunique() < 2:
            continue

        m_up = fit_model(train[feats], train["burst_up"])
        m_down = fit_model(train[feats], train["burst_down"])
        p_up = m_up.predict_proba(test[feats])[:, 1]
        p_down = m_down.predict_proba(test[feats])[:, 1]

        raw_side = np.where(
            (p_up >= p_thr_up) & (p_up >= p_down + choose_delta),
            "Up",
            np.where((p_down >= p_thr_down) & (p_down >= p_up + choose_delta), "Down", "Flat"),
        )
        side = apply_confirmation(raw_side, confirm_bars)

        pnl = np.where(side == "Up", test["market_pnl_up"].to_numpy(), np.where(side == "Down", test["market_pnl_down"].to_numpy(), 0.0))
        trade_mask = side != "Flat"
        wins = ((side == "Up") & (test["market_pnl_up"].to_numpy() > 0)) | ((side == "Down") & (test["market_pnl_down"].to_numpy() > 0))

        month = {
            "month": period,
            "rows": len(test),
            "trades": int(trade_mask.sum()),
            "winrate": float(wins[trade_mask].mean()) if trade_mask.any() else np.nan,
            "pnl": float(pnl.sum()),
            "avg_pnl_per_trade": float(pnl[trade_mask].mean()) if trade_mask.any() else np.nan,
            "coverage": float(trade_mask.mean()),
            "up_trades": int((side == "Up").sum()),
            "down_trades": int((side == "Down").sum()),
            "auc_up": float(roc_auc_score(test["burst_up"], p_up)) if test["burst_up"].nunique() > 1 else np.nan,
            "auc_down": float(roc_auc_score(test["burst_down"], p_down)) if test["burst_down"].nunique() > 1 else np.nan,
        }
        monthly_rows.append(month)

        tmp = pd.DataFrame(
            {
                "ts": test.index.astype(str),
                "month": period,
                "p_up_burst": p_up,
                "p_down_burst": p_down,
                "raw_side": raw_side,
                "side": side,
                "pnl": pnl,
            },
            index=test.index,
        )
        decision_rows.append(tmp)

    monthly = pd.DataFrame(monthly_rows)
    decisions = pd.concat(decision_rows).sort_index() if decision_rows else pd.DataFrame()
    return monthly, decisions


def write_report(path: Path, cfg: dict, monthly: pd.DataFrame) -> None:
    lines = ["# Burst Regime Walk-Forward", ""]
    for k, v in cfg.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    if monthly.empty:
        lines.append("No monthly results.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    total_trades = int(monthly["trades"].sum())
    total_pnl = float(monthly["pnl"].sum())
    w = monthly["trades"].clip(lower=0).to_numpy(dtype=float)
    wr_vals = monthly["winrate"].fillna(0).to_numpy(dtype=float)
    wr = float(np.average(wr_vals, weights=w)) if w.sum() > 0 else np.nan
    coverage = np.average(monthly["coverage"], weights=monthly["rows"])
    lines += [
        "## Summary",
        "",
        f"- months: `{len(monthly)}`",
        f"- trades: `{total_trades}`",
        f"- pnl: `{total_pnl:+.2f}`",
        f"- weighted winrate: `{(wr * 100 if not np.isnan(wr) else np.nan):.2f}%`",
        f"- avg coverage: `{coverage * 100:.2f}%`",
        "",
        "## Monthly",
        "",
        "| month | trades | winrate | pnl | avg_pnl_per_trade | coverage | up_trades | down_trades | auc_up | auc_down |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in monthly.iterrows():
        wrs = "-" if pd.isna(r["winrate"]) else f"{100 * float(r['winrate']):.1f}%"
        aup = "-" if pd.isna(r["auc_up"]) else f"{float(r['auc_up']):.3f}"
        adn = "-" if pd.isna(r["auc_down"]) else f"{float(r['auc_down']):.3f}"
        lines.append(
            f"| {r['month']} | {int(r['trades'])} | {wrs} | {float(r['pnl']):+.2f} | "
            f"{float(r['avg_pnl_per_trade']) if not pd.isna(r['avg_pnl_per_trade']) else np.nan:+.3f} | "
            f"{100 * float(r['coverage']):.1f}% | {int(r['up_trades'])} | {int(r['down_trades'])} | {aup} | {adn} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward burst regime models (UP/DOWN).")
    parser.add_argument("--data", type=Path, default=ROOT / "data/btc_5m_poly_features_aggtrades.parquet")
    parser.add_argument("--horizon", type=int, default=96, help="Forward markets for burst label (96 ~= 8h on 5m).")
    parser.add_argument("--burst-pnl-thr", type=float, default=40.0, help="Future pnl threshold for burst label.")
    parser.add_argument("--burst-rate-thr", type=float, default=0.56, help="Future directional hit-rate threshold.")
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--p-thr-up", type=float, default=0.60)
    parser.add_argument("--p-thr-down", type=float, default=0.60)
    parser.add_argument("--choose-delta", type=float, default=0.05, help="Min proba gap between UP and DOWN to select a side.")
    parser.add_argument("--confirm-bars", type=int, default=1, help="Require N consecutive same-side regime signals before trading.")
    parser.add_argument("--proxy-mode", choices=["fixed05", "empirical_recent"], default="fixed05")
    parser.add_argument("--quotes-glob", default="logs/polymarket_crypto5m_quotes_*.csv")
    parser.add_argument("--resolved-glob", default="logs/polymarket_crypto5m_scan_resolved_*.csv")
    parser.add_argument("--out-monthly", type=Path, default=ROOT / "logs/burst_regime_monthly.csv")
    parser.add_argument("--out-decisions", type=Path, default=ROOT / "logs/burst_regime_decisions.csv")
    parser.add_argument("--out-report", type=Path, default=ROOT / "logs/burst_regime_report.md")
    args = parser.parse_args()

    raw = pd.read_parquet(args.data).sort_index()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    if args.proxy_mode == "empirical_recent":
        proxy = estimate_empirical_proxy(args.quotes_glob, args.resolved_glob)
    else:
        proxy = {"up_win": 1.0, "up_loss": -1.0, "down_win": 1.0, "down_loss": -1.0}

    ds = build_dataset(
        raw,
        horizon=args.horizon,
        burst_pnl_thr=args.burst_pnl_thr,
        burst_rate_thr=args.burst_rate_thr,
        proxy=proxy,
        lookahead_shift=1,
    )

    monthly, decisions = run_walkforward(
        ds,
        min_train_months=args.min_train_months,
        p_thr_up=args.p_thr_up,
        p_thr_down=args.p_thr_down,
        choose_delta=args.choose_delta,
        confirm_bars=args.confirm_bars,
    )

    args.out_monthly.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.out_monthly, index=False)
    decisions.to_csv(args.out_decisions, index=False)

    cfg = {
        "data": args.data,
        "horizon": args.horizon,
        "burst_pnl_thr": args.burst_pnl_thr,
        "burst_rate_thr": args.burst_rate_thr,
        "min_train_months": args.min_train_months,
        "p_thr_up": args.p_thr_up,
        "p_thr_down": args.p_thr_down,
        "choose_delta": args.choose_delta,
        "confirm_bars": args.confirm_bars,
        "proxy_mode": args.proxy_mode,
        "proxy": proxy,
        "rows_after_build": len(ds),
        "feature_count": len(feature_columns(ds)),
    }
    write_report(args.out_report, cfg, monthly)

    print(f"wrote {args.out_monthly}")
    print(f"wrote {args.out_decisions}")
    print(f"wrote {args.out_report}")
    if monthly.empty:
        print("no monthly rows")
        return
    print(monthly.to_string(index=False))
    total_pnl = float(monthly["pnl"].sum())
    total_trades = int(monthly["trades"].sum())
    print(f"TOTAL trades={total_trades} pnl={total_pnl:+.2f}")


if __name__ == "__main__":
    main()
