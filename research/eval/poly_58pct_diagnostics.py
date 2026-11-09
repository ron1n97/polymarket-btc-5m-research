#!/usr/bin/env python3
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_poly_bet_dataset import add_buckets, add_orderbook_features, load_poly_rows


DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "winrate": np.nan,
            "pnl": 0.0,
            "avg_pnl": np.nan,
            "profit_factor": np.nan,
            "avg_price": np.nan,
            "avg_ev": np.nan,
            "avg_p_side": np.nan,
            "breakeven_wr": np.nan,
        }
    return {
        "trades": int(len(df)),
        "winrate": float(df["win_int"].mean()),
        "pnl": float(df["pnl"].sum()),
        "avg_pnl": float(df["pnl"].mean()),
        "profit_factor": profit_factor(df["pnl"]),
        "avg_price": float(df["mkt_price"].mean()),
        "avg_ev": float(df["ev"].mean()),
        "avg_p_side": float(df["p_side"].mean()),
        "breakeven_wr": float(df["mkt_price"].mean()),
    }


def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / "btc_5m_poly_features.parquet"
    if df.empty or not path.exists():
        return df

    cols = [
        "close",
        "ret_1",
        "ret_3",
        "ret_12",
        "roll_std_20",
        "atr_ratio",
        "body_ratio",
        "high_low_ratio",
        "regime_trend",
        "regime_vol_high",
        "regime_sideways",
        "pos_in_range_20",
        "session_asia",
        "session_europe",
        "session_us",
        "session_overlap",
    ]
    all_cols = pd.read_parquet(path).columns
    feat = pd.read_parquet(path, columns=[c for c in cols if c in all_cols])
    if feat.index.tz is not None:
        feat.index = feat.index.tz_convert("UTC").tz_localize(None)
    feat = feat.sort_index().reset_index().rename(columns={feat.index.name or "index": "entry_ts"})
    if "open_time" in feat.columns and "entry_ts" not in feat.columns:
        feat = feat.rename(columns={"open_time": "entry_ts"})
    feat["entry_ts"] = pd.to_datetime(feat["entry_ts"], errors="coerce").astype("datetime64[ns]")
    left = df.sort_values("entry_ts").copy()
    left["entry_ts"] = pd.to_datetime(left["entry_ts"], errors="coerce").astype("datetime64[ns]")

    out = pd.merge_asof(
        left,
        feat.sort_values("entry_ts"),
        on="entry_ts",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )
    out["hour_utc"] = pd.to_datetime(out["entry_ts"]).dt.hour
    out["dow"] = pd.to_datetime(out["entry_ts"]).dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    regime = np.full(len(out), "unknown", dtype=object)
    if "regime_sideways" in out:
        regime[out["regime_sideways"].fillna(0).to_numpy() == 1] = "ranging"
    if "regime_vol_high" in out:
        regime[out["regime_vol_high"].fillna(0).to_numpy() == 1] = "volatile"
    if "regime_trend" in out:
        trend = out["regime_trend"].fillna(0).to_numpy()
        regime[(trend > 0) & (regime != "volatile")] = "trending_up"
        regime[(trend < 0) & (regime != "volatile")] = "trending_down"
    out["market_regime"] = regime
    return out


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty or by not in df:
        return pd.DataFrame()
    res = df.groupby(by, observed=False, dropna=False).agg(
        trades=("pnl", "size"),
        winrate=("win_int", "mean"),
        pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        pf=("pnl", profit_factor),
        avg_price=("mkt_price", "mean"),
        avg_ev=("ev", "mean"),
        avg_p_side=("p_side", "mean"),
    )
    return res.sort_values(["pnl", "winrate"], ascending=False)


def filter_mask(df: pd.DataFrame, params: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if params["horizon"] != "all":
        mask &= df["horizon"].eq(params["horizon"])
    mask &= df["mkt_price"].between(params["min_price"], params["max_price"])
    mask &= df["ev"].between(params["min_ev"], params["max_ev"])
    mask &= df["p_side"].between(params["min_pside"], params["max_pside"])
    if params.get("direction") != "all":
        mask &= df["direction"].eq(params["direction"])
    if params.get("regime") != "all" and "market_regime" in df:
        mask &= df["market_regime"].eq(params["regime"])
    return mask


def candidate_filters(df: pd.DataFrame, min_trades: int = 8) -> pd.DataFrame:
    rows = []
    grids = product(
        ["all", 1, 3, 6],
        [0.35, 0.40, 0.45],
        [0.55, 0.60, 0.65, 1.0],
        [0.0, 0.01, 0.02],
        [0.05, 0.08, 0.10, 1.0],
        [0.50, 0.52, 0.53],
        [0.55, 0.57, 0.60, 1.0],
        ["all", "Up", "Down"],
        ["all"] + sorted([r for r in df.get("market_regime", pd.Series(dtype=str)).dropna().unique()]),
    )
    for horizon, min_price, max_price, min_ev, max_ev, min_pside, max_pside, direction, regime in grids:
        if min_price >= max_price or min_ev >= max_ev or min_pside >= max_pside:
            continue
        params = {
            "horizon": horizon,
            "min_price": min_price,
            "max_price": max_price,
            "min_ev": min_ev,
            "max_ev": max_ev,
            "min_pside": min_pside,
            "max_pside": max_pside,
            "direction": direction,
            "regime": regime,
        }
        selected = df[filter_mask(df, params)]
        if len(selected) < min_trades:
            continue
        s = summarize(selected)
        rows.append({**params, **s})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["pnl", "profit_factor", "trades"], ascending=False)
    return out


def expanding_filter_validation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.sort_values("entry_ts").copy()
    work["period"] = work["entry_ts"].dt.to_period("D")
    periods = list(work["period"].dropna().unique())
    rows = []

    for i in range(2, len(periods)):
        train = work[work["period"].isin(periods[:i])]
        test = work[work["period"].eq(periods[i])]
        if len(train) < 25 or len(test) < 5:
            continue

        filters = candidate_filters(train, min_trades=max(6, int(len(train) * 0.10)))
        if filters.empty:
            continue
        best = filters.iloc[0].to_dict()
        selected_test = test[filter_mask(test, best)]
        base = summarize(test)
        gated = summarize(selected_test)
        rows.append(
            {
                "test_period": str(periods[i]),
                "train_trades": len(train),
                "test_trades": len(test),
                "selected_test_trades": gated["trades"],
                "base_test_wr": base["winrate"],
                "base_test_pnl": base["pnl"],
                "gated_test_wr": gated["winrate"],
                "gated_test_pnl": gated["pnl"],
                "gated_test_pf": gated["profit_factor"],
                "best_train_trades": int(best["trades"]),
                "best_train_wr": best["winrate"],
                "best_train_pnl": best["pnl"],
                "horizon": best["horizon"],
                "price": f"{best['min_price']:.2f}-{best['max_price']:.2f}",
                "ev": f"{best['min_ev']:.2f}-{best['max_ev']:.2f}",
                "p_side": f"{best['min_pside']:.2f}-{best['max_pside']:.2f}",
                "direction": best["direction"],
                "regime": best["regime"],
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"
    out = df.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    out = out.reset_index()
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in out.iterrows():
        vals = [str(row[col]) for col in out.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    df = load_poly_rows()
    df = add_orderbook_features(df)
    df = add_buckets(df)
    df = add_market_context(df)
    if not df.empty:
        df["entry_day"] = df["entry_ts"].dt.date

    out_dataset = LOG_DIR / "poly_58pct_dataset.csv"
    df.to_csv(out_dataset, index=False)

    filters = candidate_filters(df)
    out_filters = LOG_DIR / "poly_58pct_filter_candidates.csv"
    filters.to_csv(out_filters, index=False)

    wf = expanding_filter_validation(df)
    out_wf = LOG_DIR / "poly_58pct_walkforward_filters.csv"
    wf.to_csv(out_wf, index=False)

    current = df[
        df["mkt_price"].between(0.45, 0.55)
        & df["ev"].between(0.0, 0.08)
        & df["p_side"].between(0.50, 0.55)
    ] if not df.empty else df

    lines = [
        "# Polymarket 58pct Diagnostics",
        "",
        f"Dataset: `{out_dataset}`",
        f"Filter candidates: `{out_filters}`",
        f"Walk-forward filters: `{out_wf}`",
        "",
        "## Overall",
        "",
        md_table(pd.DataFrame([summarize(df)]), 5),
        "",
        "## Current Runtime Filter",
        "",
        "`price 0.45-0.55`, `EV 0-0.08`, `p_side 0.50-0.55`",
        "",
        md_table(pd.DataFrame([summarize(current)]), 5),
        "",
        "## By Hour UTC",
        "",
        md_table(table(df, "hour_utc"), 24),
        "",
        "## By Market Regime",
        "",
        md_table(table(df, "market_regime"), 20),
        "",
        "## By Horizon",
        "",
        md_table(table(df, "horizon_name"), 20),
        "",
        "## By Price Bucket",
        "",
        md_table(table(df, "price_bucket"), 20),
        "",
        "## Top In-Sample Filters",
        "",
        md_table(filters, 20),
        "",
        "## Expanding Walk-Forward Filter Validation",
        "",
        md_table(wf, 30),
        "",
    ]

    if not wf.empty:
        selected = wf[wf["selected_test_trades"] > 0]
        lines += [
            "## Walk-Forward Summary",
            "",
            md_table(
                pd.DataFrame(
                    [
                        {
                            "periods": len(wf),
                            "periods_with_trades": len(selected),
                            "base_total_pnl": wf["base_test_pnl"].sum(),
                            "gated_total_pnl": selected["gated_test_pnl"].sum(),
                            "base_avg_wr": wf["base_test_wr"].mean(),
                            "gated_avg_wr": selected["gated_test_wr"].mean() if len(selected) else np.nan,
                            "avg_selected_trades": selected["selected_test_trades"].mean() if len(selected) else 0,
                        }
                    ]
                ),
                5,
            ),
            "",
        ]

    report = LOG_DIR / "poly_58pct_diagnostics.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved dataset: {out_dataset}")
    print(f"Saved filters: {out_filters}")
    print(f"Saved walk-forward: {out_wf}")
    print(f"Saved report: {report}")
    if not df.empty:
        s = summarize(df)
        c = summarize(current)
        print(f"Overall: n={s['trades']} WR={s['winrate']:.1%} PnL={s['pnl']:+.4f} PF={s['profit_factor']:.2f}")
        print(f"Current filter: n={c['trades']} WR={c['winrate']:.1%} PnL={c['pnl']:+.4f} PF={c['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
