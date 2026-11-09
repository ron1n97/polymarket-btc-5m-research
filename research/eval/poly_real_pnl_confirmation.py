#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"

HORIZON_TO_TARGET = {1: "y_1", 3: "y_3"}
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
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


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
            "coverage": 0.0,
            "avg_price": np.nan,
            "avg_runtime_p_side": np.nan,
            "avg_mtf_p_side": np.nan,
            "avg_mtf_edge": np.nan,
        }
    return {
        "trades": int(len(df)),
        "winrate": float(df["win_int"].mean()),
        "pnl": float(df["pnl"].sum()),
        "avg_pnl": float(df["pnl"].mean()),
        "profit_factor": profit_factor(df["pnl"]),
        "coverage": np.nan,
        "avg_price": float(df["mkt_price"].mean()),
        "avg_runtime_p_side": float(df["p_side"].mean()),
        "avg_mtf_p_side": float(df["mtf_p_side"].mean()) if "mtf_p_side" in df else np.nan,
        "avg_mtf_edge": float(df["mtf_edge"].mean()) if "mtf_edge" in df else np.nan,
    }


def fit_model(train: pd.DataFrame, features: list[str], target: str) -> lgb.LGBMClassifier:
    split = max(1, int(len(train) * 0.90))
    tr = train.iloc[:split]
    val = train.iloc[split:]
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
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
    model.fit(
        tr[features],
        tr[target],
        eval_set=[(val[features], val[target])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def load_poly_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    df = df.dropna(subset=["entry_ts", "horizon", "direction", "mkt_price", "p_side", "ev", "pnl", "win_int"])
    df = df[df["horizon"].isin([1, 3])].copy()
    df["horizon"] = df["horizon"].astype(int)
    df["runtime_edge"] = df["p_side"] - df["mkt_price"]
    return df.sort_values("entry_ts").reset_index(drop=True)


def add_mtf_predictions(poly: pd.DataFrame, feature_path: Path, clear_only_train: bool) -> pd.DataFrame:
    feat = pd.read_parquet(feature_path).sort_index()
    if feat.index.tz is not None:
        feat.index = feat.index.tz_convert("UTC").tz_localize(None)
    features = feature_columns(feat)
    feat[features] = feat[features].replace([np.inf, -np.inf], np.nan)
    feat_reset = feat.reset_index().rename(columns={feat.index.name or "index": "entry_ts"})
    if "open_time" in feat_reset.columns and "entry_ts" not in feat_reset.columns:
        feat_reset = feat_reset.rename(columns={"open_time": "entry_ts"})
    feat_reset["entry_ts"] = pd.to_datetime(feat_reset["entry_ts"], errors="coerce").astype("datetime64[ns]")
    feat_reset["feature_ts"] = feat_reset["entry_ts"]

    out = poly.copy()
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], errors="coerce").astype("datetime64[ns]")
    merge_cols = ["entry_ts", "feature_ts"] + features
    if "is_clear_candle" in feat_reset.columns and "is_clear_candle" not in merge_cols:
        merge_cols.append("is_clear_candle")
    merged = pd.merge_asof(
        out.sort_values("entry_ts"),
        feat_reset[merge_cols].sort_values("entry_ts"),
        on="entry_ts",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
        suffixes=("", "_feat"),
    )
    merged["mtf_p_up"] = np.nan

    first_entry = merged["entry_ts"].min()
    train_cutoff = pd.Timestamp(first_entry).to_period("M").to_timestamp()
    for horizon, target in HORIZON_TO_TARGET.items():
        rows = merged["horizon"].eq(horizon) & merged["feature_ts"].notna()
        if not rows.any():
            continue
        train = feat[feat.index < train_cutoff].dropna(subset=[target]).copy()
        if clear_only_train and "is_clear_candle" in train:
            clear = train[train["is_clear_candle"] == 1]
            if len(clear) >= 5_000:
                train = clear
        if len(train) < 10_000 or train[target].nunique() < 2:
            continue
        model = fit_model(train, features, target)
        merged.loc[rows, "mtf_p_up"] = model.predict_proba(merged.loc[rows, features])[:, 1]

    merged["mtf_p_side"] = np.where(
        merged["direction"].eq("Up"),
        merged["mtf_p_up"],
        1.0 - merged["mtf_p_up"],
    )
    merged["mtf_edge"] = merged["mtf_p_side"] - merged["mkt_price"]
    merged["mtf_agrees"] = np.where(
        merged["direction"].eq("Up"),
        merged["mtf_p_up"] >= 0.5,
        merged["mtf_p_up"] < 0.5,
    )
    return merged


def sweep(poly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    horizons = ["all", 1, 3]
    price_ranges = [(0.45, 0.55), (0.40, 0.55), (0.45, 0.60)]
    runtime_pmins = [0.50, 0.52, 0.53, 0.54, 0.55]
    mtf_pmins = [0.50, 0.52, 0.53, 0.54, 0.55, 0.56]
    edge_mins = [0.00, 0.01, 0.02, 0.03]
    ev_maxes = [0.08, 0.10, 1.00]
    clear_opts = [False, True]

    configs = []
    for horizon, (min_price, max_price), pmin, edge_min, ev_max in product(
        horizons, price_ranges, runtime_pmins, edge_mins, ev_maxes
    ):
        configs.append(
            {
                "mode": "runtime_logged",
                "horizon": horizon,
                "min_price": min_price,
                "max_price": max_price,
                "runtime_pmin": pmin,
                "mtf_pmin": np.nan,
                "edge_min": edge_min,
                "ev_max": ev_max,
                "clear_only": False,
            }
        )

    for horizon, (min_price, max_price), runtime_pmin, mtf_pmin, edge_min, ev_max, clear_only in product(
        horizons, price_ranges, runtime_pmins, mtf_pmins, edge_mins, ev_maxes, clear_opts
    ):
        configs.append(
            {
                "mode": "runtime_plus_mtf_confirm",
                "horizon": horizon,
                "min_price": min_price,
                "max_price": max_price,
                "runtime_pmin": runtime_pmin,
                "mtf_pmin": mtf_pmin,
                "edge_min": edge_min,
                "ev_max": ev_max,
                "clear_only": clear_only,
            }
        )

    for cfg in configs:
        mask = poly["mkt_price"].between(cfg["min_price"], cfg["max_price"])
        mask &= poly["ev"].between(0.0, cfg["ev_max"])
        if cfg["horizon"] != "all":
            mask &= poly["horizon"].eq(cfg["horizon"])
        mask &= poly["p_side"] >= cfg["runtime_pmin"]
        mask &= poly["runtime_edge"] >= cfg["edge_min"]

        if cfg["mode"] == "runtime_plus_mtf_confirm":
            mask &= poly["mtf_p_side"].notna()
            mask &= poly["mtf_agrees"].fillna(False)
            mask &= poly["mtf_p_side"] >= cfg["mtf_pmin"]
            mask &= poly["mtf_edge"] >= cfg["edge_min"]
            if cfg["clear_only"]:
                clear = poly.get("is_clear_candle", pd.Series(False, index=poly.index))
                if isinstance(clear, pd.DataFrame):
                    clear = clear.iloc[:, 0]
                clear_mask = pd.Series(np.asarray(clear.fillna(0).astype(int).eq(1)), index=poly.index)
                mask = mask & clear_mask

        selected = poly[mask]
        if len(selected) < 3:
            continue
        summary = summarize(selected)
        summary["coverage"] = len(selected) / len(poly)
        rows.append({**cfg, **summary})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["pnl", "profit_factor", "trades"], ascending=False)


def render_table(df: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if df.empty:
        return ["No rows."]
    out = df.head(max_rows).copy()
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in out.iterrows():
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
    parser.add_argument("--poly", type=Path, default=LOG_DIR / "poly_58pct_dataset.csv")
    parser.add_argument("--features", type=Path, default=DATA_DIR / "btc_5m_poly_features_mtf.parquet")
    parser.add_argument("--name", default="", help="Optional output name suffix, e.g. mtf or aggtrades.")
    parser.add_argument("--clear-only-train", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    poly = load_poly_dataset(args.poly)
    enriched = add_mtf_predictions(poly, args.features, clear_only_train=args.clear_only_train)
    train_suffix = "clear_train" if args.clear_only_train else "all_train"
    suffix = f"{args.name}_{train_suffix}" if args.name else train_suffix
    dataset_path = LOG_DIR / f"poly_real_pnl_confirmation_{suffix}_dataset.csv"
    enriched.to_csv(dataset_path, index=False)

    results = sweep(enriched)
    results_path = LOG_DIR / f"poly_real_pnl_confirmation_{suffix}.csv"
    results.to_csv(results_path, index=False)

    coverage = enriched.groupby("horizon").agg(
        rows=("pnl", "size"),
        mtf_matched=("mtf_p_side", lambda s: int(s.notna().sum())),
        runtime_wr=("win_int", "mean"),
        runtime_pnl=("pnl", "sum"),
    ).reset_index()

    report_path = LOG_DIR / f"poly_real_pnl_confirmation_{suffix}.md"
    lines = [
        "# Polymarket Real PnL Confirmation",
        "",
        "Only horizons `1` / `3` are included.",
        "The feature model is used only as a confirmation filter for the logged trade direction.",
        "",
        f"Poly dataset: `{args.poly}`",
        f"Features: `{args.features}`",
        f"Clear-only train: `{args.clear_only_train}`",
        f"Enriched dataset: `{dataset_path}`",
        f"Results CSV: `{results_path}`",
        "",
        "## Coverage",
        "",
    ]
    lines.extend(render_table(coverage, 10))
    lines += ["", "## Top Runtime Logged Filters", ""]
    lines.extend(render_table(results[results["mode"] == "runtime_logged"], 20))
    lines += ["", "## Top Runtime + MTF Confirmation Filters", ""]
    lines.extend(render_table(results[results["mode"] == "runtime_plus_mtf_confirm"], 20))
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved enriched dataset: {dataset_path}")
    print(f"Saved results:          {results_path}")
    print(f"Saved report:           {report_path}")
    print("")
    print("Coverage:")
    print(coverage.to_string(index=False))
    print("")
    if not results.empty:
        print(results.groupby("mode").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
