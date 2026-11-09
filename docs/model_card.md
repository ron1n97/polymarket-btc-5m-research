# Model Card — BTC 5m Direction Ensemble (Frame A)

> **Scope:** candle direction on Binance BTC/USDT 5m, plus a futures-style execution simulator.  
> **Not in scope:** Polymarket PnL. See `research_note.md` and `failure_modes.md`.

## Model details

| Item | Value |
|---|---|
| Task | Binary: next 5m candle close > open |
| Ensemble | `0.30 × LightGBM_v2 + 0.70 × CandleTransformer` |
| Weight selection | Grid on OOS (optimistic for deployment claims) |
| Primary reported driver | Transformer (OOS AUC 0.605) |

### LightGBM v2

- Features: 121 engineered → top-50 by importance
- Family: GBDT; ~100 trees after val AUC search; `num_leaves=31`, `lr=0.05`, `feature_fraction=0.8`
- OOS: AUC **0.514**, accuracy **51.1%**

### CandleTransformer

- Input: sequence length **60** × **59** features (~5 hours)
- Blocks: Linear → d_model 128 → sinusoidal PE → TransformerEncoder ×3 (4 heads, FFN 256, dropout 0.1, **Pre-LN**)
- Head: last timestep → LN → Linear 128→64 → GELU → Dropout → Linear → sigmoid
- Scaler: RobustScaler
- Train note: walk-forward folds, Adam `1e-3`, up to 30 epochs, batch 512 (Apple Silicon MPS in original lab)
- OOS: AUC **0.605**, accuracy **57.1%**

**Implementation pitfall:** loading weights without `norm_first=True` silently degrades AUC (~0.605 → ~0.523).

## Data

| Source | Role | Approximate span |
|---|---|---|
| BTC 5m Binance | Primary series | Apr 2024 – Apr 2026 (~210k bars) |
| ETH 5m Binance | Lead / correlation features | Same |
| BTC 1m Binance | Sub-bar microstructure-ish features | ~90 days in original setup |

Split: **80% train / 20% OOS** (OOS ~ Nov 2025 – Apr 2026 in lab notes). All headline classification metrics claimed on OOS only.

## Feature families

**v1 (Transformer, 59):** EMA stack + MACD/ADX, RSI/Stoch/CCI/Williams, Bollinger/ATR, volume (OBV/VWAP/MFI/CMF), candle geometry, multi-horizon returns, hour/dow, 1m slopes / taker ratio.

**v2 (LightGBM, 121 → 50):** v1 + lags, streaks, candlestick patterns, rolling stats, regime flags, range position, ETH lead, cyclic time, session dummies.

Builders in-repo: `research/features/build_features.py`, `build_features_v2.py` (Polymarket-oriented variant: `build_features_btc_5m_poly.py`).

## Selective prediction

| Threshold | Coverage | Accuracy | Signals / hour (lab) |
|---|---|---|---|
| 0.50 | 100% | 56.9% | 12.0 |
| 0.54 | 44.2% | 62.2% | 5.3 |
| 0.57 | 19.4% | 67.2% | 2.3 |
| **0.60** | **7.5%** | **72.8%** | **0.9** |
| 0.63 | 2.4% | 78.0% | 0.3 |

## Futures-style simulator (reported OOS)

Parameters: entry threshold 0.60; leverage 5×; SL 1%; TP 3%; max hold 24 bars; risk 2%/trade; fee 0.04%/side; entry on **next** bar open; simultaneous TP+SL → count SL.

| Metric | Reported value |
|---|---|
| Return | +322.4% ($100 → $422) |
| Sharpe | 1.30 |
| Sortino | 2.15 |
| Calmar | 18.50 |
| Max drawdown | −17.4% |
| Winrate | 57.7% |
| Profit factor | 1.42 |
| Trades | 843 (~5.8/day) |
| Exit via close (not TP/SL) | ~85% |

Monthly OOS (lab):

| Month | Trades | Winrate | Return |
|---|---|---|---|
| 2025-11 | 38 | 57.9% | +3.7% |
| 2025-12 | 123 | 54.5% | +24.1% |
| 2026-01 | 158 | 51.9% | +28.5% |
| 2026-02 | 171 | 53.8% | +24.2% |
| 2026-03 | 232 | 64.7% | +75.8% |
| 2026-04 | 121 | 60.3% | +17.0% |

## Intended use

- Portfolio evidence of feature work, model comparison, and selective prediction.
- Starting point for stricter walk-forward re-runs.

## Out-of-scope / hazards

- Transferring sim returns to Polymarket binaries.
- Treating OOS-tuned ensemble weights as production-ready.
- Ignoring OpenMP/MPS interaction when mixing LightGBM + PyTorch on macOS (`OMP_NUM_THREADS=1`; run Transformer before LGBM predict in the original live path).
