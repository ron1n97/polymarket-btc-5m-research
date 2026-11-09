# Failure Modes — Polymarket Translation

This document lists reasons the Frame A direction work did **not** become durable Polymarket edge.

## 1. Wrong loss function for the venue

Optimizing candle accuracy / futures sim PnL ignores the binary ask.

At ask \(a\), approximate edge is \(WR - a\). A 53% model at \(a=0.55\) loses ~2% per trade in expectation before other costs.

**Fix that was prioritized:** real-ask / historical-price PnL scripts (`research/eval/poly_real_pnl_confirmation.py`) instead of even-odds backtests.

## 2. Single holdout optimism

One 20% OOS slice can look stable while monthly walk-forward variance is ugly. Production decisions need fold-level mean **and** std of WR/AUC/PnL.

**Fix:** walk-forward utilities (`poly_lgbm_walkforward.py`, `burst_regime_walkforward.py`).

## 3. Predicting every bar

A large fraction of 5m bars is near-noise. Trading all signals dilutes any scarce high-confidence regime.

**Partial mitigation:** confidence thresholds (see model card). Still insufficient once asks sit at 0.54–0.56.

## 4. Fragile sequence model under venue shift

Transformer helped Frame A metrics but was a weak / unstable ingredient once the objective moved to Polymarket executable EV. Lab notes favored simpler LGBM + gate structures for PM iteration.

## 5. Feature scale mismatches

Raw aggTrade streams as primary features added noise relative to bar features; better as confirmation derivatives than as the main matrix.

## 6. Operational success ≠ alpha

The VPS stack (API, dry-run/live switch, redeemer, Telegram) could run for days. That proves engineering, not edge. Small live PnL (~$200) was easy to over-interpret emotionally and is **explicitly downgraded** in the public narrative.

## 7. Multi-asset / rotation branches

Truth-labeled shadow logs showed naive flow ~50.7% WR / negative PnL; rotation filters improved WR on small samples (`results/coin_rotation_truth_report.md`) but did not graduate to a claimed strategy.

## Checklist before calling something “edge”

- [ ] Walk-forward, not one holdout  
- [ ] PnL at **executable** prices  
- [ ] Coverage vs WR tradeoff stated  
- [ ] Costs / latency / partial fills discussed  
- [ ] Live sample sized for noise (or labeled anecdotal)  
- [ ] Negative ablations written down  
