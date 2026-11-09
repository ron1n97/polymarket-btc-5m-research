# Coin Rotation Truth Report (aggr3, 0.45-0.55)

Frame B shadow / truth-labeled experiment (multi-asset 5m crypto markets).  
**Not a production performance claim** — small sample, research branch only.

Data source:
- `logs/polymarket_crypto5m_shadow_truth_recomputed.csv`
- dedupe by latest run per `window+coin`
- use `truth_aggr3_*` columns (final outcomeprices truth)

Baseline:
- `aggr3 all`: `477 trades`, `WR 50.7%`, `PnL -30.83`
- `aggr3 0.45-0.55`: `229 trades`, `WR 54.6%`, `PnL +13.32`

Rotation model:
- At each 5m window, choose one coin with highest rolling WR
- Coin eligible if:
  - trailing trades `>= min_hist`
  - trailing WR `>= wr_thr`
- switching cooldown in windows (`cooldown`)
- trade only if price in `0.45-0.55`

Best robust configs (grid):
- `wr_thr=0.56`, `min_hist=12`, `cooldown=4`, `conf filter = none`
  - `40 trades`
  - `WR 57.5%`
  - `PnL +4.72`
  - `pnl/trade +0.118`
  - `coverage 27.0% of windows`
  - `switch rate 2.7% of windows`
- `wr_thr=0.58`, `min_hist=12`, `cooldown=4`, `conf filter = none`
  - `38 trades`
  - `WR 57.9%`
  - `PnL +4.72`
  - `pnl/trade +0.124`
  - `coverage 25.7% of windows`
  - `switch rate 3.4% of windows`

Observed coin mix for `wr_thr=0.58/min_hist=12/cooldown=4`:
- `SOL`: `15 trades`, `10 wins`, `+4.32`
- `HYPE`: `9 trades`, `6 wins`, `+3.07`
- `ETH`: `14 trades`, `6 wins`, `-2.67`

Conclusion:
- Rotation improves quality vs truth baseline in this sample.
- Switches are low-frequency (~3% windows), not noisy.
- Confidence filter (`aggr3_avg_side_prob`) did not improve top configs in this dataset.
