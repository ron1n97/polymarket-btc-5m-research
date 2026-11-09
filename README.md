# Polymarket BTC 5m Research

[![Research Note PDF](https://img.shields.io/badge/Research%20Note-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](docs/pdf/research_note.pdf)
[![Markdown](https://img.shields.io/badge/Research%20Note-Markdown-blue?style=for-the-badge)](docs/research_note.md)

Honest research notes and code from building, deploying, and killing short-horizon crypto prediction strategies on [Polymarket](https://polymarket.com) BTC Up/Down 5-minute markets.

This is a **curated public extract**, not a dump of the private lab.

Included: research note + figures, model/decision docs, feature builders, walk-forward / EV eval scripts.  
**Not** included: raw `data/`, trained weights, `.env`, wallet/redeemer secrets, full production Docker stack, live logs.

Private origins (kept offline):

| Local lab (`simple_prediction`) | Production stack (`btc_poly_alg`) |
|---|---|
| Feature engineering, model training, walk-forward / OOS diagnostics | Dockerized live bot (Postgres, Redis, FastAPI, Grafana) |
| Paper / shadow trading and truth recomputation | Real CLOB orders + redeemer sidecar for settled positions |

**Bottom line:** the interesting result is not a claimed alpha. Directional models looked fine on candle OOS / futures-style simulation, but **Polymarket economics (entry ask, fees, selective fill quality) erased the edge**. Early live PnL (~$200 on a small bankroll) is treated as noise, not proof.

Start here: **[open the PDF](docs/pdf/research_note.pdf)** · or read the [Markdown version](docs/research_note.md).

---

## Why this repo exists

Hiring and contract screens in crypto / prediction markets reward:

1. Clear market microstructure thinking (breakeven WR at real ask)
2. Walk-forward discipline instead of one lucky holdout
3. Documented negative results
4. A research → deploy → feedback loop

This project was built for that standard — including the failures.

---

## Repository layout

```
docs/
  research_note.md          # main write-up (start here)
  model_card.md             # BTC 5m direction ensemble (candle / futures-style sim)
  failure_modes.md          # why Polymarket live EV broke
  production_appendix.md    # server architecture (sanitized)
  decision_log.md           # distilled Claude Code decisions (no raw chats)
research/
  features/                 # feature builders (Binance OHLCV → model inputs)
  eval/                     # walk-forward / real-PnL diagnostic scripts
results/
  coin_rotation_truth_report.md
figures/                    # plots (add as you regenerate)
```

Large raw datasets, trained weights, `.env` files, wallet keys, and live logs are **not** included.

---

## Key findings (short)

1. **Candle direction model (OOS):** Transformer-led ensemble reached ~57% accuracy / AUC ~0.60 on Binance BTC 5m holdout; selective threshold 0.60 → ~73% accuracy at low coverage. Futures-style sim with leverage looked strong (+322% OOS) — **not** a Polymarket result.
2. **Polymarket translation failure:** ~53% directional hit-rate at typical entry ask 0.54–0.56 implies **negative EV**. Breakeven at ask 0.55 is ~55% WR; useful edge needs ~57–58%+.
3. **Live:** small bankroll experiments; ~$200 early PnL is **not** treated as validated alpha.
4. **Ops:** production bot + auto-redeemer ran on a VPS; research continued locally and was promoted when ready.

Details and tables: [`docs/research_note.md`](docs/research_note.md), [`docs/failure_modes.md`](docs/failure_modes.md).

---

## Quick start (research scripts)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Example: walk-forward / PnL diagnostics expect prepared datasets under ./data
# (download Binance klines yourself; see docs/research_note.md)
python research/eval/poly_lgbm_walkforward.py --help
python research/eval/poly_real_pnl_confirmation.py --help
```

Feature builders:

```bash
python -c "from research.features import build_features, build_features_v2"
```

---

## Related private systems (not published)

- **Production bot** — dual mode: ML strategy vs rule-based cheap-contract; dry-run → live
- **Polymarket redeemer** — periodic gasless redeem + Telegram reports (ops only; contains secrets; stays private)
- **Other PM research** (Truth Social post-count markets, weather markets) — separate write-ups later

---

## Disclaimer

This is educational research. It is not financial advice. Past backtests and small live samples do not imply future profits. Do not run live trading code with real funds unless you understand the risks, custody, and compliance constraints in your jurisdiction.

## License

MIT — see [`LICENSE`](LICENSE).
