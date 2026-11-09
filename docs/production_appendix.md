# Production Appendix (Sanitized)

Private production repository (`btc_poly_alg`) ran the live loop on a VPS while research continued locally. This appendix describes **architecture only**. No credentials, compose secrets, or private configs are published.

## Roles

| Component | Role |
|---|---|
| Local lab | Train / ablate / shadow; promote artifacts when ready |
| Production bot | Poll BTC + Polymarket; decide; dry-run or live orders |
| Redeemer (separate private service) | Periodically redeem resolved winning positions; Telegram summary |

## Runtime shape

```
Binance price/indicator window (Redis rolling buffer)
        ↓
Strategy layer
  - STRATEGY_MODE=ml          # model probabilities
  - STRATEGY_MODE=cheap_contract  # rule-based cheap share entries (often preferred in late runtime)
        ↓
Trading engine (dry_run | live) → Polymarket CLOB client
        ↓
Postgres trade/history store
Prometheus / Grafana (ops visibility)
```

Supporting services in the private compose stack included FastAPI (REST + WS), PostgreSQL, Redis, Prometheus, and Grafana.

## Modes that mattered

- **`dry_run` first** — mandatory before `live`.
- **`cheap_contract`** — late recommended runtime path for small bankrolls (price band filters, target seconds inside the 5m window). Coexisted with ML mode rather than replacing research.
- **Stop-loss service** — optional; disabled by default in the integration notes.

## What is intentionally omitted

- Wallet private keys, builder API secrets, Telegram tokens  
- Exact live parameter dumps tied to funded accounts  
- Redeemer source (ops-only; high secret density; low research signal)  
- Full Grafana/Prometheus config trees  

## Portfolio takeaway

Publishing a giant infra dump would dilute the research story. The useful claim is:

> I ran a promotion path from local research to a containerized Polymarket bot with dry-run/live controls and settlement automation — and still concluded the strategies under test lacked robust edge.
