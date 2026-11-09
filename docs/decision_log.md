# Decision Log (distilled from Claude Code sessions)

This is **not** a chat dump. It is a cleaned timeline of research / product decisions made while building the Polymarket BTC systems with Claude Code (and related Codex sessions), spanning roughly **March–April 2026**.

Sources (private, not published):

- Claude Code history for `btc-indicator` / production bot (~195 prompts, 11 sessions)
- Claude Code history for `simple_prediction` (~200 prompts, 2 sessions)
- Claude project memory under `simple_prediction` (Apr 20, 2026)
- Selected Codex threads on model comparison / redeem / exits

Tone in the original chats was often frustrated and operational. Here only **decisions and rationale** are kept.

---

## Phase A — Production TCN bot (`btc-indicator`, early March 2026)

### A1. Venue reality over model beauty
**Decision:** Treat Polymarket CLOB quirks (allowance cache, delayed share balance after fill, redeem ≠ CLOB sell) as first-class engineering, not afterthoughts.  
**Why:** Live exits failed repeatedly (`zero_sellable`, allowance=0 while balance eventually appeared). Redeem required on-chain / relayer path, not “sell YES on CLOB”.

### A2. One market window → one bet attempt
**Decision:** Deduplicate by `window_key`; block in-flight duplicates; short retry backoff on transient CLOB errors (e.g. 425); do not chase windows after `post_window_delay`.  
**Why:** Autotrader previously re-fired `quick_trade` whenever confidence stayed high.

### A3. Stop-loss policy kept changing (and that itself is a finding)
Evolution observed in chats:

1. Price / share-price stop ideas  
2. **Confidence drawdown** (`entry_confidence - current_confidence`)  
3. **Direction flip** emergency exit  
4. Later: stop-loss often **disabled in config** while validating entry logic  

**Portfolio framing:** exits were unstable; research value is documenting the search, not claiming a final SL edge.

### A4. Time-aware prediction inside the 5m window
**Decision:** Train/serve forecasts not only at T+5m, but along the open window (interest band roughly **minute 2–4**), updating as 1-second BTC path extends. Windows aligned to UTC `:00/:05/:10/...`.  
**Why:** Remaining time changes decision quality; “always bet at a fixed second” was too blunt.

### A5. Architecture bets
**Decision:** Prefer **TCN** over a quick LSTM baseline for the streaming 1s path; keep alternative baselines offline before production.  
**Why:** Explicit request for max accuracy on second-stream updates; production coupling postponed until validated.

### A6. Threshold / coverage discipline
**Decision:** Sweep confidence thresholds (e.g. 0.75 → 0.85); remove or comment-out filters when threshold alone dominated the tradeoff. Prefer **confidence** over a FLAT class.  
**Why:** Model showed UP/DOWN confidence asymmetry; user wanted selective betting, not forced ternary labels.

### A7. Sizing
**Decision:** Keep **Kelly-style** fractional sizing with a **$1 / 1-share floor**; drop take-profit; stop-loss configurable/off during some live checks.  
**Why:** Fixed bankroll felt wrong; minimum venue size mattered more than fancy TP.

### A8. Honest model skepticism (Codex + Claude)
**Decision:** Treat some high backtest / high-PnL Polymarket-feature runs as possibly **trend-lucky**; keep a more **balanced** TCN candidate in the comparison set.  
**Why:** User explicitly asked whether `polymarket_1m` “just got lucky with the market trend” vs `tcn_balanced_*`.

### A9. Feature shift: include Polymarket path, not only BTC
**Decision:** Rebuild training around historical Polymarket UP/DOWN prices merged with BTC 1s (painful download/repair loops accepted).  
**Why:** Pure BTC direction ≠ executable PM economics; need market-implied path in the dataset.

---

## Phase B — Local lab → Docker framework (`simple_prediction`, mid–late April 2026)

### B1. Fresh start: candle direction lab
**Decision:** Build a cleaner research stack predicting BTC 5m candle direction (parquet features → Transformer + LightGBM), then map to Polymarket.  
**Why:** Parallel to production bot; faster iteration locally.

### B2. ETH as context, not the traded asset
**Decision:** Keep ETH features as lead/context despite “we trade BTC” intuition.  
**Why:** Cross-asset lead was part of the feature set; questioned once, retained.

### B3. Dual Polymarket profiles
**Decision:** Run **conservative** and **aggressive** Polymarket profiles with separate bankroll sleeves (lab notes: ~$50 each; later allocation discussions around larger sleeves / BTC sleeve).  
**Why:** Compare risk appetites without mixing PnL attribution.

### B4. Entry timing rules
**Decision:** Restrict evaluation/entry to market-aligned clocks (15m logic tied to quarter-hour boundaries; later explicit **5m** markets alongside 15m). Add **1-minute confirmation** after signal before execution when backtests showed ~+1.4% WR.  
**Why:** Betting into the wrong minute / wrong slug caused ghost fills and “no market” confusion; waiting improved hit quality at some cost to price.

### B5. Executable price > raw probability
**Decision:** Add EV checks vs live ask; poll briefly (tens of seconds) for a better price before giving up; extend poll windows when fills failed.  
**Why:** Logs showed negative EV at ask ~0.53 even when direction prob looked fine (`EV = f(prob, price)`).

### B6. Min size = venue min
**Decision:** Size to **≥ 1 share (~$1)** Polymarket minimum (abandon pure $0.50 fantasy sizing).  
**Why:** Venue constraint.

### B7. Kelly tempered by path risk
**Decision:** Make Kelly more aggressive only if historical monthly paths stay non-negative / above ruin constraints in sim.  
**Why:** User rejected “percentages that look good” without month-by-month survival checks.

### B8. Regime filter after live pain
**Decision:** Add a Polymarket-specific **regime / trend-vol filter** (memory: block when `tr_24h > 5%` and `vol_1h > 0.37%`), inspired by the BTC 1m “trend filter” kill-switch. Analyze losses to choose the rule; validate on recent live + history before deploy.  
**Why:** Overnight / multi-hour drawdowns; both profiles moved together in adverse regimes; need on/off, not only better accuracy.

### B9. Meta-idea (explored, not claimed as alpha)
**Idea discussed:** learn a gate that predicts *when* the base strategies historically made money (regime classifier / “good market detector”).  
**Status:** Research direction, not a validated edge in the public narrative.

### B10. Label / selection target
**Decision:** Point VI feature selection at **`y_3`** (horizon aligned to trading), not `y_1`. Retrain with fresher data; consider multi-horizon heads if profitable in monthly PnL tables.  
**Why:** Trade horizon mismatch.

### B11. Engineering process rules (from explicit user feedback to Claude)
1. **Do not fix one path by breaking another** (polling vs EV check).  
2. **Ask before deleting** model artifacts / training files.  
3. Prefer extract/refactor over silent replacement of working live paths.  
4. Dashboard / portfolio must include Polymarket fills in total PnL; purge ghost trades that never existed on-chain/UI.

### B12. Backtest ↔ live reconciliation
**Decision:** Whenever live PnL drifts overnight, ask first: **does this match the backtest distribution of drawdowns?** If not, treat as bug (wrong candle/slug/price) before “market regime” stories.  
**Why:** Repeated incidents of wrong market URL, wrong candle alignment, and phantom PnL rows.

---

## Phase C — Weather markets (April 2026, adjacent)

**Decisions:** resolve timezone / local close rules carefully; pull the same data vendors Polymarket uses; plot calibration-style winrate vs forecast before building a fancy model.  
**Status:** Separate research stream; not merged into the BTC 5m public claim set yet.

---

## What this log adds beyond metrics

| Theme | Evidence in dialogues |
|---|---|
| Edge skepticism | Trend-luck hypothesis; EV-at-ask; overnight loss audits |
| Microstructure | Allowance/redeem; FOK; polling; min share; wrong slug |
| Process | Dry-run → live; don’t break working paths; artifact hygiene |
| Iteration | Threshold sweeps; regime filters; 1m confirm; multi-horizon |

Raw transcripts stay private. If a claim in `research_note.md` conflicts with this log, **prefer the more conservative interpretation** (usually: “not validated live edge”).
