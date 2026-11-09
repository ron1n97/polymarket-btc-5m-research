# Figures

Generated for the public research note (`docs/research_note.md`).

| File | Role |
|---|---|
| `research_path.png` | Editorial: path → negative PM conclusion |
| `frame_a_model_ablation.png` | LightGBM / Transformer / ensemble OOS ablation |
| `selective_accuracy_coverage.png` | Confidence gate tradeoff |
| `frame_a_monthly_returns.png` | Futures-style sim months (**not** Polymarket) |
| `breakeven_wr_vs_ask.png` | Frame B economics |
| `reliability_diagram_poly_5m.png` | Calibration process artifact (from lab) |
| `bankroll_proxy_vs_short_real.png` | Proxy fantasy vs short real paths (from lab) |

Regenerate scripted charts (not the two copied lab PNGs):

```bash
python3 scripts/make_figures.py
```
