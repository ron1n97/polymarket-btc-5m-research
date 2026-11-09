#!/usr/bin/env python3
"""Regenerate research-note figures from reported lab tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def breakeven() -> None:
    asks = np.linspace(0.45, 0.65, 200)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(asks, asks, color="#1f4e79", lw=2, label="Breakeven WR = ask")
    ax.axhline(0.53, color="#c0392b", ls="--", lw=1.5, label="Illustrative model WR ≈ 53%")
    ax.axvline(0.55, color="#7f8c8d", ls=":", lw=1.4, label="Typical entry ask ≈ 0.55")
    ax.fill_between(
        asks,
        asks,
        0.53,
        where=(asks > 0.53),
        color="#c0392b",
        alpha=0.12,
        label="Negative EV region at 53% WR",
    )
    ax.set_xlabel("Executable ask price")
    ax.set_ylabel("Required win rate")
    ax.set_title("Frame B: breakeven win rate vs Polymarket ask")
    ax.set_xlim(0.45, 0.65)
    ax.set_ylim(0.45, 0.65)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "breakeven_wr_vs_ask.png", dpi=160)
    plt.close()


def selective() -> None:
    thr = [0.50, 0.54, 0.57, 0.60, 0.63]
    cov = [100.0, 44.2, 19.4, 7.5, 2.4]
    acc = [56.9, 62.2, 67.2, 72.8, 78.0]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax2 = ax1.twinx()
    l1 = ax1.plot(thr, acc, "o-", color="#1f4e79", lw=2, label="Accuracy")
    l2 = ax2.plot(thr, cov, "s--", color="#d35400", lw=1.8, label="Coverage")
    ax1.axvline(0.60, color="#27ae60", ls=":", lw=1.5)
    ax1.annotate(
        "operating point\nthr=0.60",
        xy=(0.60, 72.8),
        xytext=(0.515, 76),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        color="#27ae60",
    )
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("OOS accuracy (%)", color="#1f4e79")
    ax2.set_ylabel("Coverage (%)", color="#d35400")
    ax1.set_title("Frame A: selective prediction (ensemble)")
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "selective_accuracy_coverage.png", dpi=160)
    plt.close()


def ablation() -> None:
    models = ["LightGBM v2", "Transformer", "Ensemble"]
    auc = [0.514, 0.605, 0.603]
    accm = [51.1, 57.1, 56.9]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    b1 = ax.bar(x - w / 2, auc, w, color="#1f4e79", label="AUC")
    b2 = ax.bar(x + w / 2, [a / 100 for a in accm], w, color="#7f8c8d", label="Accuracy")
    ax.axhline(0.5, color="#c0392b", ls="--", lw=1, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.45, 0.65)
    ax.set_ylabel("Score")
    ax.set_title("Frame A ablation: OOS classification (milestones, not final PM claim)")
    ax.legend(fontsize=8)
    for bar, val in zip(b1, auc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(b2, accm):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 100 + 0.005, f"{val:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "frame_a_model_ablation.png", dpi=160)
    plt.close()


def monthly() -> None:
    months = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
    rets = [3.7, 24.1, 28.5, 24.2, 75.8, 17.0]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(months, rets, color="#1f4e79")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Monthly return (%)")
    ax.set_title("Frame A futures-style sim monthly OOS returns (NOT Polymarket)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "frame_a_monthly_returns.png", dpi=160)
    plt.close()


def path() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, 1.8, 1.2, "Candle\nmodels\n(Frame A)"),
        (2.5, 1.0, 1.8, 1.2, "Gates &\nthresholds"),
        (4.7, 1.0, 1.8, 1.2, "Map to PM\nask / EV"),
        (6.9, 1.0, 2.4, 1.2, "Live / shadow\n→ kill claim"),
    ]
    for x0, y0, w, h, t in boxes:
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, fill=True, facecolor="#eef3f8", edgecolor="#1f4e79", lw=1.5)
        )
        ax.text(x0 + w / 2, y0 + h / 2, t, ha="center", va="center", fontsize=9)
    for x in [2.1, 4.3, 6.5]:
        ax.annotate(
            "",
            xy=(x + 0.35, 1.6),
            xytext=(x, 1.6),
            arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.5),
        )
    ax.set_title("What this note argues: path → final negative Polymarket conclusion", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(FIG / "research_path.png", dpi=160)
    plt.close()


def main() -> None:
    FIG.mkdir(exist_ok=True)
    style()
    breakeven()
    selective()
    ablation()
    monthly()
    path()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
