"""
analysis/plots.py
------------------
Visualisation utilities for Famine Trader results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def plot_portfolio_curves(
    results: dict[str, list[float]],
    shock_flags: Optional[list[int]] = None,
    title: str = "Portfolio Value — Agent Comparison",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Line chart of portfolio values per agent, with famine shock bands shaded.

    Parameters
    ----------
    results : {agent_name: [portfolio_values_per_step]}
    shock_flags : binary list (1 = shock active), same length as values
    save_path : if given, save the figure there
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, values in results.items():
        ax.plot(values, label=name, linewidth=1.5)

    # Shade shock periods
    if shock_flags is not None:
        shock_arr = np.array(shock_flags)
        in_shock = False
        start = 0
        for i, s in enumerate(shock_arr):
            if s and not in_shock:
                start = i
                in_shock = True
            elif not s and in_shock:
                ax.axvspan(start, i, alpha=0.12, color="red")
                in_shock = False
        if in_shock:
            ax.axvspan(start, len(shock_arr), alpha=0.12, color="red")

    shock_patch = mpatches.Patch(color="red", alpha=0.25, label="Famine Shock")
    handles, labels = ax.get_legend_handles_labels()
    if shock_flags is not None:
        handles.append(shock_patch)

    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("Trading Step")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_price_series(
    prices: np.ndarray,
    shock_flags: Optional[np.ndarray] = None,
    commodity: str = "Wheat",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the price time-series with shock periods highlighted."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(prices, color="steelblue", linewidth=1)
    ax1.set_ylabel(f"{commodity} Price (USD/tonne)")
    ax1.set_title(f"{commodity} Price Series")
    ax1.grid(alpha=0.3)

    if shock_flags is not None:
        ax2.fill_between(range(len(shock_flags)), shock_flags, alpha=0.6, color="tomato")
        ax2.set_ylabel("Shock Active")
        ax2.set_yticks([0, 1])
        ax2.set_xlabel("Day")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    metric: str = "sharpe",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart comparing a single metric across agents."""
    fig, ax = plt.subplots(figsize=(8, 4))
    col_map = {
        "sharpe": "Sharpe",
        "total_return": "Total Return",
        "max_drawdown": "Max Drawdown",
        "final_value": "Final Value ($)",
    }
    col = col_map.get(metric, metric)
    if col not in metrics_df.columns:
        col = metrics_df.columns[0]

    colors = ["#2196F3" if v >= 0 else "#F44336" for v in metrics_df[col]]
    metrics_df[col].plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"{col} by Agent")
    ax.set_xlabel("")
    ax.set_ylabel(col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
