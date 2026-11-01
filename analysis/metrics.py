"""
analysis/metrics.py
--------------------
Portfolio performance metrics for comparing agents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(portfolio_values: list[float] | np.ndarray) -> dict:
    """
    Given a time-series of portfolio values, compute standard metrics.

    Parameters
    ----------
    portfolio_values : sequence of portfolio values (one per step)

    Returns
    -------
    dict with keys: total_return, sharpe, max_drawdown, calmar, volatility
    """
    pv = np.array(portfolio_values, dtype=float)
    if len(pv) < 2:
        return {}

    # Daily returns
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)

    total_return = (pv[-1] - pv[0]) / pv[0]
    volatility = float(np.std(returns) * np.sqrt(252))  # annualised
    mean_return = float(np.mean(returns) * 252)         # annualised
    sharpe = mean_return / (volatility + 1e-8)

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / np.maximum(peak, 1e-8)
    max_drawdown = float(np.min(drawdown))

    calmar = total_return / (abs(max_drawdown) + 1e-8)

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "volatility": float(volatility),
        "final_value": float(pv[-1]),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of {agent_name: metrics_dict}.
    """
    rows = []
    for name, m in results.items():
        row = {"Agent": name, **m}
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Agent")
    df = df[["final_value", "total_return", "sharpe", "max_drawdown", "calmar", "volatility"]]
    df.columns = ["Final Value ($)", "Total Return", "Sharpe", "Max Drawdown", "Calmar", "Ann. Vol"]
    return df
