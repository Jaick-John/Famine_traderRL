"""
agents/rule_based.py
---------------------
Baseline: simple mean-reversion rule.

Logic:
  - Observe price vs 5-day SMA (feature index 3 in obs: price_sma5_ratio).
  - If price < SMA × (1 − threshold) → BUY at full size.
  - If price > SMA × (1 + threshold) → SELL all.
  - Else → HOLD.
  - During an active shock → SELL (risk-off).
"""
import numpy as np
from gymnasium import Env

# Observation feature indices (must match envs/famine_market.py OBS_FEATURES)
IDX_SMA5_RATIO = 3
IDX_SHOCK_ACTIVE = 6


class RuleBasedAgent:
    """Mean-reversion + shock-aware rule baseline."""

    def __init__(self, env: Env, threshold: float = 0.02):
        self.threshold = threshold

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        sma5_ratio = float(obs[IDX_SMA5_RATIO])
        shock = float(obs[IDX_SHOCK_ACTIVE])

        if shock > 0.5:
            # Sell everything during shock (risk-off / humanitarian constraint)
            action = np.array([-1.0], dtype=np.float32)
        elif sma5_ratio < 1 - self.threshold:
            action = np.array([1.0], dtype=np.float32)   # cheap → BUY
        elif sma5_ratio > 1 + self.threshold:
            action = np.array([-1.0], dtype=np.float32)  # expensive → SELL
        else:
            action = np.array([0.0], dtype=np.float32)   # HOLD

        return action, None

    @property
    def name(self) -> str:
        return "Rule-Based (Mean-Reversion)"
