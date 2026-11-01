"""
envs/famine_market.py
----------------------
FamineMarketEnv — a Gymnasium environment modelling a food-commodity trader
operating under supply/famine shocks.

Action space  : Box([-1], [1], float32)  — fraction of max position to buy(+) / sell(-)
Observation   : Box of shape (OBS_DIM,)  — price features + macro + portfolio state
Reward        : mark-to-market PnL − holding cost − shock penalty

Usage:
    from envs.famine_market import FamineMarketEnv
    env = FamineMarketEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from data.loader import load_market_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation feature names (keep in sync with _get_obs)
# ---------------------------------------------------------------------------
OBS_FEATURES = [
    "price_norm",           # current price normalised by episode start price
    "price_return_1d",      # 1-day log return
    "price_return_5d",      # 5-day log return
    "price_sma5_ratio",     # price / 5-day SMA
    "price_sma20_ratio",    # price / 20-day SMA
    "supply_index",         # 0–1 supply level
    "shock_active",         # 0 or 1
    "shock_severity",       # 0–1 severity
    "weather_index",        # seasonal+noise signal
    "conflict_index",       # 0 or 1
    "inventory_norm",       # held units / max_inventory
    "cash_norm",            # remaining cash / initial_cash
    "portfolio_value_norm", # (cash + mark) / initial_cash
    "days_since_shock",     # normalised 0–1 (clipped at 60 days)
    "position_ratio",       # current position / max_inventory
]
OBS_DIM = len(OBS_FEATURES)


class FamineMarketEnv(gym.Env):
    """
    Parameters
    ----------
    data_source : path to a CSV, a data folder (e.g. "data/raw"), or None (synthetic)
    data_split : "all", "train", "val" or "test" (chronological; real data only)
    episode_length : trading steps per episode
    initial_cash : starting cash in USD
    max_inventory : maximum units held at one time
    holding_cost_rate : cost per unit per day (storage)
    trade_cost_rate : fraction of trade value paid as transaction cost
    shock_penalty : extra penalty subtracted from reward when shock is active
    seed : RNG seed (for synthetic data and episode sampling)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        data_source: Optional[str | Path] = None,
        episode_length: int = 252,
        initial_cash: float = 10_000.0,
        max_inventory: int = 100,
        holding_cost_rate: float = 0.005,
        trade_cost_rate: float = 0.002,
        shock_penalty: float = 5.0,
        relief_bonus: float = 10.0,
        seed: int = 42,
        render_mode: Optional[str] = None,
        data_split: str = "all",
    ):
        super().__init__()

        self.episode_length = episode_length
        self.initial_cash = initial_cash
        self.max_inventory = max_inventory
        self.holding_cost_rate = holding_cost_rate
        self.trade_cost_rate = trade_cost_rate
        self.shock_penalty = shock_penalty
        self.relief_bonus = relief_bonus
        self.render_mode = render_mode
        self._seed = seed

        # --- Action space: continuous [-1, 1] ---
        # +1 = buy max_inventory units, -1 = sell all held units
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Observation space ---
        # Clipped bounds: features are normalised; values beyond ±10 are anomalies
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Load data once; episodes are random windows from it
        self._df = load_market_data(source=data_source, seed=seed, split=data_split)
        self._prices = self._df["wheat_price"].values.astype(np.float64)
        self._supply = self._df["supply_index"].values.astype(np.float64)
        self._shock_active = self._df["shock_active"].values.astype(np.float64)
        self._shock_severity = self._df["shock_severity"].values.astype(np.float64)
        self._weather = self._df["weather_index"].values.astype(np.float64)
        self._conflict = self._df["conflict_index"].values.astype(np.float64)

        self._total_steps = len(self._df)

        # Internal state (set in reset)
        self._t: int = 0                   # index into data arrays
        self._start_t: int = 0             # episode start index
        self._step_count: int = 0          # steps taken this episode
        self._cash: float = 0.0
        self._inventory: float = 0.0       # units held (float for partial fills)
        self._start_price: float = 0.0
        self._last_portfolio_value: float = 0.0
        self._last_shock_t: int = -999
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        # Sample a random episode start (leave room for episode_length + SMA window)
        margin = 20 + self.episode_length
        max_start = self._total_steps - margin
        if max_start <= 0:
            raise ValueError(
                f"Data too short ({self._total_steps} days) for "
                f"episode_length={self.episode_length}."
            )
        self._start_t = int(self._np_rng.integers(20, max_start))
        self._t = self._start_t
        self._step_count = 0

        self._cash = self.initial_cash
        self._inventory = 0.0
        self._start_price = float(self._prices[self._t])
        self._last_portfolio_value = self.initial_cash
        self._last_shock_t = -999

        obs = self._get_obs()
        info = self._get_info()
        return np.clip(obs, -10.0, 10.0), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        price = float(self._prices[self._t])
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # --- Execute trade ---
        reward = self._execute_trade(action_val, price)

        # --- Advance time ---
        self._t += 1
        self._step_count += 1

        terminated = False  # no early termination for now (bankruptcy is just low reward)
        truncated = self._step_count >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human(price, action_val, reward)

        return np.clip(obs, -10.0, 10.0), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            price = float(self._prices[self._t])
            print(
                f"t={self._step_count:4d}  price={price:7.2f}  "
                f"inv={self._inventory:5.1f}  cash={self._cash:9.2f}  "
                f"shock={int(self._shock_active[self._t])}"
            )

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_trade(self, action_val: float, price: float) -> float:
        """
        action_val in [-1, 1]:
          > 0 → buy  action_val * max_inventory units
          < 0 → sell action_val * inventory units  (fraction of held)
          = 0 → hold
        """
        units_sold = 0.0

        if action_val > 0:
            # Buy
            desired_units = action_val * self.max_inventory
            max_affordable = self._cash / (price * (1 + self.trade_cost_rate))
            units = min(desired_units, max_affordable, self.max_inventory - self._inventory)
            units = max(0.0, units)
            cost = units * price * (1 + self.trade_cost_rate)
            self._cash -= cost
            self._inventory += units

        elif action_val < 0:
            # Sell
            desired_units = abs(action_val) * self._inventory
            units = min(desired_units, self._inventory)
            units = max(0.0, units)
            revenue = units * price * (1 - self.trade_cost_rate)
            self._cash += revenue
            self._inventory -= units
            units_sold = units

        # Holding cost (storage)
        holding_cost = self._inventory * self.holding_cost_rate
        self._cash -= holding_cost

        # Mark-to-market portfolio value after trade execution and holding cost
        new_portfolio = self._portfolio_value(price)
        # PnL includes both asset price movement from last step and trading costs
        pnl = new_portfolio - self._last_portfolio_value

        # Shock penalty: penalises hoarding excessive inventory during famine
        shock = float(self._shock_severity[self._t])
        hoarding_penalty = shock * self.shock_penalty * (self._inventory / max(1, self.max_inventory))

        # Humanitarian relief bonus: rewards supplying food (selling) during famine
        relief = shock * self.relief_bonus * (units_sold / max(1, self.max_inventory))

        reward = float(pnl - hoarding_penalty + relief)

        # Clip to prevent reward explosion
        reward = float(np.clip(reward, -500.0, 500.0))
        self._last_portfolio_value = new_portfolio
        return reward

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._inventory * price

    def _get_obs(self) -> np.ndarray:
        t = self._t
        price = float(self._prices[t])

        # Price features
        price_norm = price / self._start_price
        ret_1d = np.log(price / self._prices[max(t - 1, 0)])
        ret_5d = np.log(price / self._prices[max(t - 5, 0)])

        window5 = self._prices[max(t - 5, 0): t + 1]
        window20 = self._prices[max(t - 20, 0): t + 1]
        sma5_ratio = price / np.mean(window5) if len(window5) > 0 else 1.0
        sma20_ratio = price / np.mean(window20) if len(window20) > 0 else 1.0

        # Days since last shock (normalised)
        if self._shock_active[t]:
            self._last_shock_t = t
        days_since_shock = min(t - self._last_shock_t, 60) / 60.0

        portfolio_val = self._portfolio_value(price)

        obs = np.array(
            [
                price_norm,
                ret_1d,
                ret_5d,
                sma5_ratio,
                sma20_ratio,
                float(self._supply[t]),
                float(self._shock_active[t]),
                float(self._shock_severity[t]),
                float(self._weather[t]),
                float(self._conflict[t]),
                self._inventory / self.max_inventory,
                self._cash / self.initial_cash,
                portfolio_val / self.initial_cash,
                days_since_shock,
                self._inventory / self.max_inventory,  # position_ratio (same as inv_norm here)
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def _get_info(self) -> dict[str, Any]:
        price = float(self._prices[self._t])
        return {
            "portfolio_value": self._portfolio_value(price),
            "cash": self._cash,
            "inventory": self._inventory,
            "price": price,
            "shock_active": bool(self._shock_active[self._t]),
            "shock_severity": float(self._shock_severity[self._t]),
            "step": self._step_count,
        }

    def _render_human(self, price: float, action: float, reward: float) -> None:
        print(
            f"t={self._step_count:3d}  price={price:7.2f}  "
            f"act={action:+.2f}  inv={self._inventory:5.1f}  "
            f"cash={self._cash:8.1f}  reward={reward:+7.2f}  "
            f"shock={int(self._shock_active[self._t])}"
        )
