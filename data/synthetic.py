"""
data/synthetic.py
-----------------
Generates synthetic food-commodity market data for training/testing the
Famine Trader RL environment.  All generation is deterministic given a seed.

Usage:
    from data.synthetic import SyntheticMarket
    market = SyntheticMarket(seed=42)
    df = market.generate(n_days=1000)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketConfig:
    """Hyper-parameters for synthetic generation."""

    # Price dynamics (Geometric Brownian Motion + mean-reversion)
    initial_price: float = 200.0        # USD / tonne (wheat baseline ~$200)
    drift: float = 0.0001               # daily log-drift (slightly positive)
    volatility: float = 0.012           # daily log-volatility
    mean_reversion_strength: float = 0.02   # Ornstein–Uhlenbeck kappa
    long_run_price: float = 200.0       # equilibrium price for OU

    # Supply index (0 = famine-level scarcity, 1 = surplus)
    initial_supply: float = 0.65
    supply_volatility: float = 0.008
    supply_mean: float = 0.60
    supply_reversion: float = 0.015

    # Famine/shock process
    shock_probability: float = 0.005    # prob per day of a new shock onset
    shock_duration_mean: int = 30       # days (exponential)
    shock_price_impact: float = 0.35    # fraction price spike during shock
    shock_supply_impact: float = -0.30  # supply drops by this fraction

    # Weather index  (seasonal + noise, affects supply)
    weather_amplitude: float = 0.10
    weather_period_days: int = 365
    weather_noise: float = 0.03

    # Conflict index (optional macro feature; drives shocks)
    conflict_base_prob: float = 0.002
    conflict_amplify_shock: float = 2.0  # shock_prob multiplier when conflict high

    # Commodities to simulate
    commodities: list = field(default_factory=lambda: ["wheat", "maize", "sorghum"])
    # Cross-commodity price correlation
    corr_matrix: Optional[np.ndarray] = None  # auto-built if None


class SyntheticMarket:
    """
    Generates a multi-commodity synthetic price + macro dataset.

    Columns returned:
        date, <commodity>_price (per commodity),
        supply_index, shock_active, shock_severity,
        weather_index, conflict_index, inventory_proxy
    """

    def __init__(self, config: Optional[MarketConfig] = None, seed: int = 42):
        self.cfg = config or MarketConfig()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_days: int = 1000, start_date: str = "2005-01-01") -> pd.DataFrame:
        """Generate `n_days` of daily synthetic market data."""
        dates = pd.date_range(start_date, periods=n_days, freq="D")

        # --- macro processes (shared across commodities) ---
        supply = self._simulate_supply(n_days)
        shock_active, shock_severity = self._simulate_shocks(n_days, supply)
        weather = self._simulate_weather(n_days)
        conflict = self._simulate_conflict(n_days)

        # Adjust supply for weather and shocks
        supply_adj = np.clip(
            supply + self.cfg.weather_amplitude * weather
            + shock_severity * self.cfg.shock_supply_impact,
            0.05, 1.0,
        )

        # --- commodity prices (correlated GBM + shock impact) ---
        price_dict = {}
        n_com = len(self.cfg.commodities)
        corr = self._build_corr_matrix(n_com)
        chol = np.linalg.cholesky(corr)

        # Correlated standard normals: shape (n_days, n_com)
        z_indep = self.rng.standard_normal((n_days, n_com))
        z_corr = z_indep @ chol.T  # correlated shocks

        for i, name in enumerate(self.cfg.commodities):
            price_dict[f"{name}_price"] = self._simulate_price(
                n_days, z_corr[:, i], shock_severity
            )

        df = pd.DataFrame(
            {
                "date": dates,
                "supply_index": supply_adj,
                "shock_active": shock_active.astype(int),
                "shock_severity": shock_severity,
                "weather_index": weather,
                "conflict_index": conflict,
                **price_dict,
            }
        )
        df = df.set_index("date")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_price(
        self, n: int, z: np.ndarray, shock_severity: np.ndarray
    ) -> np.ndarray:
        """Ornstein–Uhlenbeck mean-reverting GBM with shock multiplier."""
        cfg = self.cfg
        log_price = np.empty(n)
        log_price[0] = np.log(cfg.initial_price)

        log_mu = np.log(cfg.long_run_price)
        kappa = cfg.mean_reversion_strength
        sigma = cfg.volatility
        mu = cfg.drift

        for t in range(1, n):
            shock_bump = shock_severity[t] * cfg.shock_price_impact
            log_price[t] = (
                log_price[t - 1]
                + mu
                + kappa * (log_mu - log_price[t - 1])  # mean reversion
                + sigma * z[t]
                + shock_bump * 0.01  # gradual shock bleed-in
            )
        return np.exp(log_price)

    def _simulate_supply(self, n: int) -> np.ndarray:
        cfg = self.cfg
        supply = np.empty(n)
        supply[0] = cfg.initial_supply
        kappa = cfg.supply_reversion
        mu = cfg.supply_mean
        sigma = cfg.supply_volatility
        z = self.rng.standard_normal(n)
        for t in range(1, n):
            supply[t] = np.clip(
                supply[t - 1] + kappa * (mu - supply[t - 1]) + sigma * z[t],
                0.05, 1.0,
            )
        return supply

    def _simulate_shocks(
        self, n: int, supply: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Poisson-onset shocks; duration ~ Geometric(1/mean_duration)."""
        cfg = self.cfg
        shock_active = np.zeros(n, dtype=bool)
        shock_severity = np.zeros(n)

        t = 0
        while t < n:
            # shock probability increases when supply is low
            p_shock = cfg.shock_probability * (1 + 2 * max(0, 0.4 - supply[t]))
            if self.rng.random() < p_shock:
                duration = int(
                    self.rng.geometric(1.0 / cfg.shock_duration_mean)
                )
                severity = float(self.rng.uniform(0.3, 1.0))
                end = min(t + duration, n)
                shock_active[t:end] = True
                # severity ramps up then tapers
                ramp = np.concatenate(
                    [
                        np.linspace(0, severity, (end - t) // 2 + 1),
                        np.linspace(severity, 0, (end - t) - (end - t) // 2),
                    ]
                )[: end - t]
                shock_severity[t:end] = np.maximum(shock_severity[t:end], ramp)
                t = end  # next potential shock after this one ends
            else:
                t += 1
        return shock_active, shock_severity

    def _simulate_weather(self, n: int) -> np.ndarray:
        t = np.arange(n)
        seasonal = np.sin(2 * np.pi * t / self.cfg.weather_period_days)
        noise = self.rng.normal(0, self.cfg.weather_noise, n)
        return seasonal + noise

    def _simulate_conflict(self, n: int) -> np.ndarray:
        z = self.rng.random(n)
        conflict = np.zeros(n)
        active = False
        for t in range(n):
            if not active:
                if z[t] < self.cfg.conflict_base_prob:
                    active = True
            else:
                if z[t] < 0.02:  # conflict ends with prob 2%/day
                    active = False
            conflict[t] = 1.0 if active else 0.0
        return conflict

    def _build_corr_matrix(self, n: int) -> np.ndarray:
        if self.cfg.corr_matrix is not None:
            return self.cfg.corr_matrix
        # Default: commodities are positively correlated (~0.65)
        corr = np.full((n, n), 0.65)
        np.fill_diagonal(corr, 1.0)
        return corr


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    market = SyntheticMarket(seed=0)
    df = market.generate(n_days=500)
    print(df.head())
    print(f"\nShock days: {df['shock_active'].sum()} / {len(df)}")
    print(f"Price range (wheat): {df['wheat_price'].min():.2f} – {df['wheat_price'].max():.2f}")
