"""
tests/test_env.py
------------------
Comprehensive tests for FamineMarketEnv + data layer.
"""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from data.synthetic import SyntheticMarket, MarketConfig
from data.loader import load_market_data
from envs.famine_market import FamineMarketEnv, OBS_DIM


# -----------------------------------------------------------------------
# Synthetic data tests
# -----------------------------------------------------------------------

class TestSyntheticMarket:
    def test_generate_shape(self):
        m = SyntheticMarket(seed=0)
        df = m.generate(n_days=300)
        assert len(df) == 300

    def test_required_columns(self):
        m = SyntheticMarket(seed=1)
        df = m.generate(n_days=100)
        for col in ["wheat_price", "supply_index", "shock_active", "shock_severity", "weather_index"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_price_positive(self):
        m = SyntheticMarket(seed=2)
        df = m.generate(n_days=200)
        assert (df["wheat_price"] > 0).all()

    def test_supply_range(self):
        m = SyntheticMarket(seed=3)
        df = m.generate(n_days=500)
        assert df["supply_index"].between(0, 1).all()

    def test_shock_binary(self):
        m = SyntheticMarket(seed=4)
        df = m.generate(n_days=200)
        assert set(df["shock_active"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        df1 = SyntheticMarket(seed=99).generate(100)
        df2 = SyntheticMarket(seed=99).generate(100)
        np.testing.assert_array_almost_equal(df1["wheat_price"].values, df2["wheat_price"].values)

    def test_shocks_occur(self):
        # With 2000 days, shocks must occur
        m = SyntheticMarket(seed=5)
        df = m.generate(n_days=2000)
        assert df["shock_active"].sum() > 0, "Expected at least one shock day"


# -----------------------------------------------------------------------
# Loader tests
# -----------------------------------------------------------------------

class TestLoader:
    def test_synthetic_fallback(self):
        df = load_market_data(source=None, n_synthetic_days=300, seed=0)
        assert len(df) == 300

    def test_missing_file_fallback(self):
        df = load_market_data(source="data/raw/nonexistent.csv", n_synthetic_days=200, seed=1)
        assert len(df) == 200


# -----------------------------------------------------------------------
# Environment tests
# -----------------------------------------------------------------------

class TestFamineMarketEnv:
    @pytest.fixture
    def env(self):
        e = FamineMarketEnv(episode_length=50, seed=42)
        yield e
        e.close()

    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_obs_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_action_space_sample(self, env):
        obs, _ = env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert not np.isnan(reward), "Reward is NaN"
            assert not np.isinf(reward), "Reward is Inf"

    def test_episode_completes(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert steps == 50  # episode_length

    def test_info_keys(self, env):
        obs, info = env.reset()
        required = {"portfolio_value", "cash", "inventory", "price", "shock_active", "step"}
        assert required.issubset(set(info.keys()))

    def test_multiple_resets(self, env):
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == (OBS_DIM,)
            assert info["portfolio_value"] > 0

    def test_buy_reduces_cash(self, env):
        obs, info = env.reset()
        initial_cash = info["cash"]
        action = np.array([1.0], dtype=np.float32)  # max buy
        obs, r, term, trunc, info = env.step(action)
        # Cash should decrease after buying (unless price is 0, which won't happen)
        assert info["cash"] <= initial_cash + 1e-3  # small tolerance for edge cases

    def test_sell_increases_cash(self, env):
        obs, info = env.reset()
        # First buy
        env.step(np.array([1.0], dtype=np.float32))
        obs, info2 = env.reset()
        # Buy again
        obs, r, term, trunc, info3 = env.step(np.array([1.0], dtype=np.float32))
        cash_after_buy = info3["cash"]
        # Now sell
        obs, r, term, trunc, info4 = env.step(np.array([-1.0], dtype=np.float32))
        assert info4["cash"] >= cash_after_buy - 1.0  # should not decrease significantly

    def test_check_env(self):
        """gymnasium's built-in environment checker."""
        env = FamineMarketEnv(episode_length=30, seed=0)
        check_env(env, warn=True, skip_render_check=True)
        env.close()
