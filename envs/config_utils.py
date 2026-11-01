"""Build a FamineMarketEnv from the YAML config, so evaluation scripts use the
same data source and parameters that training used."""

from __future__ import annotations

from pathlib import Path

import yaml

from envs.famine_market import FamineMarketEnv


def make_env_from_config(
    config_path: str = "configs/ppo_default.yaml",
    seed: int = 42,
    split: str = "test",
    **overrides,
) -> FamineMarketEnv:
    cfg_env = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg_env = (yaml.safe_load(f) or {}).get("env", {})
    kwargs = dict(
        data_source=cfg_env.get("data_source"),
        episode_length=cfg_env.get("episode_length", 252),
        initial_cash=cfg_env.get("initial_cash", 10_000.0),
        max_inventory=cfg_env.get("max_inventory", 100),
        holding_cost_rate=cfg_env.get("holding_cost_rate", 0.005),
        trade_cost_rate=cfg_env.get("trade_cost_rate", 0.002),
        shock_penalty=cfg_env.get("shock_penalty", 5.0),
        relief_bonus=cfg_env.get("relief_bonus", 10.0),
        seed=seed,
        data_split=split,
    )
    kwargs.update(overrides)
    return FamineMarketEnv(**kwargs)
