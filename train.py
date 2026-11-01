"""
train.py
--------
Main PPO training entry point for Famine Trader.

Usage:
    python train.py                          # use configs/ppo_default.yaml
    python train.py --config configs/ppo_default.yaml --timesteps 1000000
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import VecNormalize

from envs.famine_market import FamineMarketEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SyncNormCallback(BaseCallback):
    """Syncs VecNormalize running stats from train → eval env before every step."""

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on FamineMarketEnv")
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="Override experiment name")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env_fn(cfg_env: dict, seed: int, split: str = "all"):
    """Factory closure for a single environment instance."""
    def _make():
        return FamineMarketEnv(
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
    return _make


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    exp_name = args.name or cfg.get("experiment_name", "ppo_famine")
    run_dir = Path("experiments") / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot to the run directory
    shutil.copy(args.config, run_dir / "config_snapshot.yaml")
    logger.info("Run directory: %s", run_dir)

    # ---------------------------------------------------------------
    # Build vectorised environments
    # ---------------------------------------------------------------
    cfg_env = cfg["env"]
    n_envs = cfg_env.get("n_envs", 4)

    train_env = make_vec_env(
        make_env_fn(cfg_env, seed, cfg_env.get("train_split", "train")),
        n_envs=n_envs,
        seed=seed,
    )
    # Normalise observations and rewards for stabler training
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(
        make_env_fn(cfg_env, seed + 1000, cfg_env.get("eval_split", "val")), n_envs=1, seed=seed + 1000)
    # norm_reward=False so EvalCallback sees real (unnormalised) reward values
    # training=False so running stats don't update during eval
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    # Sync obs normalisation stats from train env so eval obs are on the same scale
    eval_env.obs_rms = train_env.obs_rms

    # ---------------------------------------------------------------
    # Build PPO model
    # ---------------------------------------------------------------
    cfg_ppo = cfg["ppo"]
    cfg_policy = cfg["policy"]

    net_arch = {
        "pi": cfg_policy.get("net_arch", {}).get("pi", [256, 256]),
        "vf": cfg_policy.get("net_arch", {}).get("vf", [256, 256]),
    }

    import torch.nn as nn
    act_fn_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
    activation_fn = act_fn_map.get(cfg_policy.get("activation_fn", "tanh"), nn.Tanh)

    policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn)

    total_timesteps = args.timesteps or cfg["training"]["total_timesteps"]

    model = PPO(
        policy=cfg_policy.get("type", "MlpPolicy"),
        env=train_env,
        learning_rate=cfg_ppo.get("learning_rate", 3e-4),
        n_steps=cfg_ppo.get("n_steps", 2048),
        batch_size=cfg_ppo.get("batch_size", 64),
        n_epochs=cfg_ppo.get("n_epochs", 10),
        gamma=cfg_ppo.get("gamma", 0.99),
        gae_lambda=cfg_ppo.get("gae_lambda", 0.95),
        clip_range=cfg_ppo.get("clip_range", 0.2),
        ent_coef=cfg_ppo.get("ent_coef", 0.01),
        vf_coef=cfg_ppo.get("vf_coef", 0.5),
        max_grad_norm=cfg_ppo.get("max_grad_norm", 0.5),
        normalize_advantage=cfg_ppo.get("normalize_advantage", True),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard") if cfg.get("logging", {}).get("tensorboard") else None,
        verbose=cfg.get("logging", {}).get("verbose", 1),
        seed=seed,
    )

    # ---------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------
    cfg_train = cfg["training"]
    sync_norm_cb = SyncNormCallback(train_env, eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(cfg_train.get("eval_freq", 20000) // n_envs, 1),
        n_eval_episodes=cfg_train.get("n_eval_episodes", 10),
        deterministic=cfg_train.get("deterministic_eval", True),
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg_train.get("save_freq", 50000) // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_famine",
        save_vecnormalize=True,
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    logger.info("Starting training for %d timesteps …", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, sync_norm_cb],
    )

    # Save final model + normalisation stats
    model.save(str(run_dir / "model_final"))
    train_env.save(str(run_dir / "vecnormalize.pkl"))
    logger.info("Training complete. Outputs saved to %s", run_dir)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
