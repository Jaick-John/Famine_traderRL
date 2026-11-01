"""
analysis/compare.py
--------------------
Run all baselines + a saved PPO model over evaluation episodes,
collect portfolio trajectories, compute metrics, and produce plots.

Usage:
    # Baselines only (before training PPO):
    python analysis/compare.py

    # Include a trained PPO model:
    python analysis/compare.py --model experiments/ppo_famine_xxx/best_model/best_model.zip \
                                --vecnorm experiments/ppo_famine_xxx/vecnormalize.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless

from envs.config_utils import make_env_from_config
from agents.random_agent import RandomAgent
from agents.buy_hold import BuyHoldAgent
from agents.rule_based import RuleBasedAgent
from analysis.metrics import compute_metrics, metrics_table
from analysis.plots import plot_portfolio_curves, plot_metrics_bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_episode(agent, env: FamineMarketEnv, deterministic: bool = True):
    """Run one episode and return portfolio values and shock flags."""
    obs, info = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    portfolio_values = [info["portfolio_value"]]
    shock_flags = [int(info["shock_active"])]
    done = False

    while not done:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        shock_flags.append(int(info["shock_active"]))
        done = terminated or truncated

    return portfolio_values, shock_flags


def evaluate_agent(agent, env: FamineMarketEnv, n_episodes: int = 10, deterministic: bool = True):
    all_metrics = []
    all_portfolios = []
    all_shocks = []

    for ep in range(n_episodes):
        pv, sf = run_episode(agent, env, deterministic=deterministic)
        all_metrics.append(compute_metrics(pv))
        all_portfolios.append(pv)
        all_shocks.append(sf)

    # Average metrics across episodes
    avg = {}
    for key in all_metrics[0]:
        avg[key] = float(np.mean([m[key] for m in all_metrics]))

    # Return the median-length portfolio for plotting
    lengths = [len(p) for p in all_portfolios]
    median_ep = int(np.argsort(lengths)[len(lengths) // 2])
    return avg, all_portfolios[median_ep], all_shocks[median_ep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to trained PPO .zip")
    parser.add_argument("--vecnorm", default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", default="analysis/results")
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"],
                        help="Which chronological split to evaluate on (real data)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env_from_config(args.config, seed=args.seed, split=args.split)

    agents = {
        "Random": RandomAgent(env, seed=args.seed),
        "Buy-and-Hold": BuyHoldAgent(env),
        "Rule-Based": RuleBasedAgent(env),
    }

    # Optionally load PPO
    if args.model and Path(args.model).exists():
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        logger.info("Loading PPO model from %s", args.model)
        ppo_model = PPO.load(args.model)

        # Look for VecNormalize statistics
        vec_norm = None
        vecnorm_path = None
        if args.vecnorm and Path(args.vecnorm).exists():
            vecnorm_path = Path(args.vecnorm)
        else:
            model_path = Path(args.model)
            candidates = [
                model_path.parent / "vecnormalize.pkl",
                model_path.parent.parent / "vecnormalize.pkl",
            ]
            for c in candidates:
                if c.exists():
                    vecnorm_path = c
                    break

        if vecnorm_path and vecnorm_path.exists():
            logger.info("Loading VecNormalize statistics from %s", vecnorm_path)
            dummy_env = DummyVecEnv([lambda: env])
            vec_norm = VecNormalize.load(str(vecnorm_path), dummy_env)
            vec_norm.training = False
            vec_norm.norm_reward = False

        class PPOWrapper:
            name = "PPO"
            def predict(self, obs, deterministic=True):
                if vec_norm is not None:
                    obs = vec_norm.normalize_obs(obs)
                return ppo_model.predict(obs, deterministic=deterministic)

        agents["PPO"] = PPOWrapper()

    results_metrics = {}
    results_portfolios = {}
    results_shocks = {}

    for name, agent in agents.items():
        logger.info("Evaluating %s over %d episodes …", name, args.episodes)
        avg_m, pv, sf = evaluate_agent(agent, env, n_episodes=args.episodes)
        results_metrics[name] = avg_m
        results_portfolios[name] = pv
        results_shocks[name] = sf
        logger.info(
            "  Sharpe=%.3f  TotalReturn=%.2f%%  MaxDD=%.2f%%",
            avg_m["sharpe"],
            avg_m["total_return"] * 100,
            avg_m["max_drawdown"] * 100,
        )

    # --- Metrics table ---
    df = metrics_table(results_metrics)
    print("\n=== Comparison Table ===")
    print(df.to_string())
    df.to_csv(out_dir / "metrics.csv")

    # --- Plots ---
    # Use shock flags from the first agent as representative
    ref_shocks = results_shocks[next(iter(results_shocks))]
    fig1 = plot_portfolio_curves(
        results_portfolios, shock_flags=ref_shocks,
        save_path=out_dir / "portfolio_curves.png",
    )
    fig2 = plot_metrics_bar(df, metric="sharpe", save_path=out_dir / "sharpe_bar.png")
    fig3 = plot_metrics_bar(df, metric="total_return", save_path=out_dir / "return_bar.png")

    logger.info("Results saved to %s", out_dir)
    env.close()


if __name__ == "__main__":
    main()
