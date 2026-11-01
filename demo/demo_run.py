"""
demo/demo_run.py
-----------------
Interactive demo: runs a single episode of FamineMarketEnv with a saved PPO
model (or a chosen baseline), printing each step and producing a final plot.

Usage:
    # Baseline demo (no model needed):
    python demo/demo_run.py --agent rule_based

    # PPO demo:
    python demo/demo_run.py --agent ppo \
        --model experiments/ppo_famine_xxx/best_model/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on path when running from demo/
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.config_utils import make_env_from_config
from agents.random_agent import RandomAgent
from agents.buy_hold import BuyHoldAgent
from agents.rule_based import RuleBasedAgent
from analysis.plots import plot_portfolio_curves, plot_price_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "buy_hold", "rule_based", "ppo"],
                        default="rule_based")
    parser.add_argument("--model", default=None, help="PPO .zip path")
    parser.add_argument("--vecnorm", default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode_length", type=int, default=252)
    parser.add_argument("--output", default="demo/demo_output")
    parser.add_argument("--config", default="configs/ppo_default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env_from_config(
        args.config,
        seed=args.seed,
        split=args.split,
        episode_length=args.episode_length,
        render_mode="human",
    )

    if args.agent == "random":
        agent = RandomAgent(env, seed=args.seed)
    elif args.agent == "buy_hold":
        agent = BuyHoldAgent(env)
    elif args.agent == "rule_based":
        agent = RuleBasedAgent(env)
    elif args.agent == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        if not args.model or not Path(args.model).exists():
            print("Provide a valid --model path for PPO agent.")
            sys.exit(1)
        ppo = PPO.load(args.model)

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
            dummy_env = DummyVecEnv([lambda: env])
            vec_norm = VecNormalize.load(str(vecnorm_path), dummy_env)
            vec_norm.training = False
            vec_norm.norm_reward = False

        class _PPOWrapper:
            def predict(self, obs, deterministic=True):
                if vec_norm is not None:
                    obs = vec_norm.normalize_obs(obs)
                return ppo.predict(obs, deterministic=deterministic)
            def reset(self): pass

        agent = _PPOWrapper()

    obs, info = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    portfolio_values = [info["portfolio_value"]]
    prices = [info["price"]]
    shock_flags = [int(info["shock_active"])]
    actions = []
    done = False

    print(f"\n{'Step':>4} {'Price':>8} {'Action':>7} {'Inventory':>10} {'Cash':>10} {'Portfolio':>10} {'Shock':>6}")
    print("-" * 65)

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        prices.append(info["price"])
        shock_flags.append(int(info["shock_active"]))
        actions.append(float(action[0]))

        shock_str = "*SHOCK*" if info['shock_active'] else ""
        print(
            f"{info['step']:4d}  {info['price']:8.2f}  {float(action[0]):+7.3f}  "
            f"{info['inventory']:10.1f}  {info['cash']:10.1f}  "
            f"{info['portfolio_value']:10.1f}  {shock_str:>8}"
        )

        done = terminated or truncated

    print(f"\nFinal portfolio value: ${portfolio_values[-1]:,.2f}")
    print(f"Total return: {(portfolio_values[-1]/portfolio_values[0]-1)*100:.1f}%")

    # Save plots
    fig1 = plot_portfolio_curves(
        {args.agent: portfolio_values},
        shock_flags=shock_flags,
        title=f"Demo Episode — {args.agent}",
        save_path=out_dir / "portfolio.png",
    )
    fig2 = plot_price_series(
        np.array(prices),
        shock_flags=np.array(shock_flags),
        save_path=out_dir / "price_series.png",
    )
    print(f"\nPlots saved to {out_dir}/")
    env.close()


if __name__ == "__main__":
    main()
