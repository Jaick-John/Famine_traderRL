"""
agents/random_agent.py
-----------------------
Baseline: random action each step.
"""
import numpy as np
from gymnasium import Env


class RandomAgent:
    """Samples uniformly from the continuous action space."""

    def __init__(self, env: Env, seed: int = 0):
        self.action_space = env.action_space
        self.action_space.seed(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        action = self.action_space.sample()
        return action, None

    @property
    def name(self) -> str:
        return "Random"
