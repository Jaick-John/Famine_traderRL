"""
agents/buy_hold.py
-------------------
Baseline: buy max on day 1, hold for the rest of the episode.
"""
import numpy as np
from gymnasium import Env


class BuyHoldAgent:
    """
    Step 0: buy everything (action = +1.0).
    Steps 1+: hold (action = 0.0).
    """

    def __init__(self, env: Env):
        self._step = 0

    def reset(self):
        self._step = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self._step == 0:
            action = np.array([1.0], dtype=np.float32)
        else:
            action = np.array([0.0], dtype=np.float32)
        self._step += 1
        return action, None

    @property
    def name(self) -> str:
        return "Buy-and-Hold"
