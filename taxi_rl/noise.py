"""Controllable action noise for the stochastic-environment experiment (part H)."""
import gymnasium as gym
import numpy as np

from .agent import ENV_ID


class RandomActionNoise(gym.ActionWrapper):
    """With probability p, execute a different, uniformly random action instead of the chosen one.

    The agent is not told about the swap, so it learns the value of *attempting* an action.
    """

    def __init__(self, env, p, seed=None):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        if self.rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            return int(self.rng.choice(others))
        return action


def make_env(p, seed, render_mode=None):
    """Taxi-v4 with action-noise level p (p=0 behaves exactly like the original)."""
    return RandomActionNoise(gym.make(ENV_ID, render_mode=render_mode), p, seed)
