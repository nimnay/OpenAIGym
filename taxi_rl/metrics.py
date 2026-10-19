"""Learning-curve metrics and greedy policy evaluation."""
import gymnasium as gym
import numpy as np

from .agent import ENV_ID

WINDOW = 100   # moving-average window used for every learning curve


def moving_average(x, window=WINDOW):
    """Mean of the previous `window` values at each point. The output starts at index window-1."""
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / window, mode="valid")


def episodes_to_reach(rewards, threshold=0.0, window=WINDOW):
    """Learning speed: first episode at which the moving-average reward reaches `threshold`.

    Returns None if it never does.
    """
    hits = np.flatnonzero(moving_average(rewards, window) >= threshold)
    return int(hits[0] + window) if len(hits) else None


def evaluate(agent, episode_seeds, env=None):
    """Run the greedy policy (no exploration, no learning) once per start seed.

    Seeding every reset means different agents face exactly the same start states.
    """
    env = env or gym.make(ENV_ID)
    rewards, steps, successes = [], [], []
    for seed in episode_seeds:
        state, _ = env.reset(seed=seed)
        total, n, done, terminated = 0, 0, False, False
        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.choose_action(state, greedy=True))
            total += reward
            n += 1
            done = terminated or truncated
        rewards.append(total)
        steps.append(n)
        successes.append(terminated)   # True only if the passenger was delivered
    return {"rewards": np.array(rewards), "steps": np.array(steps), "success": np.array(successes)}
