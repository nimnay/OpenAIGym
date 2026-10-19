"""Tabular Q-learning agent and training loop (assignment part B)."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

ENV_ID = "Taxi-v4"


@dataclass
class Config:
    """Every hyperparameter in one place. Change one with e.g. Config(alpha=0.5)."""

    alpha: float = 0.1            # learning rate: fraction of the TD error applied per update
    gamma: float = 0.99           # discount factor: weight of future vs. immediate reward
    epsilon_start: float = 1.0    # initial exploration rate (1.0 = fully random)
    epsilon_min: float = 0.01     # exploration never drops below this
    epsilon_decay: float = 0.995  # epsilon is multiplied by this after every episode
    episodes: int = 3000
    seed: int = 0                 # seeds the agent's RNG and the first env.reset

    def epsilon_schedule(self):
        """Exploration rate used in each training episode (for plotting)."""
        decayed = self.epsilon_start * self.epsilon_decay ** np.arange(self.episodes)
        return np.maximum(self.epsilon_min, decayed)


class QLearningAgent:
    """Q-table plus epsilon-greedy action selection and the Q-learning update."""

    def __init__(self, n_states, n_actions, cfg):
        self.Q = np.zeros((n_states, n_actions))   # 500 x 6 table, all zeros = "know nothing yet"
        self.cfg = cfg
        self.n_actions = n_actions
        self.epsilon = cfg.epsilon_start
        self.rng = np.random.default_rng(cfg.seed)

    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action. greedy=True disables exploration (used for evaluation)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # Break ties randomly: at the start every Q-value is 0, and always taking the
        # first action (South) would bias early behaviour.
        q = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s_next, terminated):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)].

        Only true termination (passenger delivered) cuts off the future value. A time-limit
        truncation is not a terminal state, so the caller passes terminated=False for it.
        """
        target = r if terminated else r + self.cfg.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.cfg.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


def train(cfg, env=None):
    """Train a fresh agent for cfg.episodes episodes.

    Pass `env` to train in a modified environment (e.g. the noisy taxi in part H).
    Returns (agent, reward per episode, steps per episode).
    """
    env = env or gym.make(ENV_ID)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, cfg)
    episode_rewards, episode_steps = [], []

    for ep in range(cfg.episodes):
        # Seed only the first reset; later resets continue the env's seeded RNG stream.
        state, _ = env.reset(seed=cfg.seed if ep == 0 else None)
        total_reward, steps, done = 0, 0, False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated   # stop on delivery OR the 200-step time limit

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    env.close()
    return agent, np.array(episode_rewards), np.array(episode_steps)


def load_agent(cfg, q_path):
    """Rebuild a trained agent from a saved Q-table."""
    agent = QLearningAgent(500, 6, cfg)
    agent.Q = np.load(q_path)
    return agent
