"""One function per assignment part (C-H), plus video recording.

Each function trains agents and writes its outputs under results/:
figures/ (PNG plots), tables/ (TXT summaries), models/ (best Q-table + config), videos/ (MP4 + GIF).
"""
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from .agent import Config, load_agent, train
from .metrics import WINDOW, episodes_to_reach, evaluate, moving_average
from .noise import make_env
from .plots import plot_band, save_figure, style_curve_axis

RESULTS = Path("results")
BEST_Q = RESULTS / "models" / "best_q.npy"
BEST_CONFIG = RESULTS / "models" / "best_config.json"

SWEEP_SEEDS = [0, 1, 2, 3, 4]    # parts D-F and H: each setting is trained 5 times
SELECTION_SEEDS = [0, 1, 2]      # part G: each grid candidate is trained 3 times
VAL_SEEDS = range(1000, 1100)    # start states used to CHOOSE the best config
TEST_SEEDS = range(5000, 5100)   # different start states used to REPORT final results

SWEEPS = {                                      # parts D, E, F: one hyperparameter at a time
    "alpha": [0.1, 0.5, 0.9],
    "gamma": [0.5, 0.9, 0.99],
    "epsilon_decay": [0.99, 0.995, 0.999],
}
SELECTION_GRID = {                              # part G: combinations that did well in D-F
    "alpha": [0.1, 0.5, 0.9],
    "gamma": [0.9, 0.99],
    "epsilon_decay": [0.99, 0.995],
}
NOISE_LEVELS = [0.0, 0.10, 0.25]                # part H: probability of a random action swap


def _out(kind, name):
    """Path results/<kind>/<name>, creating the folder if needed."""
    folder = RESULTS / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _save_table(name, lines):
    text = "\n".join(lines)
    print(text + "\n")
    _out("tables", name).write_text(text + "\n", encoding="utf-8")


def _load_best_config():
    return Config(**json.loads(BEST_CONFIG.read_text()))


# --------------------------------------------------------------------------- Part C
def run_baseline():
    """Train the baseline config; plot reward, episode length and epsilon per episode."""
    cfg = Config()
    _, rewards, steps = train(cfg)
    x = np.arange(WINDOW, cfg.episodes + 1)   # the moving average exists once a full window is available

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(rewards, color="lightsteelblue", linewidth=0.6, label="Reward per episode")
    ax1.plot(x, moving_average(rewards), color="navy", linewidth=2, label=f"{WINDOW}-episode moving average")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Total reward")
    ax1.set_title("Taxi-v4 baseline: reward vs. training episode")
    ax1.legend(loc="lower right")
    ax_eps = ax1.twinx()   # epsilon on a second y-axis, to relate exploration to reward
    ax_eps.plot(cfg.epsilon_schedule(), color="darkorange", linewidth=1.2)
    ax_eps.set_ylabel("epsilon (exploration rate)", color="darkorange")

    ax2.plot(steps, color="lightsteelblue", linewidth=0.6)
    ax2.plot(x, moving_average(steps), color="navy", linewidth=2)
    ax2.set_ylabel("Steps per episode")
    ax2.set_xlabel("Training episode")
    save_figure(fig, _out("figures", "baseline_rewards.png"))

    _save_table("baseline.txt", [
        f"Baseline config: {cfg}",
        f"Avg reward, first 100 episodes: {rewards[:100].mean():.2f}",
        f"Avg reward, last 100 episodes:  {rewards[-100:].mean():.2f}",
        f"Avg steps,  last 100 episodes:  {steps[-100:].mean():.2f}",
        f"Episodes until 100-ep average > 0: {episodes_to_reach(rewards)}",
    ])


# --------------------------------------------------------------------------- Parts D, E, F
def _train_over_seeds(cfg):
    """Train once per sweep seed; return learning curves and the three comparison metrics."""
    curves, finals, speeds, stabilities = [], [], [], []
    for seed in SWEEP_SEEDS:
        _, rewards, _ = train(replace(cfg, seed=seed))
        curves.append(moving_average(rewards))
        finals.append(rewards[-100:].mean())                 # final performance
        reached = episodes_to_reach(rewards)
        speeds.append(np.nan if reached is None else reached)  # learning speed
        stabilities.append(rewards[-500:].std())             # stability (lower = steadier)
    return {"curves": np.array(curves), "finals": np.array(finals),
            "speeds": np.array(speeds), "stabilities": np.array(stabilities)}


def run_sweeps(params=None):
    """Vary one hyperparameter at a time, keeping the others at the baseline."""
    for param in params or SWEEPS:
        results = {v: _train_over_seeds(replace(Config(), **{param: v})) for v in SWEEPS[param]}

        # Left panel: whole run (learning speed). Right panel: zoomed in (final performance).
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(WINDOW, Config().episodes + 1)
        for i, (value, r) in enumerate(results.items()):
            for ax in axes:
                plot_band(ax, x, r["curves"], f"{param} = {value}", f"C{i}")
        style_curve_axis(axes[0], f"Effect of {param} (mean of {len(SWEEP_SEEDS)} seeds, band = ±1 std)")
        style_curve_axis(axes[1], "Zoomed in: final performance", ylim=(-60, 12))
        save_figure(fig, _out("figures", f"exp_{param}.png"))

        lines = [f"Experiment: {param}  (other settings = baseline, {len(SWEEP_SEEDS)} seeds each)",
                 f"{'value':>8} | {'final reward (last 100)':>24} | {'episodes to avg > 0':>20} | {'std of last 500':>15}"]
        for value, r in results.items():
            lines.append(f"{value:>8} | {r['finals'].mean():>14.2f} ± {r['finals'].std():<7.2f} | "
                         f"{np.nanmean(r['speeds']):>20.0f} | {r['stabilities'].mean():>15.2f}")
        _save_table(f"summary_{param}.txt", lines)


# --------------------------------------------------------------------------- Part G
def _describe(name, res):
    return (f"{name:<10} mean reward = {res['rewards'].mean():6.2f} ± {res['rewards'].std():4.2f} | "
            f"success rate = {100 * res['success'].mean():5.1f}% | "
            f"mean steps = {res['steps'].mean():5.2f} | "
            f"min/max reward = {res['rewards'].min()}/{res['rewards'].max()}")


def run_selection():
    """Grid-search on validation episodes, then compare baseline vs. best on unseen test episodes."""
    # 1. Score every candidate by its greedy reward on the VALIDATION start states.
    candidates = []
    for alpha, gamma, decay in itertools.product(*SELECTION_GRID.values()):
        val_means, speeds = [], []
        for seed in SELECTION_SEEDS:
            cfg = Config(alpha=alpha, gamma=gamma, epsilon_decay=decay, seed=seed)
            agent, rewards, _ = train(cfg)
            val_means.append(evaluate(agent, VAL_SEEDS)["rewards"].mean())
            speeds.append(episodes_to_reach(rewards) or cfg.episodes)
        candidates.append({"alpha": alpha, "gamma": gamma, "epsilon_decay": decay,
                           "val_reward": np.mean(val_means), "speed": np.mean(speeds)})

    # Highest validation reward wins; ties (to 2 decimals) go to the faster learner.
    candidates.sort(key=lambda c: (-round(c["val_reward"], 2), c["speed"]))
    best = candidates[0]
    best_cfg = Config(alpha=best["alpha"], gamma=best["gamma"], epsilon_decay=best["epsilon_decay"])

    # 2. Retrain baseline and best (seed 0) and evaluate both on the same TEST start states.
    base_agent, base_train, _ = train(Config())
    best_agent, best_train, _ = train(best_cfg)
    base_test, best_test = evaluate(base_agent, TEST_SEEDS), evaluate(best_agent, TEST_SEEDS)

    lines = ["Model selection (greedy reward on validation episodes, mean of 3 training seeds):",
             f"{'alpha':>6} {'gamma':>6} {'decay':>7} | {'val reward':>10} | {'episodes to avg > 0':>19}"]
    lines += [f"{c['alpha']:>6} {c['gamma']:>6} {c['epsilon_decay']:>7} | {c['val_reward']:>10.2f} | {c['speed']:>19.0f}"
              for c in candidates]
    lines += [f"\nBest config: {best_cfg}",
              "\nFinal evaluation: greedy policy, exploration OFF, 100 unseen test episodes",
              _describe("Baseline", base_test), _describe("Best", best_test),
              "\nFor reference: TRAINING reward (with exploration), last 100 episodes",
              f"Baseline {base_train[-100:].mean():.2f} | Best {best_train[-100:].mean():.2f}",
              f"Episodes to 100-ep avg > 0: Baseline {episodes_to_reach(base_train)} | "
              f"Best {episodes_to_reach(best_train)}"]
    _save_table("evaluation.txt", lines)

    # Saved for part H and the video, so they reuse this exact agent instead of retraining.
    np.save(_out("models", BEST_Q.name), best_agent.Q)
    BEST_CONFIG.write_text(json.dumps(asdict(best_cfg), indent=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(WINDOW, len(base_train) + 1)
    ax1.plot(x, moving_average(base_train), label="Baseline (alpha=0.1, gamma=0.99, decay=0.995)")
    ax1.plot(x, moving_average(best_train),
             label=f"Best (alpha={best_cfg.alpha}, gamma={best_cfg.gamma}, decay={best_cfg.epsilon_decay})")
    style_curve_axis(ax1, "Training: baseline vs. best", ylim=(-200, 12))
    ax2.boxplot([base_test["rewards"], best_test["rewards"]], showmeans=True)
    ax2.set_xticks([1, 2], ["Baseline", "Best"])
    ax2.set_ylabel("Total reward per test episode")
    ax2.set_title("Greedy evaluation on 100 test episodes (triangle = mean)")
    save_figure(fig, _out("figures", "evaluation.png"))


# --------------------------------------------------------------------------- Part H
def _pool(evals):
    """Merge per-seed evaluation results into one set of numbers."""
    rewards = np.concatenate([e["rewards"] for e in evals])
    return {"reward": rewards.mean(),
            "seed_std": np.std([e["rewards"].mean() for e in evals]),
            "success": 100 * np.concatenate([e["success"] for e in evals]).mean(),
            "steps": np.concatenate([e["steps"] for e in evals]).mean()}


def _noise_study(train_cfg, clean_agent, tag):
    """Train at each noise level, then evaluate (a) those agents and (b) the clean p=0 agent under that noise."""
    results = {}
    for p in NOISE_LEVELS:
        curves, speeds, finals, own_evals, clean_evals = [], [], [], [], []
        for seed in SWEEP_SEEDS:
            agent, rewards, _ = train(replace(train_cfg, seed=seed), env=make_env(p, seed))
            curves.append(moving_average(rewards))
            speeds.append(episodes_to_reach(rewards))
            finals.append(rewards[-100:].mean())
            own_evals.append(evaluate(agent, TEST_SEEDS, env=make_env(p, 100 + seed)))
            clean_evals.append(evaluate(clean_agent, TEST_SEEDS, env=make_env(p, 100 + seed)))
        reached = [s for s in speeds if s is not None]
        results[p] = {"curves": np.array(curves), "speed": np.mean(reached) if reached else None,
                      "n_reached": len(reached), "final": np.mean(finals),
                      "own": _pool(own_evals), "clean": _pool(clean_evals)}

    lines = [f"Noise experiment [{tag}]: alpha={train_cfg.alpha}, gamma={train_cfg.gamma}, "
             f"decay={train_cfg.epsilon_decay}, episodes={train_cfg.episodes}",
             f"{len(SWEEP_SEEDS)} training seeds per noise level; greedy evaluation on 100 test episodes per seed",
             "(eps to avg>0 = mean over seeds that got there, (k/5) = how many seeds did; 'never' = none)\n",
             f"{'p':>5} | {'eps to avg>0':>14} | {'train reward':>12} | "
             f"{'TRAINED AT p: reward / success / steps':>41} | {'TRAINED AT p=0: reward / success / steps':>42}"]
    for p, r in results.items():
        o, c = r["own"], r["clean"]
        speed = "never" if r["speed"] is None else f"{r['speed']:.0f} ({r['n_reached']}/{len(SWEEP_SEEDS)})"
        lines.append(f"{p:>5} | {speed:>14} | {r['final']:>12.2f} | "
                     f"{o['reward']:>13.2f} ± {o['seed_std']:<5.2f} / {o['success']:5.1f}% / {o['steps']:5.1f} | "
                     f"{c['reward']:>15.2f} ± {c['seed_std']:<5.2f} / {c['success']:5.1f}% / {c['steps']:5.1f}")
    _save_table(f"noise_{tag}.txt", lines)

    # Left: learning curves per noise level. Right: noise-trained vs. clean policy under each noise level.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(WINDOW, train_cfg.episodes + 1)
    for i, (p, r) in enumerate(results.items()):
        plot_band(ax1, x, r["curves"], f"p = {p}", f"C{i}")
    style_curve_axis(ax1, f"Learning under action noise (alpha={train_cfg.alpha}, trained at each p)", ylim=(-200, 12))

    idx = np.arange(len(NOISE_LEVELS))
    for offset, key, label in [(-0.2, "own", f"Trained at the same p (alpha={train_cfg.alpha})"),
                               (0.2, "clean", "Trained at p = 0 (clean policy)")]:
        stats = [results[p][key] for p in NOISE_LEVELS]
        ax2.bar(idx + offset, [s["reward"] for s in stats], 0.4,
                yerr=[s["seed_std"] for s in stats], capsize=4, label=label)
    ax2.set_xticks(idx, [f"p = {p}" for p in NOISE_LEVELS])
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Mean greedy reward (100 test episodes)")
    ax2.set_title("Evaluation under noise: robustness of the clean policy")
    ax2.legend(loc="lower left")
    save_figure(fig, _out("figures", f"noise_{tag}.png"))


def run_noise():
    """Part H, run twice: with the deterministic-best config, then with a noise-appropriate one."""
    best_cfg = _load_best_config()
    clean_agent = load_agent(best_cfg, BEST_Q)   # trained on the clean taxi in part G
    _noise_study(best_cfg, clean_agent, "best_config")
    # A smaller alpha averages over random outcomes instead of chasing each one, but needs more episodes.
    _noise_study(replace(best_cfg, alpha=0.1, episodes=10000), clean_agent, "low_alpha")


# --------------------------------------------------------------------------- Video
def record_video(p=0.0, episode_seeds=(5000, 5001, 5002, 5003, 5004), fps=4):
    """Render the best greedy agent on a few test episodes to MP4 (plus a GIF for the README)."""
    best_cfg = _load_best_config()
    agent = load_agent(best_cfg, BEST_Q)
    env = make_env(p, seed=0, render_mode="rgb_array")

    frames = []
    for seed in episode_seeds:
        state, _ = env.reset(seed=seed)
        frames += [env.render()] * 2   # brief pause on the start state
        total, steps, done = 0, 0, False
        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.choose_action(state, greedy=True))
            frames.append(env.render())
            total += reward
            steps += 1
            done = terminated or truncated
        frames += [frames[-1]] * 4      # pause on the final state
        print(f"seed {seed}: reward = {total}, steps = {steps}, delivered = {terminated}")
    env.close()

    name = "taxi_trained_agent" if p == 0 else f"taxi_trained_agent_p{p}"
    imageio.mimsave(_out("videos", f"{name}.mp4"), frames, fps=fps, macro_block_size=1)
    imageio.mimsave(_out("videos", f"{name}.gif"), frames, duration=1000 / fps, loop=0)
    print(f"Saved results/videos/{name}.mp4 and .gif ({len(frames)} frames)")
