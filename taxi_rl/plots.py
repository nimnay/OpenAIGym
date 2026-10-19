"""Plotting helpers shared by the experiments."""
import matplotlib

matplotlib.use("Agg")   # render to files only; no GUI window needed
import matplotlib.pyplot as plt   # noqa: E402

from .metrics import WINDOW   # noqa: E402


def plot_band(ax, x, curves, label, color):
    """Mean of several runs as a line, with a +/-1 std band across runs."""
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def style_curve_axis(ax, title, ylim=None):
    """Common labels for a reward-vs-episode panel."""
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Reward ({WINDOW}-episode moving average)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    if ylim:
        ax.set_ylim(*ylim)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
