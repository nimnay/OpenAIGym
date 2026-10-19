"""Run the assignment experiments.

Usage (from the repository root):
    python main.py baseline              # part C
    python main.py sweeps [alpha ...]    # parts D, E, F (all three sweeps by default)
    python main.py select                # part G, saves the best agent to results/models/
    python main.py noise                 # part H, needs `select` to have run first
    python main.py video [--p 0.25]      # MP4 of the best agent, optionally under action noise
    python main.py all                   # everything above, in order
"""
import argparse
import time

from taxi_rl import experiments as ex


def main():
    parser = argparse.ArgumentParser(description="Tabular Q-learning on Taxi-v4")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline", help="part C: baseline training curve")
    sweeps = sub.add_parser("sweeps", help="parts D-F: one-hyperparameter-at-a-time sweeps")
    sweeps.add_argument("params", nargs="*", choices=list(ex.SWEEPS), help="default: all")
    sub.add_parser("select", help="part G: pick the best config and evaluate it greedily")
    sub.add_parser("noise", help="part H: training and robustness under action noise")
    video = sub.add_parser("video", help="record the best agent")
    video.add_argument("--p", type=float, default=0.0, help="action-noise probability")
    sub.add_parser("all", help="run every step in order")
    args = parser.parse_args()

    steps = {
        "baseline": ex.run_baseline,
        "sweeps": lambda: ex.run_sweeps(getattr(args, "params", None)),
        "select": ex.run_selection,
        "noise": ex.run_noise,
        "video": lambda: ex.record_video(getattr(args, "p", 0.0)),
    }
    if args.command == "all":
        todo = [*steps.values(), lambda: ex.record_video(0.25)]
    else:
        todo = [steps[args.command]]

    start = time.time()
    for step in todo:
        step()
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
