"""Plot train/test accuracy vs step (log-x) from a run's metrics.csv."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_run(run_dir: str, out_path: str | None = None):
    csv_path = os.path.join(run_dir, "metrics.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    # Clip step to >=1 so the log axis handles the step=0 row cleanly.
    x = df["step"].clip(lower=1)
    ax.plot(x, df["train_acc"], label="train", color="tab:blue")
    ax.plot(x, df["test_acc"], label="test", color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(os.path.basename(os.path.normpath(run_dir)))
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(run_dir, "plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to results/<run_name>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    plot_run(args.run, args.out)


if __name__ == "__main__":
    main()
