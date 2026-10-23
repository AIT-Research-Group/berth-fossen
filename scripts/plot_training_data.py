"""Plot one scenario CSV without merging or changing training observations."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/training_v1/dataset/scenario_000.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_v1/evaluation/training_scenario_000.png"))
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty scenario CSV")
    time = np.array([float(row["time_s"]) for row in rows]) / 60
    valid = np.array([int(row["quality_ok"]) == 1 for row in rows])
    variables = [
        ("wind_x_mean_mps", "Wind normal to quay", "m/s"),
        ("wind_y_mean_mps", "Wind along quay", "m/s"),
        ("wind_gust_mps", "Gust speed", "m/s"),
        ("hs_reflected_est_m", "Estimated reflected significant wave height", "m"),
    ]
    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for axis, (key, title, unit), color in zip(axes, variables, ("tab:blue", "tab:green", "tab:orange", "tab:purple")):
        values = np.array([float(row[key]) for row in rows])
        if key == "hs_reflected_est_m":
            values[~valid] = np.nan
        axis.plot(time, values, color=color, linewidth=1.5)
        axis.set(title=title, ylabel=unit)
        axis.grid(alpha=.25)
    axes[-1].set_xlabel("Minutes since simulation start")
    figure.suptitle(f"Training data: {args.input.name}\nSynthetic scenario | {len(rows)} one-minute feature rows | valid wave rows: {valid.sum()}", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    print(f"Saved {args.output.as_posix()}")


if __name__ == "__main__":
    main()
