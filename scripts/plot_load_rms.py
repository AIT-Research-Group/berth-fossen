"""Plot one-minute RMS of wind, wave and total BODY loads from load_features.csv."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/loads_demo/load_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/loads_demo/load_rms.png"))
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    minutes = [float(row["time_s"]) / 60 for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for axis, (name, unit, scale, label) in zip(axes, (("X", "N", 1e3, "kN"), ("Y", "N", 1e3, "kN"), ("N", "Nm", 1e6, "MNm"))):
        for source in ("wind", "wave", "environment"):
            axis.plot(minutes, [float(row[f"{source}_{name}_rms_{unit}"]) / scale for row in rows], label=source)
        axis.set(ylabel=f"RMS {name} ({label})")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set(xlabel="Time (min)")
    figure.suptitle("Synthetic BODY loads, one-minute RMS")
    figure.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
