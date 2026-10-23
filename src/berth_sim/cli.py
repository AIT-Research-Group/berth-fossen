import argparse
import csv
import json
from pathlib import Path
import platform
import numpy as np
from . import __version__
from .simulation import simulate
from .loads import load_features


def main():
    parser = argparse.ArgumentParser(description="Fossen-form synthetic berth simulator")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    # Refuse overwrite so previous experimental runs remain reproducible.
    args.output.mkdir(parents=True, exist_ok=False)
    raw, truth, vessel, features = simulate(config)
    np.savez_compressed(args.output / "observations.npz", **raw)
    np.savez_compressed(args.output / "truth.npz", **truth)
    if vessel:
        np.savez_compressed(args.output / "vessel.npz", time_s=raw["time_s"], **vessel)
        window = round(config["feature_step_s"] / config["dt_s"])
        load_rows = load_features(raw["time_s"], vessel["wind_force_body"],
                                  vessel["wave_force_body"], window, window)
        with (args.output / "load_features.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(load_rows[0]))
            writer.writeheader()
            writer.writerows(load_rows)
    # Keep oracle truth out of the training feature file.
    truth_key = "hs_reflected_truth_window_m"
    with (args.output / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [key for key in features[0] if key != truth_key]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(features)
    with (args.output / "feature_truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", truth_key], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(features)
    metadata = {"config": config, "package_version": __version__, "python": platform.python_version(),
                "numpy": np.__version__, "synthetic_only": True,
                "frame": "right-handed z-up; global x toward wall, y along quay; heading CCW",
                "load_features": {"file": "load_features.csv" if vessel else None,
                    "window_s": config["feature_step_s"], "frame": "BODY",
                    "label_source": "simulated loads, not measured ground truth",
                    "environment_definition": "wind + wave, excludes control and hydrodynamic reaction",
                    "wave_zero_axes": [axis for i, axis in enumerate(["X", "Y", "N"])
                        if config["vessel"]["demo_incident_force_rao"][i] == 0
                        and config["vessel"]["demo_reflected_force_rao"][i] == 0]},
                "limitations": ["Uncalibrated demo ship and constant force RAOs",
                    "Fixed-reference wave loads; no moving-ship encounter update",
                    "No radiation memory, second-order drift, contact or gap resonance",
                    "Slow envelope is prescribed, not coupled wind-wave physics"],
                "samples": len(raw["time_s"]), "feature_rows": len(features),
                "valid_feature_rows": sum(row["quality_ok"] for row in features)}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "samples": metadata["samples"],
                      "feature_rows": metadata["feature_rows"], "valid_feature_rows": metadata["valid_feature_rows"]}))


if __name__ == "__main__":
    main()
