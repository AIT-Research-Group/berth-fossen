"""Forecast from the last complete 60-minute observation history in a CSV."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .data import FEATURES
from .train import load_model, predict
from .evaluation import VARIABLES, event_labels, apply_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrators", type=Path)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("Use at least two ensemble samples")
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint)
    if checkpoint["config"].get("task") == "loads":
        raise ValueError("Use berth_sim.training.force_predict for load checkpoints")
    with args.features.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))[-checkpoint["config"]["history_steps"]:]
    history = np.array([[float(row[key]) for key in FEATURES] for row in rows], dtype=np.float32)
    if len(history) != checkpoint["config"]["history_steps"] or not np.isfinite(history).all():
        raise ValueError("A full finite history is required")
    if not all(int(row["quality_ok"]) == 1 for row in rows):
        raise ValueError("Input history contains invalid wave features")
    if not np.allclose(np.diff([float(row["time_s"]) for row in rows]), 60):
        raise ValueError("History must have consecutive one-minute timestamps")
    ensemble = predict(model, history[None], args.samples, seed=args.seed)[0]
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"forecast.npz", ensemble=ensemble, history=history)
    mean = ensemble.mean(axis=0)
    low, high = np.quantile(ensemble, [.05, .95], axis=0)
    with (args.output/"forecast.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["horizon_min", *[f"{name}_{stat}" for name in VARIABLES for stat in ("mean", "q05", "q95")]]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for k in range(len(mean)):
            row = {"horizon_min": k+1}
            for j, name in enumerate(VARIABLES):
                row.update({f"{name}_mean": mean[k,j], f"{name}_q05": low[k,j], f"{name}_q95": high[k,j]})
            writer.writerow(row)
    calibrators = json.loads(args.calibrators.read_text()) if args.calibrators else {}
    events = {}
    for horizon in (5, 10, 30):
        for event, labels in event_labels(ensemble, horizon, checkpoint["config"]["demo_thresholds"]).items():
            raw = float(labels.mean())
            entry = {"raw": raw}
            key = f"{checkpoint['model_name']}/{horizon}/{event}"
            if key in calibrators:
                entry["calibrated"] = float(apply_calibration(np.array([raw]), calibrators[key])[0])
            events[f"{horizon}/{event}"] = entry
    (args.output/"events.json").write_text(json.dumps({"synthetic_model": True,
        "demo_thresholds": checkpoint["config"]["demo_thresholds"], "events": events}, indent=2), encoding="utf-8")
    # JSON escapes non-ASCII paths for Windows consoles using legacy encodings.
    print(json.dumps({"output": str(args.output.resolve()), "samples": len(ensemble)}))


if __name__ == "__main__":
    main()
