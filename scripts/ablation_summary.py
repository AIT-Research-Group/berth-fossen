"""Mean ± std over training seeds for the v3 ablation (base vs upgraded GAN/Energy-LSTM).

All runs share outputs/v3_data/dataset, so differences come from the model and training
changes and from the training seed, not from the data.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from berth_sim.training.evaluation import apply_calibration, event_labels  # noqa: E402
from plot_analysis import confusion  # noqa: E402

MODELS = ("persistence", "lstm_point", "energy_lstm", "gan_lstm")
VARIABLES = {"gust_mps": "gust", "reflected_hs_m": "hs"}


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value) if value not in ("", None) else np.nan


def run_metrics(run):
    """Flat dict of scalar metrics for one run directory."""
    out = {}
    config = json.loads((run / "experiment.json").read_text(encoding="utf-8"))["config"]
    for row in read_csv(run / "evaluation" / "forecast_metrics.csv"):
        if row["variable"] not in VARIABLES:
            continue
        key = f"{row['model']}/{VARIABLES[row['variable']]}/{row['horizon_min']}"
        for metric in ("rmse", "crps", "coverage_90", "width_90", "coverage_90_conformal", "width_90_conformal"):
            out[f"{key}/{metric}"] = number(row.get(metric, ""))
    for row in read_csv(run / "evaluation" / "joint_metrics.csv"):
        out[f"{row['model']}/joint_es"] = number(row["normalized_trajectory_energy_score"])
    for row in read_csv(run / "evaluation" / "warning_metrics.csv"):
        if row["horizon_min"] == "30":
            out[f"{row['model']}/brier_{row['event']}"] = number(row["brier_calibrated"])
    calibrators = json.loads((run / "evaluation" / "calibrators.json").read_text(encoding="utf-8"))
    for model in MODELS:
        with np.load(run / "evaluation" / f"{model}_test.npz") as archive:
            ensemble, target = archive["ensemble"], archive["target"]
        truth = event_labels(target, 30, config["demo_thresholds"])
        members = event_labels(ensemble, 30, config["demo_thresholds"])
        for event in ("wave", "any"):
            probability = members[event].mean(axis=1)
            key = f"{model}/30/{event}"
            if key in calibrators:
                probability = apply_calibration(probability, calibrators[key])
            result = confusion(truth[event], probability >= .5)
            out[f"{model}/f1_{event}"] = result["f1"] if result["f1"] is not None else np.nan
            out[f"{model}/pod_{event}"] = result["recall"] if result["recall"] is not None else np.nan
            out[f"{model}/far_{event}"] = 1 - result["precision"] if result["precision"] is not None else np.nan
    for model in MODELS[1:]:
        summary = json.loads((run / model / "training_summary.json").read_text(encoding="utf-8"))
        history = read_csv(run / model / "history.csv")
        out[f"{model}/epochs"] = summary["epochs_run"]
        out[f"{model}/best_epoch"] = int(np.argmin([float(r["validation_score"]) for r in history])) + 1
        out[f"{model}/train_s"] = summary["duration_s"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="v3")
    parser.add_argument("--variants", nargs="+", default=["base", "upgraded"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_results/ablation_v3.json")
    args = parser.parse_args()
    result = {}
    for variant in args.variants:
        runs = sorted(p for p in (ROOT / "outputs").glob(f"{args.prefix}_{variant}_s*")
                      if (p / "evaluation" / "joint_metrics.csv").exists())
        metrics = [run_metrics(run) for run in runs]
        keys = sorted(set().union(*metrics)) if metrics else []
        result[variant] = {"runs": [r.name for r in runs], "metrics": {}}
        for key in keys:
            values = np.array([m.get(key, np.nan) for m in metrics], dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue  # e.g. CRPS of deterministic models: not defined
            result[variant]["metrics"][key] = {
                "mean": float(finite.mean()), "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.,
                "values": [float(v) if np.isfinite(v) else None for v in values]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    show = ["gan_lstm/hs/30/rmse", "energy_lstm/hs/30/rmse", "persistence/hs/30/rmse",
            "gan_lstm/gust/30/rmse", "energy_lstm/gust/30/rmse", "persistence/gust/30/rmse",
            "gan_lstm/hs/30/crps", "energy_lstm/hs/30/crps",
            "gan_lstm/hs/30/coverage_90", "gan_lstm/hs/30/coverage_90_conformal",
            "energy_lstm/hs/30/coverage_90", "energy_lstm/hs/30/coverage_90_conformal",
            "gan_lstm/joint_es", "energy_lstm/joint_es", "gan_lstm/f1_wave", "energy_lstm/f1_wave", "persistence/f1_wave",
            "gan_lstm/brier_wave", "energy_lstm/brier_wave", "gan_lstm/epochs", "gan_lstm/best_epoch", "gan_lstm/train_s"]
    for variant, data in result.items():
        print(f"== {variant}: {data['runs']}")
        for key in show:
            if key in data["metrics"]:
                m = data["metrics"][key]
                print(f"  {key:42s} {m['mean']:.4f} ± {m['std']:.4f}")


if __name__ == "__main__":
    main()
