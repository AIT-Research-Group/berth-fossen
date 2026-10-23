"""Independent scenario test metrics and calibration on a separate split."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .model import energy_score
from .train import load_model, predict

VARIABLES = ("wind_x_mps", "wind_y_mps", "gust_mps", "reflected_hs_m")
HORIZONS = (5, 10, 30)


def crps(ensemble, target):
    """Exact empirical scalar CRPS with O(M log M) sorting."""
    samples = ensemble.shape[1]
    ordered = np.sort(ensemble, axis=1)
    weights = (2*np.arange(1, samples+1) - samples - 1).reshape(1, samples, *([1]*(ensemble.ndim-2)))
    return np.abs(ensemble - target[:, None]).mean(axis=1) - (ordered * weights).sum(axis=1) / samples**2


def cluster_interval(values, scenarios, replicates, seed=17, square_root=False):
    groups = np.unique(scenarios)
    sums = np.array([values[scenarios == group].sum() for group in groups])
    counts = np.array([(scenarios == group).sum() for group in groups])
    draws = np.random.default_rng(seed).integers(0, len(groups), (replicates, len(groups)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    if square_root:
        estimates = np.sqrt(estimates)
    return np.quantile(estimates, [.025, .975])


def event_labels(values, horizon, thresholds):
    wind = values[..., :horizon, 2] > thresholds["gust_mps"]
    wave = values[..., :horizon, 3] > thresholds["reflected_hs_m"]
    return {"wind": wind.any(axis=-1), "wave": wave.any(axis=-1),
            "simultaneous": (wind & wave).any(axis=-1), "any": (wind | wave).any(axis=-1)}


def conformal_offsets(ensemble, target, alpha=.1):
    """Split-conformal widening of the central (1 - alpha) ensemble interval, per lead time and variable.

    Conformalized quantile regression (Romano et al. 2019) on the calibration split:
    score = distance by which the observation falls outside the raw interval; the
    finite-sample (1 - alpha) quantile of the scores widens (or narrows) the test
    interval. Overlapping calibration windows are only approximately exchangeable.
    """
    low, high = np.quantile(ensemble, [alpha / 2, 1 - alpha / 2], axis=1)
    scores = np.maximum(low - target, target - high)
    n = len(scores)
    level = min(1., math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, axis=0, method="higher")


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit_calibration(probabilities, labels, sample_count):
    """Regularized logistic calibration; constant fallback if one class only.

    Uses calibration scenarios only. Its reliability still needs independent
    testing; overlapping origins are not independent events.
    """
    epsilon = .5 / sample_count
    labels = labels.astype(float)
    if len(np.unique(labels)) < 2:
        return {"kind": "constant", "value": float((labels.sum()+1)/(len(labels)+2)),
                "reason": "calibration split contains one class", "positives": int(labels.sum())}
    p = np.clip(probabilities, epsilon, 1-epsilon)
    design = np.column_stack((np.log(p/(1-p)), np.ones(len(p))))
    coefficients = np.array([1., 0.])
    prior = coefficients.copy()
    regularization = .01
    for _ in range(60):
        fitted = sigmoid(design @ coefficients)
        gradient = design.T @ (fitted-labels) / len(p) + regularization*(coefficients-prior)
        hessian = (design.T * (fitted*(1-fitted))) @ design / len(p) + regularization*np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        # Limit a Newton step for numerical stability on near-separated data.
        step *= min(1., 1/max(np.linalg.norm(step), 1e-12))
        coefficients -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return {"kind": "logistic", "slope": float(coefficients[0]), "intercept": float(coefficients[1]),
            "epsilon": epsilon, "positives": int(labels.sum()), "regularization": regularization}


def apply_calibration(probabilities, fitted):
    if fitted["kind"] == "constant":
        return np.full_like(probabilities, fitted["value"], dtype=float)
    p = np.clip(probabilities, fitted["epsilon"], 1-fitted["epsilon"])
    return sigmoid(fitted["slope"]*np.log(p/(1-p)) + fitted["intercept"])


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config, scaler, calibration, test, directory):
    directory = Path(directory)
    destination = directory / "evaluation"
    destination.mkdir(exist_ok=False)
    metrics, warning_metrics, reliability, joint_metrics = [], [], [], []
    examples = {"history": test["x"][:8], "target": test["y"][:8],
                "scenario": test["scenario"][:8], "origin_s": test["origin_s"][:8]}
    all_calibrators = {}
    conformal = {}
    threshold = config["demo_thresholds"]
    std = np.asarray(scaler["physical_std"], dtype=np.float32)
    for name in ("persistence", "lstm_point", "energy_lstm", "gan_lstm"):
        probabilistic = name in ("energy_lstm", "gan_lstm")
        if name == "persistence":
            ensemble = np.repeat(test["x"][:, None, -1:, :], config["horizon_steps"], axis=2)
        else:
            model, checkpoint = load_model(directory / name / "best.pt", config["device"])
            ensemble = predict(model, test["x"], config["ensemble_test"], config["batch_size"], config["seed"]+2000)
            if probabilistic:
                cal_ensemble = predict(model, calibration["x"], config["ensemble_test"], config["batch_size"], config["seed"]+3000)
        examples[name] = ensemble[:8]
        np.savez_compressed(destination / f"{name}_test.npz", ensemble=ensemble,
                            target=test["y"], scenario=test["scenario"], origin_s=test["origin_s"])
        mean = ensemble.mean(axis=1)
        scores = crps(ensemble, test["y"]) if probabilistic else None
        if probabilistic:
            low, high = np.quantile(ensemble, [.05, .95], axis=1)
            offsets = conformal_offsets(cal_ensemble, calibration["y"])
            conformal[name] = offsets.tolist()
            conformal_low, conformal_high = low - offsets, high + offsets
            total = 0.
            for start in range(0, len(ensemble), 32):
                p = torch.from_numpy(ensemble[start:start+32] / std)
                y = torch.from_numpy(test["y"][start:start+32] / std)
                total += energy_score(p, y).item() * len(p)
            joint_metrics.append({"model": name, "normalized_trajectory_energy_score": total/len(ensemble)})
        for horizon in HORIZONS:
            index = horizon - 1
            for variable, label in enumerate(VARIABLES):
                error = mean[:, index, variable] - test["y"][:, index, variable]
                rmse_ci = cluster_interval(error**2, test["scenario"], config["bootstrap_replicates"], square_root=True)
                row = {"model": name, "horizon_min": horizon, "variable": label,
                       "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
                       "rmse_ci_low": float(rmse_ci[0]), "rmse_ci_high": float(rmse_ci[1]),
                       "crps": "", "coverage_90": "", "width_90": "", "coverage_90_conformal": "", "width_90_conformal": ""}
                if probabilistic:
                    row.update(crps=float(scores[:, index, variable].mean()),
                        coverage_90=float(((test["y"][:, index, variable] >= low[:, index, variable]) & (test["y"][:, index, variable] <= high[:, index, variable])).mean()),
                        width_90=float((high[:, index, variable]-low[:, index, variable]).mean()),
                        coverage_90_conformal=float(((test["y"][:, index, variable] >= conformal_low[:, index, variable]) & (test["y"][:, index, variable] <= conformal_high[:, index, variable])).mean()),
                        width_90_conformal=float((conformal_high[:, index, variable]-conformal_low[:, index, variable]).mean()))
                metrics.append(row)
            if probabilistic:
                labels = event_labels(test["y"], horizon, threshold)
                raw_events = event_labels(ensemble, horizon, threshold)
                calibration_labels = event_labels(calibration["y"], horizon, threshold)
                calibration_events = event_labels(cal_ensemble, horizon, threshold)
                for event, label in labels.items():
                    raw = raw_events[event].mean(axis=1)
                    fitted = fit_calibration(calibration_events[event].mean(axis=1), calibration_labels[event], config["ensemble_test"])
                    key = f"{name}/{horizon}/{event}"
                    all_calibrators[key] = fitted
                    probability = apply_calibration(raw, fitted)
                    interval = cluster_interval((probability-label)**2, test["scenario"], config["bootstrap_replicates"])
                    warning_metrics.append({"model": name, "horizon_min": horizon, "event": event,
                        "brier_raw": float(np.mean((raw-label)**2)),
                        "brier_calibrated": float(np.mean((probability-label)**2)),
                        "brier_ci_low": float(interval[0]), "brier_ci_high": float(interval[1]),
                        "positive_origins": int(label.sum()), "total_origins": len(label),
                        "positive_scenarios": len(np.unique(test["scenario"][label])),
                        "calibration_kind": fitted["kind"]})
                    for bin_index in range(10):
                        mask = (np.minimum((probability*10).astype(int), 9) == bin_index)
                        reliability.append({"model": name, "horizon_min": horizon, "event": event,
                            "bin": bin_index, "count": int(mask.sum()),
                            "mean_probability": float(probability[mask].mean()) if mask.any() else "",
                            "observed_fraction": float(label[mask].mean()) if mask.any() else ""})
        print(f"evaluated {name}", flush=True)
    write_csv(destination / "forecast_metrics.csv", metrics)
    write_csv(destination / "warning_metrics.csv", warning_metrics)
    write_csv(destination / "reliability.csv", reliability)
    write_csv(destination / "joint_metrics.csv", joint_metrics)
    (destination / "calibrators.json").write_text(json.dumps(all_calibrators, indent=2), encoding="utf-8")
    (destination / "conformal.json").write_text(json.dumps({"alpha": .1, "method": "CQR on calibration split",
        "offsets": conformal}, indent=2), encoding="utf-8")
    np.savez_compressed(destination / "examples.npz", **examples)
    return metrics, warning_metrics, joint_metrics
