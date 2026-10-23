"""Scenario-isolated data generation; no simulator oracle in model inputs."""
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ..simulation import simulate

FEATURES = ["wind_x_mean_mps", "wind_y_mean_mps", "wind_gust_mps", "hs_reflected_est_m"]
SPLITS = ("train", "validation", "calibration", "test")


# Only these keys define a dataset; training seeds and model settings may change while reusing it.
DATA_KEYS = ("data_seed", "simulation_config", "scenario_counts", "scenario_duration_s", "simulation_dt_s",
             "wave_components", "history_steps", "horizon_steps", "train_stride", "eval_stride")


def data_seed(experiment):
    return experiment.get("data_seed", experiment["seed"])


def configuration_hash(config):
    data = {key: config.get(key) for key in DATA_KEYS}
    data["data_seed"] = data_seed(config)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def make_scenario(base, rng, seed, experiment):
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    config["duration_s"] = experiment["scenario_duration_s"]
    config["dt_s"] = experiment["simulation_dt_s"]
    config["vessel"]["enabled"] = False  # Environmental forecasting scope.
    config["waves"].update(
        components=experiment["wave_components"], hs_m=float(rng.uniform(.25, .95)),
        tp_s=float(rng.uniform(3.5, 7.5)), gamma=float(rng.uniform(1, 5)),
        reflection=float(rng.uniform(.3, .9)), sensor_noise_std_m=float(rng.uniform(.002, .015)))
    # Same berth geometry/depth in all splits; vary environment, not only seed.
    config["wind"].update(mean_speed_mps=float(rng.uniform(4, 11)),
                          sigma_mps=float(rng.uniform(.4, 1.5)),
                          direction_toward_rad=float(rng.uniform(-np.pi, np.pi)))
    duration = config["duration_s"]
    config["storm"].update(center_s=float(rng.uniform(.3, .7) * duration),
        width_s=float(rng.uniform(900, 2400)), wind_gain=float(rng.uniform(.1, 1.1)),
        wave_gain=float(rng.uniform(.1, 1.1)), wave_lag_s=float(rng.uniform(0, 1200)))
    return config


def scenario_plan(base, experiment):
    """Draw every scenario configuration sequentially from the experiment seed.

    Drawing all configurations before simulating keeps the random stream
    identical whether the simulations then run serially or in parallel.
    """
    rng = np.random.default_rng(data_seed(experiment))
    plan = []
    for split in SPLITS:
        for _ in range(experiment["scenario_counts"][split]):
            seed = int(rng.integers(0, 2**31-1))
            plan.append((split, make_scenario(base, rng, seed, experiment)))
    return plan


def map_scenarios(function, configs, workers):
    """Yield ``function(config)`` in input order, using CPU processes if workers > 1."""
    if workers <= 1:
        yield from map(function, configs)
        return
    # One BLAS thread per process avoids oversubscribing the cores.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    with ProcessPoolExecutor(workers) as pool:
        yield from pool.map(function, configs)


def data_workers(experiment):
    return int(experiment.get("data_workers", 1))


def _feature_rows(config):
    return simulate(config)[3]


def cut_windows(values, times, quality, history, horizon, stride):
    xs, ys, origins = [], [], []
    valid = np.isfinite(values).all(axis=1) & quality
    for index in range(history - 1, len(values) - horizon, stride):
        start, stop = index - history + 1, index + horizon + 1
        if not valid[start:stop].all() or not np.allclose(np.diff(times[start:stop]), 60):
            continue
        xs.append(values[start:index+1])
        ys.append(values[index+1:stop])
        origins.append(times[index])
    return (np.asarray(xs, dtype=np.float32).reshape(-1, history, 4),
            np.asarray(ys, dtype=np.float32).reshape(-1, horizon, 4),
            np.asarray(origins))


def prepare_dataset(experiment, directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["experiment_hash"] != configuration_hash(experiment):
            raise ValueError("Dataset configuration differs; choose a new directory")
        return manifest
    directory.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path(experiment["simulation_config"]).read_text(encoding="utf-8-sig"))
    if base["feature_step_s"] != 60:
        raise ValueError("This forecasting experiment requires one-minute feature steps")
    plan = scenario_plan(base, experiment)
    simulated = map_scenarios(_feature_rows, [config for _, config in plan], data_workers(experiment))
    manifest = {"experiment_hash": configuration_hash(experiment), "features": FEATURES,
        "synthetic_only": True, "scenario_split_before_windowing": True,
        "spectral_window_s": base["feature_window_s"],
        "target_window_overlap_note": "5-minute Hs target overlaps 10-minute observed spectral window; 10/30-minute endpoints do not",
        "scenarios": [], "splits": {}}
    scenario_id = 0
    for split in SPLITS:
        arrays = {name: [] for name in ("x", "y", "scenario", "origin_s")}
        unique_rows = []
        for _ in range(experiment["scenario_counts"][split]):
            config = plan[scenario_id][1]
            rows = next(simulated)
            values = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float32)
            times = np.asarray([row["time_s"] for row in rows])
            quality = np.asarray([bool(row["quality_ok"]) for row in rows])
            stride = experiment["train_stride"] if split == "train" else experiment["eval_stride"]
            x, y, origin = cut_windows(values, times, quality,
                experiment["history_steps"], experiment["horizon_steps"], stride)
            if len(x) == 0:
                raise RuntimeError(f"Scenario {scenario_id} produced no valid windows")
            arrays["x"].append(x)
            arrays["y"].append(y)
            arrays["scenario"].append(np.full(len(x), scenario_id, dtype=np.int64))
            arrays["origin_s"].append(origin)
            unique_rows.append(values[quality & np.isfinite(values).all(axis=1)])
            entry = {"id": scenario_id, "split": split, "config": config,
                     "valid_feature_rows": int(quality.sum()), "windows": len(x)}
            manifest["scenarios"].append(entry)
            with (directory / f"scenario_{scenario_id:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
                columns = ["time_s", *FEATURES, "quality_ok"]
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"data {split} scenario={scenario_id} windows={len(x)}", flush=True)
            scenario_id += 1
        combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
        combined["unique_features"] = np.concatenate(unique_rows)
        np.savez_compressed(directory / f"{split}.npz", **combined)
        manifest["splits"][split] = {"scenarios": len(arrays["x"]), "windows": len(combined["x"])}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(directory, split):
    with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
