"""Figures (300 dpi) and table data for the results section, from saved experiment outputs.

Reads outputs/training_v2 (environmental forecasting) and outputs/force_training_v2
(load forecasting). Nothing is retrained; the test scenarios are re-simulated only to
recover the clean reflected-wave truth for the separation-accuracy figure.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from berth_sim.simulation import simulate  # noqa: E402
from berth_sim.training.evaluation import apply_calibration, cluster_interval, crps, event_labels  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from plot_analysis import confusion  # noqa: E402

MODELS = ("persistence", "lstm_point", "energy_lstm", "gan_lstm")
NAMES = {"persistence": "Persistence", "lstm_point": "LSTM (point)", "energy_lstm": "Energy-LSTM", "gan_lstm": "GAN-LSTM"}
# MATLAB default line colours, one per model.
COLORS = {"persistence": "#7f7f7f", "lstm_point": "#d95319", "energy_lstm": "#0072bd", "gan_lstm": "#77ac30"}
SPLIT_COLORS = {"train": "#0072bd", "validation": "#edb120", "calibration": "#7e2f8e", "test": "#a2142f"}
ENV_VARIABLES = ["Mean wind x (m/s)", "Mean wind y (m/s)", "Gust (m/s)", "Reflected $H_{m0}$ (m)"]
WIDTH = 6.5  # inches, full text width

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
                     "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": True, "axes.spines.right": True, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "figure.constrained_layout.use": True})


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(figure, out, name):
    figure.savefig(out / name)
    plt.close(figure)
    return name


def scenario_rows(manifest):
    rows = []
    for entry in manifest["scenarios"]:
        c = entry["config"]
        rows.append({"id": entry["id"], "split": entry["split"], "hs": c["waves"]["hs_m"], "tp": c["waves"]["tp_s"],
                     "gamma": c["waves"]["gamma"], "kr": c["waves"]["reflection"],
                     "noise_mm": 1000 * c["waves"]["sensor_noise_std_m"], "u": c["wind"]["mean_speed_mps"],
                     "sigma_u": c["wind"]["sigma_mps"], "direction_deg": math.degrees(c["wind"]["direction_toward_rad"]),
                     "storm_center_min": c["storm"]["center_s"] / 60, "storm_width_min": c["storm"]["width_s"] / 60,
                     "wind_gain": c["storm"]["wind_gain"], "wave_gain": c["storm"]["wave_gain"],
                     "wave_lag_min": c["storm"]["wave_lag_s"] / 60, "windows": entry["windows"],
                     "valid_rows": entry.get("valid_feature_rows")})
    return rows


def figure_scenarios(rows, out):
    figure, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))
    pairs = [("hs", "tp", "Design $H_s$ (m)", "$T_p$ (s)"), ("kr", "gamma", "Reflection coefficient $K_r$", r"JONSWAP $\gamma$"),
             ("u", "sigma_u", "Mean wind speed $U$ (m/s)", r"Turbulence $\sigma_u$ (m/s)"),
             ("wind_gain", "wave_gain", "Storm wind gain", "Storm wave gain")]
    for axis, (kx, ky, lx, ly), tag in zip(axes.flat, pairs, "abcd"):
        for split, color in SPLIT_COLORS.items():
            subset = [r for r in rows if r["split"] == split]
            axis.scatter([r[kx] for r in subset], [r[ky] for r in subset], s=22 if split == "test" else 14, color=color,
                         marker="D" if split == "test" else "o", label=f"{split} ({len(subset)})",
                         edgecolor="k" if split == "test" else "none", linewidth=.5)
        axis.set(xlabel=lx, ylabel=ly, title=f"({tag})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    return save(figure, out, "fig01_scenario_design.png")


def dataset_dir(run):
    """Dataset used by a run: a shared directory (v3) or the run's own dataset/ folder."""
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    path = Path(experiment.get("dataset") or run / "dataset")
    return path if path.is_absolute() else ROOT / path


def representative_tests(manifest, dataset, count=8):
    """Test scenarios spread evenly over the range of peak reflected Hm0, for grid figures."""
    tests = [e["id"] for e in manifest["scenarios"] if e["split"] == "test"]
    peaks = []
    for sid in tests:
        rows = read_csv(dataset / f"scenario_{sid:03d}.csv")
        peaks.append(max(float(r["hs_reflected_est_m"]) for r in rows if r["quality_ok"] == "1"))
    order = [tests[i] for i in np.argsort(peaks)]
    if len(order) <= count:
        return order
    return sorted(order[int(round(i))] for i in np.linspace(0, len(order) - 1, count))


def figure_test_series(manifest, dataset, selected, thresholds, out):
    tests = [e for e in manifest["scenarios"] if e["id"] in selected]
    figure, axes = plt.subplots(4, 2, figsize=(WIDTH, 6.4), sharex=True)
    for axis, entry in zip(axes.flat, tests):
        data = read_csv(dataset / f"scenario_{entry['id']:03d}.csv")
        t = [float(r["time_s"]) / 60 for r in data]
        hs = [float(r["hs_reflected_est_m"]) if r["quality_ok"] == "1" else np.nan for r in data]
        gust = [float(r["wind_gust_mps"]) for r in data]
        axis.plot(t, hs, color="#0072bd", linewidth=1)
        axis.axhline(thresholds["reflected_hs_m"], color="#0072bd", linestyle=":", linewidth=.8)
        axis.set_ylabel("$H_{m0,R}$ (m)", color="#0072bd")
        twin = axis.twinx()
        twin.plot(t, gust, color="#d95319", linewidth=.8)
        twin.axhline(thresholds["gust_mps"], color="#d95319", linestyle=":", linewidth=.8)
        twin.set_ylabel("Gust (m/s)", color="#d95319")
        twin.grid(False)
        axis.set_title(f"Test scenario {entry['id']:03d}", loc="left")
    for axis in axes[-1]:
        axis.set_xlabel("Time (min)")
    return save(figure, out, "fig02_test_scenarios.png")


def separation_accuracy(manifest):
    """Re-simulate test scenarios to compare Hm0,R estimated from 3 probes with clean truth."""
    result = []
    for entry in manifest["scenarios"]:
        if entry["split"] != "test":
            continue
        _, _, _, rows = simulate(entry["config"])
        est = np.array([r["hs_reflected_est_m"] for r in rows], dtype=float)
        truth = np.array([r["hs_reflected_truth_window_m"] for r in rows], dtype=float)
        ok = np.array([r["quality_ok"] for r in rows], dtype=bool) & np.isfinite(est)
        error = est[ok] / truth[ok] - 1
        result.append({"id": entry["id"], "est": est[ok].tolist(), "truth": truth[ok].tolist(),
                       "bias_pct": 100 * float(error.mean()), "std_pct": 100 * float(error.std()),
                       "rmse_m": float(np.sqrt(np.mean((est[ok] - truth[ok])**2))),
                       "valid": int(ok.sum()), "total": len(rows)})
    return result


def figure_separation(accuracy, out):
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={"width_ratios": [1, 1.3]})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    hi = 0
    for n, item in enumerate(accuracy):
        axes[0].scatter(item["truth"], item["est"], s=4, color=colors[n % 10], label=f"{item['id']:03d}")
        hi = max(hi, max(item["truth"]), max(item["est"]))
    axes[0].plot([0, hi * 1.05], [0, hi * 1.05], "k--", linewidth=.8)
    axes[0].set(xlabel="True $H_{m0,R}$ (m)", ylabel="Estimated $H_{m0,R}$ (m)", title="(a) Three-probe estimate vs truth",
                xlim=(0, hi * 1.05), ylim=(0, hi * 1.05), aspect="equal")
    axes[0].legend(ncol=2, markerscale=2, title="Scenario", fontsize=6.5)
    errors = [100 * (np.array(i["est"]) / np.array(i["truth"]) - 1) for i in accuracy]
    axes[1].boxplot(errors, tick_labels=[f"{i['id']:03d}" for i in accuracy], showfliers=False)
    axes[1].axhline(0, color="k", linewidth=.8)
    axes[1].set(xlabel="Test scenario", ylabel="Relative error (%)", title="(b) Error per scenario (Welch 150 s)")
    return save(figure, out, "fig03_separation_accuracy.png")


def histories(run):
    return {name: read_csv(run / name / "history.csv") for name in MODELS[1:]}


def figure_training(run, title, out, name):
    data = histories(run)
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5))
    for axis, (model, rows) in zip(axes, data.items()):
        epoch = [int(r["epoch"]) for r in rows]
        train = [float(r["train_loss"]) for r in rows]
        val = [float(r["validation_score"]) for r in rows]
        axis.plot(epoch, train, color="#0072bd", label="Training loss")
        axis.plot(epoch, val, color="#d95319", label="Validation score")
        best = int(np.argmin(val))
        axis.plot(epoch[best], val[best], "k*", markersize=7, label="Best validation (selected checkpoint)")
        if model == "gan_lstm":
            twin = axis.twinx()
            twin.plot(epoch, [float(r["discriminator_loss"]) for r in rows], color="#7f7f7f", linewidth=.8,
                      label="Discriminator loss")
            twin.set_ylabel("Discriminator BCE", color="#7f7f7f")
            twin.grid(False)
        axis.set(xlabel="Epoch", title=NAMES[model], ylabel="MSE" if model == "lstm_point" else "Energy score / loss")
        axis.text(.97, .95, f"best epoch {epoch[best]}", transform=axis.transAxes, ha="right", va="top", fontsize=7,
                  bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "edgecolor": "none", "alpha": .8})
    # One shared legend under the panels keeps the curves unobstructed.
    handles, labels = axes[0].get_legend_handles_labels()
    extra = twin.get_legend_handles_labels() if "twin" in locals() else ([], [])
    figure.legend(handles + extra[0], [l.split(" (")[0] for l in labels] + extra[1],
                  loc="outside lower center", ncol=4)
    figure.suptitle(title, fontsize=9)
    return save(figure, out, name)


def per_horizon(run, variable, scale=1.):
    """RMSE with scenario-bootstrap CI, CRPS and 90% coverage for every horizon 1..30."""
    result = {}
    for model in MODELS:
        with np.load(run / "evaluation" / f"{model}_test.npz") as archive:
            ensemble = archive["ensemble"][..., variable] * scale
            target = archive["target"][..., variable] * scale
            scenario = archive["scenario"]
        mean = ensemble.mean(axis=1)
        rmse, low, high = [], [], []
        for h in range(target.shape[1]):
            error = (mean[:, h] - target[:, h])**2
            rmse.append(float(np.sqrt(error.mean())))
            ci = cluster_interval(error, scenario, 400, square_root=True)
            low.append(float(ci[0])); high.append(float(ci[1]))
        entry = {"rmse": rmse, "rmse_low": low, "rmse_high": high}
        if ensemble.shape[1] > 1:
            q05, q95 = np.quantile(ensemble, [.05, .95], axis=1)
            entry["crps"] = crps(ensemble, target).mean(axis=0).tolist()
            entry["coverage"] = ((target >= q05) & (target <= q95)).mean(axis=0).tolist()
            entry["width"] = (q95 - q05).mean(axis=0).tolist()
        result[model] = entry
    return result


def figure_env_skill(run, out):
    gust, hs = per_horizon(run, 2), per_horizon(run, 3)
    horizons = np.arange(1, 31)
    figure, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.8))
    for column, (label, data) in enumerate((("Gust (m/s)", gust), ("Reflected $H_{m0}$ (m)", hs))):
        axis = axes[0, column]
        for model in MODELS:
            d = data[model]
            axis.plot(horizons, d["rmse"], color=COLORS[model], label=NAMES[model])
            axis.fill_between(horizons, d["rmse_low"], d["rmse_high"], color=COLORS[model], alpha=.12, linewidth=0)
        axis.set(xlabel="Lead time (min)", ylabel=f"RMSE, {label}", title=f"({'ab'[column]}) RMSE, 95% scenario-bootstrap CI")
        axis = axes[1, column]
        for model in ("energy_lstm", "gan_lstm"):
            axis.plot(horizons, data[model]["crps"], color=COLORS[model], label=f"CRPS {NAMES[model]}")
        axis.set(xlabel="Lead time (min)", ylabel=f"CRPS, {label}", title=f"({'cd'[column]}) CRPS and 90% coverage")
        twin = axis.twinx()
        for model in ("energy_lstm", "gan_lstm"):
            twin.plot(horizons, 100 * np.array(data[model]["coverage"]), color=COLORS[model], linestyle="--",
                      label=f"Coverage {NAMES[model]}")
        twin.axhline(90, color="k", linestyle=":", linewidth=.8)
        twin.set(ylim=(0, 100), ylabel="Coverage of 90% interval (%)")
        twin.grid(False)
        if column == 1:
            lines = axis.get_legend_handles_labels()
            more = twin.get_legend_handles_labels()
            axis.legend(lines[0] + more[0], lines[1] + more[1], loc="lower right", fontsize=6.5)
    axes[0, 0].legend(loc="upper left")
    return save(figure, out, "fig06_env_skill_by_lead_time.png"), {"gust": gust, "hs": hs}


def figure_examples(run, out):
    with np.load(dataset_dir(run) / "test.npz") as dataset:
        x, y, scenario, origin = dataset["x"], dataset["y"], dataset["scenario"], dataset["origin_s"]
    # Issue time with the largest observed reflected Hm0 in the next 30 min: the hardest, most relevant case.
    pick = int(np.argmax(y[:, :, 3].max(axis=1)))
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    for axis, variable, label in zip(axes, (3, 2), ("Reflected $H_{m0}$ (m)", "Gust (m/s)")):
        past = np.arange(-x.shape[1] + 1, 1)
        future = np.arange(1, y.shape[1] + 1)
        axis.plot(past, x[pick, :, variable], "k", linewidth=1, label="Observed history")
        axis.plot(future, y[pick, :, variable], "k--", linewidth=1, label="Observed future")
        for model in ("energy_lstm", "gan_lstm"):
            with np.load(run / "evaluation" / f"{model}_test.npz") as archive:
                samples = archive["ensemble"][pick, :, :, variable]
            q05, q95 = np.quantile(samples, [.05, .95], axis=0)
            axis.plot(future, samples.mean(axis=0), color=COLORS[model], label=f"{NAMES[model]} mean")
            axis.fill_between(future, q05, q95, color=COLORS[model], alpha=.18, linewidth=0, label=f"{NAMES[model]} 90%")
        axis.axvline(0, color="#7f7f7f", linewidth=.8)
        axis.set(xlabel="Minutes relative to issue time", ylabel=label)
    axes[0].legend(fontsize=6.5, loc="upper left")
    figure.suptitle(f"Test scenario {int(scenario[pick]):03d}, issued at {origin[pick] / 60:.0f} min "
                    "(largest reflected $H_{m0}$ ahead)", fontsize=9)
    return save(figure, out, "fig07_forecast_examples.png"), {"scenario": int(scenario[pick]), "origin_min": float(origin[pick] / 60)}


def figure_reliability(run, out):
    rows = [r for r in read_csv(run / "evaluation" / "reliability.csv") if r["horizon_min"] == "30"]
    events = ("wind", "wave", "any")
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    for axis, event in zip(axes, events):
        axis.plot([0, 1], [0, 1], "k:", linewidth=.8)
        for model in ("energy_lstm", "gan_lstm"):
            subset = [r for r in rows if r["model"] == model and r["event"] == event and r["mean_probability"]]
            p = [float(r["mean_probability"]) for r in subset]
            o = [float(r["observed_fraction"]) for r in subset]
            n = [int(r["count"]) for r in subset]
            axis.plot(p, o, "-", color=COLORS[model], linewidth=.8)
            axis.scatter(p, o, s=[6 + 1.5 * math.sqrt(k) * 4 for k in n], color=COLORS[model], label=NAMES[model],
                         edgecolor="k", linewidth=.3)
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Forecast probability", title=f"{event} event, 30 min")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left")
    return save(figure, out, "fig08_reliability_30min.png")


EVENTS = {"wind": "Gust event", "wave": "Reflected $H_{m0}$ event", "any": "Either event", "simultaneous": "Both events"}


def warning_confusion(run, thresholds, out, horizon=30, decision=0.5):
    """Confusion matrices of 30-min exceedance warnings on the test set.

    Probabilistic models warn when the calibrated probability (logistic fit on the
    calibration split) reaches ``decision``; the threshold is fixed a priori, not tuned
    on test. Persistence and the point LSTM give yes/no forecasts directly.
    """
    calibrators = json.loads((run / "evaluation" / "calibrators.json").read_text(encoding="utf-8"))
    rows, matrices = [], {}
    for model in MODELS:
        with np.load(run / "evaluation" / f"{model}_test.npz") as archive:
            ensemble, target, scenario = archive["ensemble"], archive["target"], archive["scenario"]
        truth = event_labels(target, horizon, thresholds)
        members = event_labels(ensemble, horizon, thresholds)
        for event in EVENTS:
            probability = members[event].mean(axis=1)
            key = f"{model}/{horizon}/{event}"
            if key in calibrators:
                probability = apply_calibration(probability, calibrators[key])
            predicted = probability >= decision
            result = confusion(truth[event], predicted)
            result.update(model=model, event=event,
                          positive_scenarios=int(len(np.unique(scenario[truth[event]]))),
                          warned_scenarios=int(len(np.unique(scenario[predicted]))))
            rows.append(result)
            matrices[(model, event)] = result
    shown = ("wind", "wave", "any")
    figure, axes = plt.subplots(len(shown), len(MODELS), figsize=(WIDTH, 5.0))
    for r, event in enumerate(shown):
        for c, model in enumerate(MODELS):
            m = matrices[(model, event)]
            grid = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
            axis = axes[r, c]
            share = grid / grid.sum(axis=1, keepdims=True).clip(min=1)
            axis.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, f"{grid[i, j]}\n{100 * share[i, j]:.0f}%", ha="center", va="center", fontsize=7,
                              color="white" if share[i, j] > .6 else "black")
            axis.set_xticks([0, 1], ["Warn", "No warn"])
            axis.set_yticks([0, 1], ["Event", "No event"])
            axis.tick_params(length=0, labelsize=6.5)
            axis.grid(False)
            if r == 0:
                axis.set_title(NAMES[model], fontsize=8.5)
            if c == 0:
                axis.set_ylabel(EVENTS[event], fontsize=8)
            recall = "—" if m["recall"] is None else f"{m['recall']:.2f}"
            precision = "—" if m["precision"] is None else f"{m['precision']:.2f}"
            axis.set_xlabel(f"POD {recall}, precision {precision}", fontsize=6.5)
    return save(figure, out, "fig09_confusion_30min.png"), rows


def figure_loads(run, out):
    targets = [(1, "RMS $F_Y$ (kN)", 1e-3), (2, "RMS $M_N$ (kNm)", 1e-3), (4, "Max $|F_Y|$ (kN)", 1e-3), (5, "Max $|M_N|$ (kNm)", 1e-3)]
    horizons = np.arange(1, 31)
    figure, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))
    summary = {}
    for axis, (variable, label, scale), tag in zip(axes.flat, targets, "abcd"):
        data = per_horizon(run, variable, scale)
        summary[label] = data
        for model in MODELS:
            axis.plot(horizons, data[model]["rmse"], color=COLORS[model], label=NAMES[model])
            axis.fill_between(horizons, data[model]["rmse_low"], data[model]["rmse_high"], color=COLORS[model],
                              alpha=.12, linewidth=0)
        axis.set(xlabel="Lead time (min)", ylabel=f"RMSE, {label}", title=f"({tag}) {label}")
    axes[0, 0].legend(loc="lower right")
    return save(figure, out, "fig10_load_skill_by_lead_time.png"), summary


def figure_ablation(ablation, out):
    """Base vs upgraded training, mean ± std over seeds (dots: individual seeds)."""
    panels = [("hs/30/rmse", "RMSE, reflected $H_{m0}$ (m)", "persistence/hs/30/rmse"),
              ("gust/30/rmse", "RMSE, gust (m/s)", "persistence/gust/30/rmse"),
              ("hs/30/crps", "CRPS, reflected $H_{m0}$ (m)", None),
              ("joint_es", "Trajectory energy score", None),
              ("brier_wave", "Brier score, $H_{m0}$ event", None),
              ("f1_wave", "F1, $H_{m0}$ event warning", "persistence/f1_wave")]
    variants = (("base", "Base", "#bdbdbd"), ("upgraded", "Upgraded", "#0072bd"))
    models = ("energy_lstm", "gan_lstm")
    figure, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.4))
    for axis, (metric, label, reference), tag in zip(axes.flat, panels, "abcdef"):
        for g, model in enumerate(models):
            for v, (variant, name, color) in enumerate(variants):
                m = ablation[variant]["metrics"][f"{model}/{metric}"]
                x = g + (v - .5) * .36
                axis.bar(x, m["mean"], .34, yerr=m["std"], color=color, edgecolor="k", linewidth=.5, capsize=3,
                         label=name if g == 0 else None)
                axis.scatter([x] * len(m["values"]), m["values"], s=8, color="k", zorder=3)
        if reference:
            value = ablation["base"]["metrics"][reference]["mean"]
            axis.axhline(value, color="#7f7f7f", linestyle="--", linewidth=1, label="Persistence")
        values = [ablation[v]["metrics"][f"{m}/{metric}"] for v, _, _ in variants for m in models]
        low = min(x["mean"] - x["std"] for x in values)
        high = max(max(x["values"]) for x in values)
        if reference:
            high = max(high, ablation["base"]["metrics"][reference]["mean"])
            low = min(low, ablation["base"]["metrics"][reference]["mean"])
        pad = .15 * (high - low)
        axis.set_ylim(low - pad, high + pad)
        axis.set_xticks(range(len(models)), [NAMES[m] for m in models])
        axis.set_title(f"({tag}) {label}", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    return save(figure, out, "fig13_ablation.png")


def training_summary(run):
    result = {}
    for model in MODELS[1:]:
        summary = json.loads((run / model / "training_summary.json").read_text(encoding="utf-8"))
        rows = read_csv(run / model / "history.csv")
        val = [float(r["validation_score"]) for r in rows]
        summary["best_epoch"] = int(np.argmin(val)) + 1
        summary["final_train_loss"] = float(rows[-1]["train_loss"])
        result[model] = summary
    return result


def main():
    parser = argparse.ArgumentParser()
    # v3: representative run = median-seed upgraded model (by GAN-LSTM joint energy score).
    parser.add_argument("--env", type=Path, default=ROOT / "outputs/v3_upgraded_s2")
    parser.add_argument("--ablation", type=Path, default=ROOT / "outputs/paper_results/ablation_v3.json")
    parser.add_argument("--loads", type=Path, default=ROOT / "outputs/force_training_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_results")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    env_experiment = json.loads((args.env / "experiment.json").read_text(encoding="utf-8"))
    load_experiment = json.loads((args.loads / "experiment.json").read_text(encoding="utf-8"))
    config = env_experiment["config"]
    dataset = dataset_dir(args.env)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    load_manifest = json.loads((dataset_dir(args.loads) / "manifest.json").read_text(encoding="utf-8"))
    selected = representative_tests(manifest, dataset)
    base = json.loads((ROOT / config["simulation_config"]).read_text(encoding="utf-8-sig"))
    rows = scenario_rows(manifest)
    figures = {}
    figures["scenarios"] = figure_scenarios(rows, out)
    figures["test_series"] = figure_test_series(manifest, dataset, selected, config["demo_thresholds"], out)
    print("simulating test scenarios for separation accuracy", flush=True)
    accuracy = separation_accuracy(manifest)
    figures["separation"] = figure_separation(accuracy, out)
    figures["training_env"] = figure_training(args.env, "Environmental forecasting (wind, gust, reflected $H_{m0}$)",
                                              out, "fig04_training_environment.png")
    figures["training_loads"] = figure_training(args.loads, "Load forecasting (RMS and max of $F_X$, $F_Y$, $M_N$)",
                                                out, "fig05_training_loads.png")
    figures["env_skill"], env_skill = figure_env_skill(args.env, out)
    figures["examples"], example = figure_examples(args.env, out)
    figures["reliability"] = figure_reliability(args.env, out)
    figures["confusion"], confusion_rows = warning_confusion(args.env, config["demo_thresholds"], out)
    figures["loads"], load_skill = figure_loads(args.loads, out)
    ablation = json.loads(args.ablation.read_text(encoding="utf-8")) if args.ablation.exists() else None
    if ablation:
        figures["ablation"] = figure_ablation(ablation, out)
    tables = {
        "base_config": base, "experiment": config, "load_experiment": load_experiment["config"],
        "env_versions": env_experiment.get("versions"), "gpu": env_experiment.get("gpu"), "cuda": env_experiment.get("cuda"),
        "splits": manifest["splits"], "load_splits": load_manifest["splits"],
        "load_hydro": load_manifest.get("hydro_reference"), "scenarios": rows, "separation": [
            {k: v for k, v in item.items() if k not in ("est", "truth")} for item in accuracy],
        "training_env": training_summary(args.env), "training_loads": training_summary(args.loads),
        "forecast_metrics": read_csv(args.env / "evaluation" / "forecast_metrics.csv"),
        "joint_metrics": read_csv(args.env / "evaluation" / "joint_metrics.csv"),
        "warning_metrics": read_csv(args.env / "evaluation" / "warning_metrics.csv"),
        "load_metrics": read_csv(args.loads / "evaluation" / "load_metrics.csv"),
        "env_skill_30": {var: {m: {k: v[-1] for k, v in d.items()} for m, d in data.items()} for var, data in env_skill.items()},
        "load_skill_30": {var: {m: {k: v[-1] for k, v in d.items()} for m, d in data.items()} for var, data in load_skill.items()},
        "example": example, "figures": figures, "confusion": confusion_rows,
        "env_run": args.env.name, "load_run": args.loads.name, "selected_tests": selected, "ablation": ablation,
    }
    (out / "tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "figures": list(figures.values())}), flush=True)


if __name__ == "__main__":
    main()
