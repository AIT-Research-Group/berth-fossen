"""Local 3D viewer of simulated berth waves, vessel loads and archived AI forecasts.

Wave fields and spectra are rebuilt from each run's saved configuration and
seed (the same generators ``simulate`` uses); vessel motion and forecasts are
read from saved outputs. Nothing is trained. Serves ``viewer/`` on 127.0.0.1.
"""
import copy
import argparse
import json
import math
import sys
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from berth_sim.simulation import causal_features, simulate, storm_envelopes, wave_field  # noqa: E402
from berth_sim.waves import dispersion, jonswap  # noqa: E402

MAX_POINTS = 250_000  # grid nodes per component, keeps one response near 2-3 MB


def catalog():
    """Map a stable source id to its simulation config; ids never become paths."""
    sources = {}
    for path in sorted((ROOT / "outputs").glob("*/metadata.json")):
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        sources[path.parent.name] = {"label": f"{path.parent.name} (demo)", "split": "demo",
                                     "run": path.parent.name, "config": config}
    for path in sorted((ROOT / "outputs").glob("*/dataset/manifest.json")):
        run = path.parent.parent.name
        for entry in json.loads(path.read_text(encoding="utf-8")).get("scenarios", []):
            key = f"{run}/scenario_{entry['id']:03d}"
            sources[key] = {"label": f"scenario {entry['id']:03d} ({entry['split']})",
                            "split": entry["split"], "run": run, "config": entry["config"]}
    return sources


def describe(key, source):
    config = source["config"]
    waves = config["waves"]
    return {"id": key, "label": source["label"], "run": source["run"], "split": source["split"],
            "duration_s": config["duration_s"], "hs_m": waves["hs_m"], "tp_s": waves["tp_s"],
            "gamma": waves["gamma"], "depth_m": waves["depth_m"], "probe_x_m": waves["probe_x_m"],
            "reflection": waves["reflection"], "f_max_hz": waves["f_max_hz"]}


@lru_cache(maxsize=16)
def cached_field(key):
    return wave_field(catalog()[key]["config"])


def number(query, name, default, low, high):
    value = float(query.get(name, [default])[0])
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


def surface(key, query):
    sources = catalog()
    if key not in sources:
        raise KeyError(key)
    config = sources[key]["config"]
    waves = config["waves"]
    duration = config["duration_s"]
    t0 = number(query, "t0", 0., 0., duration)
    span = number(query, "span", 120., 5., 1800.)
    x_min = number(query, "x_min", -60., -2000., -1.)
    nx = int(number(query, "nx", 161, 11, 801))
    nt = int(number(query, "nt", 481, 11, 4001))
    if nx * nt > MAX_POINTS:
        raise ValueError(f"Grid {nx}x{nt} exceeds {MAX_POINTS} nodes; lower the resolution")
    t1 = min(t0 + span, duration)
    x = np.linspace(x_min, 0., nx)
    t = np.linspace(t0, t1, nt)
    field = cached_field(key)
    envelope = storm_envelopes(t, config["storm"])[1]
    incident, reflected = field.elevation_grid(t, x, envelope)
    total = incident + reflected
    # Stationary spectral profile of Hm0(x), scaled by the mean squared envelope.
    power = field.amplitude**2 / 2 * np.mean(envelope**2)
    kr = field.reflection
    phase = 2 * field.kx[:, None] * x[None, :] - field.reflection_phase[:, None]
    m0_total = (power[:, None] * (1 + kr[:, None]**2 + 2 * kr[:, None] * np.cos(phase))).sum(axis=0)
    wp = 2 * np.pi / waves["tp_s"]
    wavelength = float(2 * np.pi / dispersion(np.array([wp]), waves["depth_m"])[0])
    dt = (t1 - t0) / (nt - 1)
    rounded = lambda values: np.round(values, 4).tolist()
    return {
        "source": describe(key, sources[key]), "x": rounded(x), "t": rounded(t),
        "incident": rounded(incident), "reflected": rounded(reflected), "total": rounded(total),
        "profile": {"hm0_incident": 4 * math.sqrt(power.sum()),
                    "hm0_reflected": 4 * math.sqrt((power * kr**2).sum()),
                    "hm0_total": rounded(4 * np.sqrt(m0_total)),
                    "hm0_total_sample": rounded(4 * total.std(axis=0))},
        "envelope": {"min": float(envelope.min()), "max": float(envelope.max())},
        "peak_wavelength_m": wavelength, "band_energy_fraction": field.energy_fraction,
        "mean_direction_rad": float(np.mean(field.direction)),
        "dt_s": dt, "aliased": bool(dt > 0.5 / waves["f_max_hz"]),
    }


def listed(values):
    """JSON lists with NaN mapped to null (strict JSON has no NaN)."""
    array = np.round(np.asarray(values, dtype=float), 6)
    return np.where(np.isfinite(array), array, None).tolist()


@lru_cache(maxsize=8)
def spectrogram(key):
    """Welch-separated incident/reflected spectra of every feature window.

    Uses ``causal_features`` itself, so the spectra are exactly those behind the
    training features. The vessel is disabled here: it does not affect waves.
    """
    source = catalog()[key]
    config = copy.deepcopy(source["config"])
    config["vessel"]["enabled"] = False
    raw, truth, _, _ = simulate(config, features=False)
    spectra = []
    rows = causal_features(raw["time_s"], raw["wind_global_mps"], raw["probe_observed_m"],
                           truth["reflected_m"], config, spectra)
    waves = config["waves"]
    template = next(result for result in spectra if result is not None)
    frequency = template["frequency_hz"]
    band = (frequency >= waves["f_min_hz"]) & (frequency <= waves["f_max_hz"])
    empty = np.full(band.sum(), np.nan)
    pick = lambda name: [result[name][band] if result is not None else empty for result in spectra]
    # Design JONSWAP per Hz, S_f = 2*pi*S_omega; each window's envelope scales its variance.
    design = 2 * np.pi * jonswap(2 * np.pi * frequency[band], waves["hs_m"], waves["tp_s"], waves["gamma"])
    wave_envelope = storm_envelopes(raw["time_s"], config["storm"])[1]
    nwin = round(config["feature_window_s"] / config["dt_s"])
    step = round(config["feature_step_s"] / config["dt_s"])
    envelope_power = [float(np.mean(wave_envelope[end - nwin + 1:end + 1]**2))
                      for end in range(nwin - 1, len(raw["time_s"]), step)]
    return {
        "source": describe(key, source), "frequency_hz": listed(frequency[band]),
        "time_min": listed([row["time_s"] / 60 for row in rows]),
        "incident_psd": listed(pick("incident_psd")), "reflected_psd": listed(pick("reflected_psd")),
        "design_incident_psd": listed(design), "envelope_power": envelope_power,
        "reflection": float(np.mean(truth["reflection_amplitude"])),
        "hm0_estimate": listed([row["hs_reflected_est_m"] for row in rows]),
        "hm0_truth": listed([row["hs_reflected_truth_window_m"] for row in rows]),
        "quality_ok": [int(row["quality_ok"]) for row in rows],
        "residual": listed([row["mean_separation_residual"] for row in rows]),
        "segment_s": waves.get("welch_segment_s") or config["feature_window_s"],
        "window_s": config["feature_window_s"],
    }


def vessel_runs():
    return {path.parent.name: path for path in sorted((ROOT / "outputs").glob("*/vessel.npz"))}


def vessel(run, query):
    """Saved 3-DOF state and BODY loads, thinned to at most ``points`` samples."""
    runs = vessel_runs()
    if run not in runs:
        raise KeyError(run)
    config = json.loads((runs[run].parent / "metadata.json").read_text(encoding="utf-8"))["config"]
    points = int(number(query, "points", 1800, 100, 20000))
    with np.load(runs[run], allow_pickle=False) as saved:
        step = max(1, int(math.ceil(len(saved["time_s"]) / points)))
        time = saved["time_s"][::step]
        state = saved["state"][::step]
        wind = saved["wind_force_body"][::step]
        wave = saved["wave_force_body"][::step]
    ship = config["vessel"]
    return {
        "run": run, "time_s": listed(time), "x": listed(state[:, 0]), "y": listed(state[:, 1]),
        "psi": listed(state[:, 2]), "u": listed(state[:, 3]), "v": listed(state[:, 4]), "r": listed(state[:, 5]),
        "wind": listed(wind), "wave": listed(wave), "hold": ship["hold_position"],
        "length_m": ship["wind_load"]["length_m"], "probe_x_m": config["waves"]["probe_x_m"],
        "geometry": ship_geometry(config),
        "wave_rao_zero_axes": [axis for axis, a, b in zip("XYN", ship["demo_incident_force_rao"],
                                                         ship["demo_reflected_force_rao"]) if a == 0 and b == 0],
    }


VARIABLES = {
    "environment": [("wind_x_mps", "Gió trung bình x", "m/s", 1.), ("wind_y_mps", "Gió trung bình y", "m/s", 1.),
                    ("gust_mps", "Gió giật", "m/s", 1.), ("reflected_hs_m", "Hm0 phản xạ", "m", 1.)],
    "loads": [(f"{axis}_{stat}", f"{label} {axis}", unit, 1e-3)
              for stat, label in (("rms", "RMS tải"), ("absmax", "Cực đại |tải|"))
              for axis, unit in (("X", "kN"), ("Y", "kN"), ("N", "kNm"))],
}
MODELS = ("persistence", "lstm_point", "energy_lstm", "gan_lstm")


def forecast_runs():
    runs = {}
    for path in sorted((ROOT / "outputs").glob("*/evaluation/gan_lstm_test.npz")):
        run = path.parent.parent
        manifest = run / "dataset" / "manifest.json"
        task = "environment"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("task") == "loads":
            task = "loads"
        with np.load(path, allow_pickle=False) as archive:
            scenarios = sorted(int(s) for s in np.unique(archive["scenario"]))
        runs[run.name] = {"run": run.name, "task": task, "scenarios": scenarios,
                          "models": [m for m in MODELS if (run / "evaluation" / f"{m}_test.npz").exists()],
                          "variables": [{"index": i, "name": v[0], "label": v[1], "unit": v[2]}
                                        for i, v in enumerate(VARIABLES[task])]}
    return runs


def forecast(query):
    runs = forecast_runs()
    run = query.get("run", [""])[0]
    model = query.get("model", ["gan_lstm"])[0]
    if run not in runs or model not in runs[run]["models"]:
        raise KeyError(run)
    info = runs[run]
    scenario = int(number(query, "scenario", info["scenarios"][0], 0, 10**6))
    variable = int(number(query, "variable", 0, 0, len(info["variables"]) - 1))
    scale = VARIABLES[info["task"]][variable][3]
    directory = ROOT / "outputs" / run
    with np.load(directory / "evaluation" / f"{model}_test.npz", allow_pickle=False) as archive:
        ids = np.flatnonzero(archive["scenario"] == scenario)
        if not len(ids):
            raise ValueError("Scenario is not in the test split")
        ensemble = archive["ensemble"][ids, :, :, variable] * scale
        target = archive["target"][ids, :, variable] * scale
        origins = archive["origin_s"][ids] / 60
    # Evaluation archives keep the order of dataset/test.npz, so histories align.
    with np.load(directory / "dataset" / "test.npz", allow_pickle=False) as dataset:
        history = dataset["x"][ids, :, variable] * scale
    mean = ensemble.mean(axis=1)
    low, high = np.quantile(ensemble, [.05, .95], axis=1)
    error = mean - target
    probabilistic = ensemble.shape[1] > 1
    return {
        "run": run, "model": model, "scenario": scenario, "variable": info["variables"][variable],
        "origin_min": listed(origins), "horizon_min": list(range(1, ensemble.shape[2] + 1)),
        "mean": listed(mean), "q05": listed(low) if probabilistic else None,
        "q95": listed(high) if probabilistic else None, "target": listed(target), "history": listed(history),
        "samples": int(ensemble.shape[1]),
        "rmse_by_horizon": listed(np.sqrt(np.mean(error**2, axis=0))),
        "coverage_by_horizon": listed(np.mean((target >= low) & (target <= high), axis=0)) if probabilistic else None,
    }


@lru_cache(maxsize=1)
def hull_proportions():
    """B/L and T/L of the MSS/WAMIT tanker, so the drawn hull has a real tanker's shape."""
    try:
        from berth_sim.hydrodynamics import TankerHydrodynamics
        info = TankerHydrodynamics(ROOT / "data/reference/mss_tanker").provenance()
        return info["beam_m"] / info["length_m"], info["draft_m"] / info["length_m"], info["name"]
    except (ImportError, OSError, KeyError):
        return 0.18, 0.04, "default"


def ship_geometry(config):
    ship = config["vessel"]
    length = ship["wind_load"]["length_m"]
    beam_ratio, draft_ratio, name = hull_proportions()
    x0, y0, psi = ship["hold_position"]
    draft = draft_ratio * length
    return {"length_m": length, "beam_m": beam_ratio * length, "draft_m": draft,
            "freeboard_m": 0.6 * draft, "x_m": x0, "y_m": y0, "heading_rad": psi, "proportions_from": name}


def wavelength_frequency(wavelength, depth, gravity=9.81):
    """Frequency [Hz] of the linear wave with this wavelength at this depth."""
    k = 2 * np.pi / wavelength
    return float(np.sqrt(gravity * k * np.tanh(k * depth)) / (2 * np.pi))


def seastate(key, query):
    """Plan-view sea surface around the moored ship, as animation frames."""
    sources = catalog()
    if key not in sources:
        raise KeyError(key)
    config = sources[key]["config"]
    waves = config["waves"]
    ship = ship_geometry(config)
    duration = config["duration_s"]
    t0 = number(query, "t0", 600., 0., duration)
    step = number(query, "step", .5, .1, 5.)
    frames = int(number(query, "frames", 40, 2, 240))
    n = int(number(query, "n", 140, 20, 200))
    component = query.get("component", ["total"])[0]
    if component not in ("total", "incident", "reflected"):
        raise ValueError("component must be total, incident or reflected")
    length = ship["length_m"]
    # Domain: from the wall out past the ship's seaward side, a ship length along the quay.
    x_lo = min(ship["x_m"] - 0.6 * length, -50.)
    x = np.linspace(x_lo, 0., n)
    y = np.linspace(ship["y_m"] - 0.8 * length, ship["y_m"] + 0.8 * length, n)
    spacing = max(x[1] - x[0], y[1] - y[0])
    t = t0 + step * np.arange(frames)
    if t[-1] > duration:
        raise ValueError("Animation runs past the end of the scenario")
    field = cached_field(key)
    envelope = storm_envelopes(t, config["storm"])[1]
    # Three grid cells per wavelength is the shortest wave the surface can still draw.
    if n * n * frames > 1_200_000:
        raise ValueError("Too many surface values; lower frames or resolution")
    inc, ref, kept = field.elevation_plane(t, x, y, envelope, min_wavelength=3 * spacing)
    surface = {"total": inc + ref, "incident": inc, "reflected": ref}[component]
    # Profiles at full resolution (no wavelength cut): along the keel line, and athwartships
    # from the ship's centre toward the wall, the direction the incident and reflected waves travel.
    c, si = math.cos(ship["heading_rad"]), math.sin(ship["heading_rad"])

    def profile(offsets, ux, uy):
        values = np.empty((frames, len(offsets)))
        for j, r in enumerate(offsets):
            a, b, _ = field.elevation_plane(t, [ship["x_m"] + ux * r], [ship["y_m"] + uy * r], envelope)
            values[:, j] = {"total": a + b, "incident": a, "reflected": b}[component][:, 0, 0]
        return values

    s = np.linspace(-0.7 * length, 0.7 * length, 141)
    line = profile(s, c, si)
    # Starboard unit vector (sin psi, -cos psi); flip it if port faces the wall instead.
    ux, uy = si, -c
    if ux < 0:
        ux, uy = -ux, -uy
    to_wall = -ship["x_m"] / ux if ux > 1e-9 else 0.7 * length
    r = np.linspace(-(ship["beam_m"] / 2 + 0.25 * length), to_wall, 161)
    across = profile(r, ux, uy)
    # Design spectrum per Hz on a fine grid, and the frequencies where lambda = L and lambda = B.
    f = np.linspace(waves["f_min_hz"], waves["f_max_hz"], 400)
    spectrum = 2 * np.pi * jonswap(2 * np.pi * f, waves["hs_m"], waves["tp_s"], waves["gamma"])
    depth = waves["depth_m"]
    wp = 2 * np.pi / waves["tp_s"]
    peak_wavelength = float(2 * np.pi / dispersion(np.array([wp]), depth)[0])
    return {
        "source": describe(key, sources[key]), "ship": ship, "component": component,
        "x": listed(x), "y": listed(y), "t": listed(t), "frames": np.round(surface, 3).tolist(),
        "centerline_s": listed(s), "centerline": np.round(line, 3).tolist(),
        "transverse_r": listed(r), "transverse": np.round(across, 3).tolist(), "wall_r": float(to_wall),
        "kept_variance_fraction": kept, "grid_spacing_m": float(spacing),
        "min_wavelength_m": float(3 * spacing), "f_min_drawn_hz": wavelength_frequency(3 * spacing, waves["depth_m"]), "envelope": listed(envelope),
        "spectrum_f_hz": listed(f), "spectrum": listed(spectrum),
        "f_ship_length_hz": wavelength_frequency(length, depth),
        "f_ship_beam_hz": wavelength_frequency(ship["beam_m"], depth),
        "f_peak_hz": 1 / waves["tp_s"], "peak_wavelength_m": peak_wavelength,
        "probe_x_m": waves["probe_x_m"],
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        query = parse_qs(url.query)
        try:
            if url.path == "/api/sources":
                result = [describe(key, source) for key, source in catalog().items()]
            elif url.path == "/api/surface":
                result = surface(query.get("source", [""])[0], query)
            elif url.path == "/api/spectrogram":
                key = query.get("source", [""])[0]
                if key not in catalog():
                    raise KeyError(key)
                result = spectrogram(key)
            elif url.path == "/api/seastate":
                result = seastate(query.get("source", [""])[0], query)
            elif url.path == "/api/vessel_runs":
                result = list(vessel_runs())
            elif url.path == "/api/vessel":
                result = vessel(query.get("run", [""])[0], query)
            elif url.path == "/api/forecast_runs":
                result = list(forecast_runs().values())
            elif url.path == "/api/forecast":
                result = forecast(query)
            else:
                return self.send_error(404)
        except KeyError:
            return self.send_error(404, "Unknown source")
        except (ValueError, OSError) as exc:
            return self.send_error(422, str(exc))
        payload = json.dumps(result, allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT / "viewer")))
    print(f"Wave surface viewer: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
