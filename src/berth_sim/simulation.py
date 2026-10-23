"""Orchestration and causal features; synthetic scenarios are not field data."""
import numpy as np
from .waves import WaveField, separate_wave_fft
from .wind import simulate_wind, wind_load
from .vessel import first_order_force, simulate_vessel


def validate(config):
    dt = config["dt_s"]
    if dt <= 0 or config["duration_s"] < config["feature_window_s"]:
        raise ValueError("Positive dt and duration >= feature window required")
    for key in ("duration_s", "feature_window_s", "feature_step_s", "gust_window_s"):
        ratio = config[key] / dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError(f"{key} must be a positive integer multiple of dt")
    if config["gust_window_s"] > config["feature_step_s"]:
        raise ValueError("Gust averaging window must fit inside reporting interval")
    if config["feature_step_s"] > config["feature_window_s"] or config["feature_window_s"] / dt < 8:
        raise ValueError("Feature window must contain >=8 samples and cover reporting interval")
    wave = config["waves"]
    if not 0 < wave["f_min_hz"] < wave["f_max_hz"] < 0.5 / dt:
        raise ValueError("Wave band must lie strictly below Nyquist")
    if wave["components"] < 2 or wave["depth_m"] <= 0:
        raise ValueError("Invalid wave resolution or depth")
    # Component spacing must not be coarser than the analysis resolution,
    # otherwise each spectral window sees isolated lines, not a random sea.
    if (wave["f_max_hz"] - wave["f_min_hz"]) / wave["components"] > 1 / config["feature_window_s"] + 1e-12:
        raise ValueError("Wave component spacing must be <= 1/feature_window_s; increase components")
    segment = wave.get("welch_segment_s")
    if segment is not None:
        ratio = segment / dt
        if ratio < 8 or not np.isclose(ratio, round(ratio)) or segment > config["feature_window_s"]:
            raise ValueError("welch_segment_s must be a multiple of dt, >=8 samples and <= feature window")
    if wave.get("spreading_s") is not None and wave["spreading_s"] <= 0:
        raise ValueError("spreading_s must be positive or null for long-crested waves")
    if len(set(wave["probe_x_m"])) < 3 or any(x >= 0 for x in wave["probe_x_m"]):
        raise ValueError("At least 3 distinct probes in water x<0 required")
    if wave["sensor_noise_std_m"] < 0 or not 0 <= wave["missing_probability"] <= 1:
        raise ValueError("Invalid sensor error parameters")
    if wave["max_condition"] <= 1 or config["storm"]["width_s"] <= 0:
        raise ValueError("Invalid conditioning threshold or storm width")
    if min(config["storm"]["wind_gain"], config["storm"]["wave_gain"]) < 0:
        raise ValueError("Demo storm gains must be nonnegative")
    if config["wind"]["mean_speed_mps"] <= 0 or config["wind"]["length_scale_m"] <= 0 or config["wind"]["sigma_mps"] < 0:
        raise ValueError("Invalid wind parameters")
    vessel = config["vessel"]
    if vessel["enabled"]:
        for key in ("mass_diagonal", "linear_damping", "quadratic_damping", "hold_position", "hold_kp", "hold_kd", "control_limits", "demo_incident_force_rao", "demo_reflected_force_rao"):
            if np.shape(vessel[key]) != (3,) or not np.isfinite(vessel[key]).all():
                raise ValueError(f"{key} requires 3 finite values")
        if np.shape(vessel["initial_state"]) != (6,):
            raise ValueError("initial_state requires 6 values")
        if np.any(np.asarray(vessel["mass_diagonal"]) <= 0):
            raise ValueError("Mass matrix must be positive definite")
        for key in ("linear_damping", "quadratic_damping", "hold_kp", "hold_kd", "control_limits"):
            if np.any(np.asarray(vessel[key]) < 0):
                raise ValueError(f"{key} must be nonnegative")


def causal_features(time, wind, observed, truth_reflected, config, spectra=None):
    """One row per reporting step; ``spectra`` (a list) also collects each window's separation."""
    dt = config["dt_s"]
    nwin = round(config["feature_window_s"] / dt)
    nstep = round(config["feature_step_s"] / dt)
    ngust = round(config["gust_window_s"] / dt)
    speed = np.linalg.norm(wind, axis=1)
    wave = config["waves"]
    segment = wave.get("welch_segment_s")
    nseg = nwin if segment is None else round(segment / dt)
    rows = []
    for end in range(nwin - 1, len(time), nstep):
        start = end - nwin + 1
        recent_wind = wind[end - nstep + 1:end + 1]
        recent_speed = speed[end - nstep + 1:end + 1]
        gust = np.convolve(recent_speed, np.ones(ngust) / ngust, mode="valid").max()
        row = {"time_s": time[end], "raw_window_start_s": time[start],
               "wind_x_mean_mps": recent_wind[:, 0].mean(),
               "wind_y_mean_mps": recent_wind[:, 1].mean(),
               "wind_speed_mean_mps": recent_speed.mean(), "wind_gust_mps": gust,
               "hs_reflected_truth_window_m": 4 * np.std(truth_reflected[start:end+1, 0]),
               "hs_reflected_est_m": np.nan, "reflected_band_energy_m2": np.nan,
               "valid_frequency_fraction": 0.0, "mean_separation_residual": np.nan,
               "quality_ok": 0}
        segment = observed[start:end+1]
        result = None
        if np.isfinite(segment).all():
            result = separate_wave_fft(segment, dt, wave["probe_x_m"], wave["depth_m"],
                                       (wave["f_min_hz"], wave["f_max_hz"]), wave["max_condition"],
                                       nseg, wave.get("mean_direction_rad", 0.))
            band = (result["frequency_hz"] >= wave["f_min_hz"]) & (result["frequency_hz"] <= wave["f_max_hz"])
            valid = result["valid"]
            row["valid_frequency_fraction"] = valid.sum() / max(band.sum(), 1)
            if valid.any():
                row["mean_separation_residual"] = result["relative_residual"][valid].mean()
            # Never report a full-band Hs when part of that band was rejected.
            if band.any() and valid[band].all():
                energy = result["reflected_psd"][band].sum() * result["df_hz"]
                row.update(hs_reflected_est_m=4 * np.sqrt(energy),
                           reflected_band_energy_m2=energy, quality_ok=1)
        if spectra is not None:
            spectra.append(result)
        rows.append(row)
    return rows


def storm_envelopes(time, storm):
    """Prescribed slow Gaussian scenario pulses for wind speed and wave amplitude."""
    time = np.asarray(time, dtype=float)
    pulse = lambda center: np.exp(-0.5 * ((time - center) / storm["width_s"])**2)
    return (1 + storm["wind_gain"] * pulse(storm["center_s"]),
            1 + storm["wave_gain"] * pulse(storm["center_s"] + storm["wave_lag_s"]))


def random_streams(config):
    """Independent wind, wave and sensor generators derived from the scenario seed."""
    return [np.random.default_rng(seed) for seed in np.random.SeedSequence(config["seed"]).spawn(3)]


def wave_field(config):
    """Rebuild the exact clean wave field that ``simulate(config)`` uses."""
    return WaveField.create(config["waves"], random_streams(config)[1])


def simulate(config, features=True):
    validate(config)
    streams = random_streams(config)
    time = np.arange(round(config["duration_s"] / config["dt_s"])) * config["dt_s"]
    wind_envelope, wave_envelope = storm_envelopes(time, config["storm"])
    wind = simulate_wind(time, config["wind"], wind_envelope, streams[0])
    field = WaveField.create(config["waves"], streams[1])
    incident, reflected = field.elevation(time, config["waves"]["probe_x_m"], wave_envelope)
    observed = incident + reflected + streams[2].normal(0, config["waves"]["sensor_noise_std_m"], incident.shape)
    observed[streams[2].random(observed.shape) < config["waves"]["missing_probability"]] = np.nan
    raw = {"time_s": time, "wind_global_mps": wind, "probe_observed_m": observed}
    truth = {"incident_m": incident, "reflected_m": reflected,
             "wave_envelope": wave_envelope, "omega_rad_s": field.omega,
             "incident_spectrum_rad": field.spectrum, "reflection_amplitude": field.reflection,
             "component_amplitude_m": field.amplitude, "direction_rad": field.direction,
             "band_energy_fraction": field.energy_fraction}
    vessel_config = config["vessel"]
    vessel = {}
    if vessel_config["enabled"]:
        shape = (len(field.omega), 3)
        wave_force = first_order_force(time, field,
            np.broadcast_to(vessel_config["demo_incident_force_rao"], shape),
            np.broadcast_to(vessel_config["demo_reflected_force_rao"], shape),
            vessel_config["initial_state"][0], wave_envelope)
        state = simulate_vessel(time, wind, wave_force, vessel_config)
        wind_force = np.array([wind_load(w, s[2], s[3:], vessel_config["wind_load"]) for w, s in zip(wind, state)])
        vessel = {"state": state, "wave_force_body": wave_force, "wind_force_body": wind_force,
                  "environment_force_body": wind_force + wave_force}
    rows = causal_features(time, wind, observed, reflected, config) if features else []
    return raw, truth, vessel, rows
