"""Absolute wind generation and Fossen-form relative aerodynamic loads."""
import numpy as np


def rotation(heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]])


def kaimal_psd(f, mean_speed, sigma, length_scale):
    """One-sided Kaimal spectrum S(f) [m^2/s^2/Hz] as used in TurbSim/IEC 61400-1."""
    return 4 * sigma**2 * length_scale / mean_speed / (1 + 6 * f * length_scale / mean_speed)**(5 / 3)


def kaimal_fluctuation(time, mean_speed, sigma, length_scale, rng):
    """Kaimal turbulence by spectral synthesis on the record's own FFT grid (TurbSim-style).

    Every frequency k/T up to 0.45/dt gets a complex Gaussian coefficient, so the
    series is Gaussian and does not repeat within the record. A fixed set of
    equally spaced harmonics would repeat every 1/df (under five minutes for
    256 components), which a forecaster could learn instead of the weather.
    """
    n = len(time)
    dt = time[1] - time[0]
    f = np.fft.rfftfreq(n, dt)
    psd = np.where((f > 0) & (f <= 0.45 / dt), kaimal_psd(np.maximum(f, 1e-12), mean_speed, sigma, length_scale), 0.)
    # irfft(n*c) = c_0 + 2 sum Re(c_k e^{i w_k t}); Var[2 Re(c_k e^{..})] = 4 s_k^2 must equal S(f_k) df.
    scale = np.sqrt(psd / (n * dt)) / 2
    coefficients = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) * scale
    return np.fft.irfft(coefficients * n, n)


def simulate_wind(time, config, envelope, rng):
    mean = config["mean_speed_mps"]
    along = mean * envelope + kaimal_fluctuation(time, mean, config["sigma_mps"], config["length_scale_m"], rng)
    across = kaimal_fluctuation(time, mean, 0.8 * config["sigma_mps"], config["length_scale_m"], rng)
    # Direction is velocity TOWARD, CCW from the global +x quay normal.
    return np.column_stack((along, across)) @ rotation(config["direction_toward_rad"]).T


def wind_load(absolute_wind, heading, body_velocity, config):
    """Signed demo coefficient law; replace with measured vessel coefficients.

    gamma denotes velocity TOWARD, not meteorological wind FROM.
    """
    relative = rotation(heading).T @ absolute_wind - np.asarray(body_velocity)[:2]
    speed = np.linalg.norm(relative)
    gamma = np.arctan2(relative[1], relative[0])
    q = 0.5 * config["air_density_kg_m3"] * speed**2
    return q * np.array([
        config["frontal_area_m2"] * config["cx"] * np.cos(gamma),
        config["lateral_area_m2"] * config["cy"] * np.sin(gamma),
        config["lateral_area_m2"] * config["length_m"] * config["cn"] * np.sin(2 * gamma),
    ])
