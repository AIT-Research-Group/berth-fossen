"""Linear random sea and coherent reflection; frequencies in rad/s.

References for the choices below:
- JONSWAP: Hasselmann et al. (1973); normalization A_gamma = 1 - 0.287 ln(gamma)
  from DNV-RP-C205 (2010), sec. 3.5.5, valid for 1 <= gamma <= 7.
- Random-phase synthesis with frequencies jittered inside each bin removes the
  exact 2*pi/d_omega repetition of equally spaced harmonics; optional Rayleigh
  amplitudes (Tucker, Challenor & Carter 1984) give Gaussian-sea statistics when
  the record length is of order 2*pi/d_omega.
- Directional spreading: cos-2s half-angle form (Longuet-Higgins et al. 1963;
  Mitsuyasu et al. 1975), single-summation method (one direction per frequency)
  to avoid phase locking between directions of the same frequency.
- Bulk reflection coefficient option: Zanuttigh & van der Meer (2008), Coastal
  Engineering 55, 771-779; also given in EurOtop (2018).
- Separation: least squares over >=3 probes (Mansard & Funke 1980; Zelt &
  Skjelbreia 1992), averaged over Hann segments with 50% overlap (Welch 1967).
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

GRAVITY = 9.81


def dispersion(omega, depth, gravity=GRAVITY):
    """Solve omega**2 = g*k*tanh(k*h) by bracketed bisection."""
    omega = np.asarray(omega, dtype=float)
    if depth <= 0 or np.any(omega <= 0):
        raise ValueError("Positive depth and angular frequencies required")
    lo = np.zeros_like(omega)
    hi = omega**2 / gravity + omega / np.sqrt(gravity * depth)
    for _ in range(70):
        mid = (lo + hi) / 2
        below = gravity * mid * np.tanh(mid * depth) < omega**2
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return (lo + hi) / 2


def group_velocity(omega, depth, gravity=GRAVITY):
    """Linear group velocity c_g = (omega/k)(1/2)(1 + 2kh/sinh(2kh)) [m/s]."""
    omega = np.asarray(omega, dtype=float)
    k = dispersion(omega, depth, gravity)
    kh2 = np.minimum(2 * k * depth, 700.)
    return omega / k * 0.5 * (1 + kh2 / np.sinh(kh2))


def jonswap(omega, hs, tp, gamma=3.3):
    """One-sided S(omega) [m^2 s/rad] in the DNV-RP-C205 form.

    The integral over 0 < omega < inf equals Hs^2/16 within about 1% for
    1 <= gamma <= 7. No renormalization to the simulated band is applied, so
    band truncation is visible (see ``band_energy_fraction``) instead of being
    hidden by inflating the in-band spectrum.
    """
    omega = np.asarray(omega, dtype=float)
    if hs < 0 or tp <= 0 or not 1 <= gamma <= 7:
        raise ValueError("JONSWAP requires hs >= 0, tp > 0 and 1 <= gamma <= 7")
    if np.any(omega <= 0):
        raise ValueError("Positive angular frequencies required")
    if hs == 0:
        return np.zeros_like(omega)
    wp = 2 * np.pi / tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    peak = np.exp(-0.5 * ((omega / wp - 1) / sigma)**2)
    a_gamma = 1 - 0.287 * np.log(gamma)
    # Log form avoids underflow far below the peak.
    log_s = (np.log(a_gamma * 5 / 16 * hs**2 * wp**4) - 5 * np.log(omega)
             - 1.25 * (wp / omega)**4 + peak * np.log(gamma))
    return np.exp(log_s)


def _integral(values, grid):
    return float(np.sum((values[1:] + values[:-1]) / 2 * np.diff(grid)))


def band_energy_fraction(f_min_hz, f_max_hz, tp, gamma, points=20001):
    """Fraction of the JONSWAP variance represented in [f_min, f_max]."""
    wp = 2 * np.pi / tp
    full = np.linspace(wp / 20, wp * 40, 200001)
    band = np.linspace(2 * np.pi * f_min_hz, 2 * np.pi * f_max_hz, points)
    return _integral(jonswap(band, 1., tp, gamma), band) / _integral(jonswap(full, 1., tp, gamma), full)


def spectral_period_m10(omega, spectrum):
    """Energy period T_{m-1,0} = 2*pi*m_{-1}/m_0 on a uniform grid [s]."""
    return float(2 * np.pi * np.sum(spectrum / omega) / np.sum(spectrum))


def reflection_coefficient(setting, hs, tm10, gravity=GRAVITY):
    """Amplitude reflection coefficient of the berth structure.

    ``setting`` is a number/array (prescribed constant) or a dict:
    {"model": "constant", "value": Kr} or
    {"model": "zanuttigh_van_der_meer", "slope_tan": tan(alpha),
     "roughness_factor": gamma_f}.
    The latter is a bulk empirical relation for sloping structures, fitted to
    laboratory data with scatter; it is not a vertical-wall result. Vertical
    quay walls are usually represented by a prescribed Kr close to 0.9-1.0.
    """
    if not isinstance(setting, dict):
        return np.asarray(setting, dtype=float)
    model = setting.get("model")
    if model == "constant":
        return np.asarray(setting["value"], dtype=float)
    if model == "zanuttigh_van_der_meer":
        slope, gamma_f = setting["slope_tan"], setting["roughness_factor"]
        if slope <= 0 or not 0 < gamma_f <= 1 or hs <= 0:
            raise ValueError("Zanuttigh-van der Meer requires slope>0, 0<gamma_f<=1, hs>0")
        wavelength = gravity * tm10**2 / (2 * np.pi)
        xi = slope / np.sqrt(hs / wavelength)
        a = 0.167 * (1 - np.exp(-3.2 * gamma_f))
        b = 1.49 * (gamma_f - 0.38)**2 + 0.86
        return np.asarray(np.tanh(a * xi**b))
    raise ValueError(f"Unknown reflection model: {model}")


def sample_directions(n, mean, spreading_s, rng, points=4097):
    """Draw one direction per component from a cos-2s spreading function.

    D(theta) ~ cos^{2s}((theta-mean)/2), truncated to cos(theta) > 0 so every
    component travels toward the wall at x=0.
    """
    grid = np.linspace(-np.pi / 2, np.pi / 2, points)[1:-1]
    weight = np.cos((grid - mean) / 2)**(2 * spreading_s)
    cdf = np.concatenate(([0.], np.cumsum((weight[1:] + weight[:-1]) / 2)))
    return np.interp(rng.uniform(0, 1, n), cdf / cdf[-1], grid)


@dataclass
class WaveField:
    omega: np.ndarray
    k: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    reflection: np.ndarray
    reflection_phase: np.ndarray
    spectrum: np.ndarray
    direction: np.ndarray
    energy_fraction: float = 1.0

    @property
    def kx(self):
        """Wavenumber component normal to the wall (probes lie on y = 0)."""
        return self.k * np.cos(self.direction)

    @classmethod
    def create(cls, config, rng):
        n = config["components"]
        edges = np.linspace(config["f_min_hz"], config["f_max_hz"], n + 1)
        width = edges[1] - edges[0]
        d_omega = 2 * np.pi * width
        # Draw order is fixed so a field can be reconstructed from its seed.
        if config.get("frequency_jitter", True):
            frequency = edges[:-1] + rng.uniform(0, 1, n) * width
        else:
            frequency = (edges[:-1] + edges[1:]) / 2
        omega = 2 * np.pi * frequency
        spectrum = jonswap(omega, config["hs_m"], config["tp_s"], config["gamma"])
        fraction = band_energy_fraction(config["f_min_hz"], config["f_max_hz"],
                                        config["tp_s"], config["gamma"])
        if fraction < config.get("min_band_energy_fraction", 0.9):
            raise ValueError(f"Frequency band keeps only {fraction:.1%} of spectral energy")
        phase = rng.uniform(0, 2 * np.pi, n)
        amplitude = np.sqrt(2 * spectrum * d_omega)
        if config.get("random_amplitude", False):
            amplitude *= np.sqrt(rng.exponential(1., n))
        mean = config.get("mean_direction_rad", 0.)
        if abs(mean) >= np.pi / 2:
            raise ValueError("Mean incident direction must travel toward the wall")
        spreading = config.get("spreading_s")
        direction = (np.full(n, float(mean)) if spreading is None
                     else sample_directions(n, mean, spreading, rng))
        design_omega = 2 * np.pi * (edges[:-1] + edges[1:]) / 2
        tm10 = spectral_period_m10(design_omega, jonswap(design_omega, 1., config["tp_s"], config["gamma"]))
        reflection = np.broadcast_to(reflection_coefficient(config["reflection"], config["hs_m"], tm10), (n,)).copy()
        if np.any((reflection < 0) | (reflection > 1)):
            raise ValueError("Passive-wall amplitude reflection must be in [0, 1]")
        return cls(omega, dispersion(omega, config["depth_m"]), amplitude, phase,
                   reflection, np.full(n, config["reflection_phase_rad"]), spectrum,
                   direction, fraction)

    def elevation(self, time, positions, envelope=None):
        """Wall x=0, water x<0; probes on y=0; incident travels toward +x.

        Oblique components reflect specularly, so only k*cos(theta) enters on
        the probe line. The envelope is applied simultaneously at all probes.
        The neglected group delay over a 2*25 m path is about 10-15 s for
        Tp = 4-8 s at 15 m depth, far shorter than the storm widths used
        (>= 300 s), so the quasi-stationary approximation holds there. It is not
        a transient propagation model.
        """
        time = np.asarray(time)
        positions = np.atleast_1d(positions)
        kx = self.kx
        inc = np.empty((len(time), len(positions)))
        ref = np.empty_like(inc)
        for start in range(0, len(time), 2048):
            stop = min(start + 2048, len(time))
            phase_t = self.phase[None, :] - time[start:stop, None] * self.omega
            for j, x in enumerate(positions):
                inc[start:stop, j] = np.cos(phase_t + kx * x) @ self.amplitude
                ref[start:stop, j] = np.cos(phase_t - kx * x + self.reflection_phase) @ (self.amplitude * self.reflection)
        if envelope is not None:
            inc *= np.asarray(envelope)[:, None]
            ref *= np.asarray(envelope)[:, None]
        return inc, ref

    def elevation_grid(self, time, positions, envelope=None):
        """Same field as ``elevation`` on a dense (time, x) grid via one complex matmul.

        eta = Re[sum_i A_i exp(i(phi_i - w_i t)) exp(+-i kx_i x)], cheaper when
        many positions are needed (plots); ``elevation`` stays the reference.
        """
        time = np.asarray(time, dtype=float)
        positions = np.atleast_1d(np.asarray(positions, dtype=float))
        temporal = np.exp(1j * (self.phase[None, :] - time[:, None] * self.omega))
        spatial = np.exp(1j * self.kx[:, None] * positions[None, :])
        inc = (temporal * self.amplitude) @ spatial
        ref = (temporal * (self.amplitude * self.reflection * np.exp(1j * self.reflection_phase))) @ spatial.conj()
        inc, ref = inc.real, ref.real
        if envelope is not None:
            inc *= np.asarray(envelope)[:, None]
            ref *= np.asarray(envelope)[:, None]
        return inc, ref


    def elevation_plane(self, time, x, y, envelope=None, min_wavelength=0.):
        """Incident and specularly reflected elevation on a plan grid, shape (time, y, x).

        Oblique components travel along (cos theta, sin theta); the wall at x=0
        mirrors kx while keeping ky. Components shorter than ``min_wavelength``
        are dropped so a coarse plotting grid does not alias them; the retained
        variance fraction is returned as the third value.
        """
        time = np.asarray(time, dtype=float)
        keep = 2 * np.pi / self.k >= min_wavelength
        kx, ky = self.kx[keep], (self.k * np.sin(self.direction))[keep]
        xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        temporal = np.exp(1j * (self.phase[keep][None, :] - time[:, None] * self.omega[keep]))
        along = np.exp(1j * ky[:, None] * yy.ravel()[None, :])
        incident_space = np.exp(1j * kx[:, None] * xx.ravel()[None, :]) * along
        reflected_space = np.exp(-1j * kx[:, None] * xx.ravel()[None, :]) * along
        amplitude = self.amplitude[keep]
        reflected_amplitude = amplitude * self.reflection[keep] * np.exp(1j * self.reflection_phase[keep])
        inc = ((temporal * amplitude) @ incident_space).real.reshape(len(time), *xx.shape)
        ref = ((temporal * reflected_amplitude) @ reflected_space).real.reshape(len(time), *xx.shape)
        if envelope is not None:
            inc *= np.asarray(envelope)[:, None, None]
            ref *= np.asarray(envelope)[:, None, None]
        variance = self.amplitude**2
        return inc, ref, float(variance[keep].sum() / variance.sum())


@lru_cache(maxsize=128)
def _separation_geometry(n, dt, positions, depth, band_hz, max_condition, incidence):
    frequency = np.fft.rfftfreq(n, dt)
    candidate = np.flatnonzero((frequency >= band_hz[0]) & (frequency <= band_hz[1]) & (frequency > 0))
    ks = dispersion(2 * np.pi * frequency[candidate], depth) * np.cos(incidence)
    phase = ks[:, None] * np.asarray(positions)[None, :]
    matrix = np.stack((np.exp(-1j * phase), np.exp(1j * phase)), axis=-1)
    conditions = np.linalg.cond(matrix)
    accepted = conditions <= max_condition
    good = matrix[accepted]
    return frequency, candidate, conditions, candidate[accepted], good, np.linalg.pinv(good)


def separate_wave_fft(elevation, dt, positions, depth, band_hz,
                      max_condition=100.0, segment_samples=None, incidence_rad=0.0):
    """Least-squares incident/reflected separation with Welch averaging.

    Each Hann segment (50% overlap) is separated per frequency; the separated
    powers are averaged, which equals applying the least-squares operator to
    the Welch cross-spectral matrix. ``segment_samples=None`` uses one window.
    ``incidence_rad`` is the assumed mean direction; directional spread about
    it is not modelled by the method and appears as residual/bias.

    NumPy rFFT uses exp(-i wt): its positive-frequency coefficients
    therefore have spatial columns exp(-ikx), exp(+ikx) for our waves.
    Returns invalid bins as NaN, never silently as measured zero energy.
    """
    elevation = np.asarray(elevation, dtype=float)
    positions = np.asarray(positions)
    if elevation.ndim != 2 or elevation.shape[1] != len(positions) or len(positions) < 3:
        raise ValueError("At least three synchronized probes required")
    if len(np.unique(positions)) != len(positions) or dt <= 0 or len(elevation) < 8:
        raise ValueError("Distinct probe positions, positive dt and >=8 samples required")
    if not np.isfinite(elevation).all():
        raise ValueError("Wave separation cannot consume missing samples")
    total = len(elevation)
    n = total if segment_samples is None else int(segment_samples)
    if not 8 <= n <= total:
        raise ValueError("Segment must hold >=8 samples and fit in the record")
    starts = range(0, total - n + 1, max(n // 2, 1))
    win = np.hanning(n)
    segments = np.stack([elevation[s:s+n] - elevation[s:s+n].mean(axis=0) for s in starts])
    transformed = np.fft.rfft(segments * win[None, :, None], axis=1)
    frequency, candidate, conditions, accepted, matrix, inverse = _separation_geometry(
        n, dt, tuple(positions), depth, tuple(band_hz), max_condition, float(incidence_rad))
    si = np.full(len(frequency), np.nan)
    sr = si.copy()
    condition = np.full(len(frequency), np.inf)
    residual = si.copy()
    condition[candidate] = conditions
    scale = dt / np.sum(win**2)
    if len(accepted):
        observed = transformed[:, accepted]
        a = np.einsum('fij,sfj->sfi', inverse, observed)
        factor = np.where((n % 2 == 0) & (accepted == n // 2), 1, 2)
        power = factor[:, None] * scale * (np.abs(a)**2).mean(axis=0)
        si[accepted], sr[accepted] = power[:, 0], power[:, 1]
        reconstruction = np.einsum('fij,sfj->sfi', matrix, a)
        misfit = np.sum(np.abs(reconstruction - observed)**2, axis=(0, 2))
        energy = np.sum(np.abs(observed)**2, axis=(0, 2))
        residual[accepted] = np.sqrt(misfit) / np.maximum(np.sqrt(energy), 1e-15)
    return {"frequency_hz": frequency, "incident_psd": si, "reflected_psd": sr,
            "valid": np.isfinite(sr), "condition": condition, "relative_residual": residual,
            "df_hz": 1 / (n * dt), "segments": len(segments)}
