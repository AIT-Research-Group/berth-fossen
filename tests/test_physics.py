import copy
import json
from pathlib import Path
import unittest
import numpy as np
from berth_sim.waves import (dispersion, jonswap, WaveField, separate_wave_fft, band_energy_fraction,
                             sample_directions, reflection_coefficient)
from berth_sim.wind import wind_load
from berth_sim.vessel import coriolis, rhs, simulate_vessel, first_order_force
from berth_sim.simulation import simulate, causal_features, validate


ROOT = Path(__file__).resolve().parents[1]


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/demo.json").read_text())

    def test_dispersion(self):
        omega = np.linspace(.1, 8, 50)
        for depth in (1., 15., 100.):
            k = dispersion(omega, depth)
            np.testing.assert_allclose(9.81 * k * np.tanh(k * depth), omega**2, rtol=1e-12)

    def test_spectrum_energy(self):
        # DNV-RP-C205 normalization: Hm0 within 1% of Hs for 1 <= gamma <= 7.
        omega = np.linspace(.05, 30, 300000)
        for gamma in (1., 2., 3.3, 5., 7.):
            spectrum = jonswap(omega, 1.2, 5., gamma)
            self.assertAlmostEqual(4 * np.sqrt(spectrum.sum() * (omega[1] - omega[0])) / 1.2, 1., delta=.01)
        with self.assertRaises(ValueError):
            jonswap(omega, 1., 5., 8.)

    def test_band_truncation_is_reported_not_hidden(self):
        self.assertGreater(band_energy_fraction(.05, .8, 5., 3.3), .97)
        self.assertLess(band_energy_fraction(.15, .8, 12., 3.3), .5)
        config = copy.deepcopy(self.config["waves"])
        config.update(f_min_hz=.15, tp_s=12.)
        with self.assertRaises(ValueError):
            WaveField.create(config, np.random.default_rng(0))

    def test_jittered_components_do_not_repeat(self):
        config = copy.deepcopy(self.config["waves"])
        config["components"] = 128
        period = 1 / ((config["f_max_hz"] - config["f_min_hz"]) / config["components"])
        lag = round(period / .25)
        time = np.arange(0, 3 * period, .25)
        correlations = {}
        for jitter in (False, True):
            config["frequency_jitter"] = jitter
            inc, _ = WaveField.create(config, np.random.default_rng(4)).elevation(time, [-25.])
            correlations[jitter] = np.corrcoef(inc[:-lag, 0], inc[lag:, 0])[0, 1]
        self.assertGreater(correlations[False], .9)
        self.assertLess(abs(correlations[True]), .3)

    def test_long_record_variance_matches_design(self):
        config = copy.deepcopy(self.config["waves"])
        field = WaveField.create(config, np.random.default_rng(8))
        inc, _ = field.elevation(np.arange(0, 7200, .25), [-25.])
        expected = config["hs_m"] * np.sqrt(field.energy_fraction)
        self.assertAlmostEqual(4 * inc.std() / expected, 1., delta=.05)

    def test_welch_separation_of_random_sea(self):
        config = copy.deepcopy(self.config["waves"])
        config.update(reflection=.6, mean_direction_rad=.5)
        field = WaveField.create(config, np.random.default_rng(11))
        dt = .25
        inc, ref = field.elevation(np.arange(0, 600, dt), config["probe_x_m"])
        estimates, residuals = {}, {}
        for incidence in (.5, 0.):
            result = separate_wave_fft(inc + ref, dt, config["probe_x_m"], config["depth_m"], (.1, .4),
                                       segment_samples=round(150 / dt), incidence_rad=incidence)
            self.assertEqual(result["segments"], 7)
            valid = result["valid"]
            estimates[incidence] = np.sqrt(result["reflected_psd"][valid].sum() / result["incident_psd"][valid].sum())
            # Hann leakage mixes neighbouring k, so a random sea never fits exactly.
            residuals[incidence] = np.nanmedian(result["relative_residual"])
        self.assertAlmostEqual(estimates[.5], .6, delta=.02)
        self.assertLess(residuals[.5], .02)
        # A wrongly assumed normal incidence shows up as a larger misfit.
        self.assertGreater(residuals[0.], 2 * residuals[.5])

    def test_grid_elevation_matches_reference(self):
        config = copy.deepcopy(self.config["waves"])
        config.update(mean_direction_rad=.4, spreading_s=6., reflection_phase_rad=.3)
        field = WaveField.create(config, np.random.default_rng(7))
        time = np.arange(0, 40, .25)
        x = np.linspace(-60, 0, 25)
        envelope = np.linspace(1, 1.5, len(time))
        for fast, slow in zip(field.elevation_grid(time, x, envelope), field.elevation(time, x, envelope)):
            np.testing.assert_allclose(fast, slow, atol=1e-10)

    def test_plan_elevation_matches_probe_line_and_mirrors_at_wall(self):
        config = copy.deepcopy(self.config["waves"])
        config.update(mean_direction_rad=.4, spreading_s=6., reflection=1., reflection_phase_rad=0.)
        field = WaveField.create(config, np.random.default_rng(9))
        time = np.arange(0, 20, .5)
        x = np.linspace(-40, 0, 9)
        inc, ref, kept = field.elevation_plane(time, x, [0., 15.])
        self.assertEqual(kept, 1.)
        line_inc, line_ref = field.elevation_grid(time, x)
        np.testing.assert_allclose(inc[:, 0], line_inc, atol=1e-10)
        np.testing.assert_allclose(ref[:, 0], line_ref, atol=1e-10)
        # A perfectly reflecting wall doubles the field at x=0 for every y.
        np.testing.assert_allclose(inc[:, :, -1], ref[:, :, -1], atol=1e-10)
        _, _, kept = field.elevation_plane(time, x, [0.], min_wavelength=10.)
        self.assertTrue(0 < kept < 1)

    def test_directional_spreading(self):
        rng = np.random.default_rng(2)
        wide = sample_directions(5000, .3, 2., rng)
        narrow = sample_directions(5000, .3, 60., rng)
        self.assertTrue(np.all(np.cos(wide) > 0))
        self.assertLess(narrow.std(), wide.std())
        self.assertAlmostEqual(np.mean(narrow), .3, delta=.02)

    def test_empirical_reflection_coefficient(self):
        setting = {"model": "zanuttigh_van_der_meer", "slope_tan": 1 / 1.5, "roughness_factor": .55}
        rock = float(reflection_coefficient(setting, 1., 6.))
        smooth = float(reflection_coefficient({**setting, "roughness_factor": 1.}, 1., 6.))
        mild = float(reflection_coefficient({**setting, "slope_tan": 1 / 4}, 1., 6.))
        self.assertTrue(0 < mild < rock < smooth < 1)
        self.assertEqual(float(reflection_coefficient(.8, 1., 6.)), .8)

    def test_wall_reflection(self):
        config = copy.deepcopy(self.config["waves"])
        config["reflection"] = 1.
        field = WaveField.create(config, np.random.default_rng(3))
        time = np.arange(0, 100, .25)
        inc, ref = field.elevation(time, [0.])
        np.testing.assert_allclose(inc, ref, atol=1e-12)
        config["reflection"] = 0.
        field = WaveField.create(config, np.random.default_rng(3))
        _, ref = field.elevation(time, [-10.])
        self.assertTrue(np.all(ref == 0))

    def test_separation_recovers_known_reflection(self):
        dt, n, frequency = .25, 2400, .2
        time = np.arange(n) * dt
        positions = np.array([-25., -22.3, -18.9])
        k = dispersion(np.array([2*np.pi*frequency]), 15.)[0]
        incident = .3 * np.cos(k*positions[None, :] - 2*np.pi*frequency*time[:, None] + .4)
        reflected = .12 * np.cos(-k*positions[None, :] - 2*np.pi*frequency*time[:, None] + .4)
        result = separate_wave_fft(incident + reflected, dt, positions, 15., (.1, .4))
        idx = np.argmin(abs(result["frequency_hz"] - frequency))
        self.assertAlmostEqual(np.sqrt(result["reflected_psd"][idx] / result["incident_psd"][idx]), .4, places=5)
        self.assertLess(result["relative_residual"][idx], 1e-6)
        # Integrate the Hann main lobe; expected variance = amplitude^2/2.
        energy = np.sum(result["reflected_psd"][idx-1:idx+2]) / (n*dt)
        self.assertAlmostEqual(energy, .12**2 / 2, delta=2e-5)

    def test_degenerate_probes_rejected(self):
        with self.assertRaises(ValueError):
            separate_wave_fft(np.zeros((100, 3)), .25, [-1, -1, -1], 15, (.1, .4))

    def test_unresolvable_bins_are_flagged(self):
        result = separate_wave_fft(np.zeros((400, 3)), .25,
                                   [-1., -1.00000001, -1.00000002], 15., (.1, .4))
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["reflected_psd"]).all())

    def test_force_rao_uses_coherent_incident_and_reflected_phases(self):
        field = WaveField.create(self.config["waves"], np.random.default_rng(5))
        time = np.arange(0, 30, .25)
        envelope = np.ones(len(time))
        inc, ref = field.elevation(time, [-10.])
        rao = np.tile([1., 2., 3.], (len(field.omega), 1))
        force = first_order_force(time, field, rao, rao, -10., envelope)
        np.testing.assert_allclose(force, (inc + ref) * [1., 2., 3.], atol=1e-12)

    def test_coriolis_no_work_and_damping_dissipates(self):
        config = self.config["vessel"]
        mass = np.diag(config["mass_diagonal"])
        velocity = np.array([.3, -.2, .0004])
        c = coriolis(mass, velocity)
        np.testing.assert_allclose(c + c.T, 0.)
        self.assertAlmostEqual(velocity @ c @ velocity, 0., places=7)
        config["hold_kp"] = [0, 0, 0]
        config["hold_kd"] = [0, 0, 0]
        config["wind_load"].update(cx=0, cy=0, cn=0)
        derivative = rhs(np.r_[config["hold_position"], velocity], np.zeros(2), np.zeros(3), config)
        self.assertLess(velocity @ mass @ derivative[3:], 0)

    def test_kaimal_turbulence_variance_and_no_repetition(self):
        from berth_sim.wind import kaimal_fluctuation, kaimal_psd
        dt = .5
        time = np.arange(0, 14400, dt)
        series = [kaimal_fluctuation(time, 8., 1., 80., np.random.default_rng(seed)) for seed in range(10)]
        f = np.fft.rfftfreq(len(time), dt)
        band = (f > 0) & (f <= .45 / dt)
        expected = (kaimal_psd(f[band], 8., 1., 80.) * (f[1] - f[0])).sum()
        self.assertAlmostEqual(np.mean([s.var() for s in series]) / expected, 1., delta=.05)
        # The former 256-harmonic synthesis repeated every ~285 s (|r| > 0.95 at that lag).
        for lag_s in (284.5, 569., 1138.):
            n = round(lag_s / dt)
            self.assertLess(abs(np.corrcoef(series[0][:-n], series[0][n:])[0, 1]), .2)

    def test_relative_wind_zero(self):
        load = wind_load(np.array([5., 0.]), 0., np.array([5., 0., 0.]), self.config["vessel"]["wind_load"])
        np.testing.assert_allclose(load, 0.)

    def test_causal_features(self):
        config = self.config
        config.update(duration_s=180., feature_window_s=60., feature_step_s=60.)
        config["waves"]["welch_segment_s"] = None
        config["vessel"]["enabled"] = False
        raw, truth, _, before = simulate(config)
        cut = round(60 / config["dt_s"])
        raw["probe_observed_m"][cut:] = 999
        raw["wind_global_mps"][cut:] = 999
        after = causal_features(raw["time_s"], raw["wind_global_mps"], raw["probe_observed_m"], truth["reflected_m"], config)
        for key in before[0]:
            np.testing.assert_allclose(before[0][key], after[0][key], equal_nan=True)

    def test_missing_data_is_not_normal_conditions(self):
        config = self.config
        config.update(duration_s=60., feature_window_s=60.)
        config["waves"]["welch_segment_s"] = None
        config["waves"]["missing_probability"] = 1.
        config["vessel"]["enabled"] = False
        _, _, _, rows = simulate(config)
        self.assertEqual(rows[0]["quality_ok"], 0)
        self.assertTrue(np.isnan(rows[0]["hs_reflected_est_m"]))

    def test_reproducibility(self):
        config = self.config
        config.update(duration_s=60., feature_window_s=60.)
        config["waves"]["welch_segment_s"] = None
        config["vessel"]["enabled"] = False
        first = simulate(config)
        second = simulate(config)
        np.testing.assert_array_equal(first[0]["probe_observed_m"], second[0]["probe_observed_m"])

    def test_nyquist_validation(self):
        self.config["dt_s"] = 1.
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_integrator_step_refinement(self):
        config = self.config["vessel"]
        config["initial_state"][3] = .2
        final = []
        for dt in (.5, .25):
            time = np.arange(0, 20 + dt/2, dt)
            final.append(simulate_vessel(time, np.zeros((len(time), 2)), np.zeros((len(time), 3)), config)[-1])
        np.testing.assert_allclose(final[0], final[1], atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
