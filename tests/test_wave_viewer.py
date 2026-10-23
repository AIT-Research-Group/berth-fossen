import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wave_viewer  # noqa: E402


class WaveViewerTests(unittest.TestCase):
    def setUp(self):
        self.sources = wave_viewer.catalog()
        if "demo" not in self.sources:
            self.skipTest("run the demo simulation into outputs/demo first")

    def test_surface_matches_simulation_and_wall_physics(self):
        query = {"t0": ["600"], "span": ["60"], "x_min": ["-30"], "nx": ["31"], "nt": ["241"]}
        result = wave_viewer.surface("demo", query)
        self.assertEqual(np.shape(result["total"]), (241, 31))
        np.testing.assert_allclose(np.add(result["incident"], result["reflected"]), result["total"], atol=2e-4)
        config = self.sources["demo"]["config"]
        # Probe signals saved by the simulator are the same field plus sensor noise.
        with np.load(ROOT / "outputs/demo/truth.npz") as truth:
            saved = truth["incident_m"][2400:2641, 0] + truth["reflected_m"][2400:2641, 0]
        field = wave_viewer.cached_field("demo")
        from berth_sim.simulation import storm_envelopes
        t = np.array(result["t"])
        inc, ref = field.elevation_grid(t, [config["waves"]["probe_x_m"][0]], storm_envelopes(t, config["storm"])[1])
        np.testing.assert_allclose((inc + ref)[:, 0], saved, atol=1e-9)
        # Full standing wave at the wall: Hm0 = (1 + Kr) Hm0_incident for a constant Kr.
        profile = result["profile"]
        kr = config["waves"]["reflection"]
        self.assertAlmostEqual(profile["hm0_total"][-1], (1 + kr) * profile["hm0_incident"], delta=1e-3)

    def test_spectrogram_reproduces_training_features(self):
        result = wave_viewer.spectrogram("demo")
        # Same windows and Hm0 estimates as the saved features.csv of this run.
        import csv
        with (ROOT / "outputs/demo/features.csv").open(encoding="utf-8") as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual(len(result["time_min"]), len(saved))
        np.testing.assert_allclose(result["hm0_estimate"], [float(r["hs_reflected_est_m"]) for r in saved], rtol=1e-5)
        # Integrating the reflected spectrum gives back the reported Hm0.
        df = result["frequency_hz"][1] - result["frequency_hz"][0]
        hm0 = 4 * np.sqrt(np.nansum(np.array(result["reflected_psd"], dtype=float)[0]) * df)
        self.assertAlmostEqual(hm0, result["hm0_estimate"][0], delta=1e-3)

    def test_vessel_and_forecast_archives(self):
        if "loads_demo" in wave_viewer.vessel_runs():
            motion = wave_viewer.vessel("loads_demo", {"points": ["500"]})
            self.assertLessEqual(len(motion["time_s"]), 500)
            self.assertEqual(len(motion["wind"][0]), 3)
        runs = wave_viewer.forecast_runs()
        if "training_v1" not in runs:
            self.skipTest("no archived forecasts")
        run = runs["training_v1"]
        result = wave_viewer.forecast({"run": ["training_v1"], "model": ["gan_lstm"],
                                       "scenario": [str(run["scenarios"][0])], "variable": ["3"]})
        self.assertEqual(np.shape(result["mean"]), (len(result["origin_min"]), 30))
        self.assertEqual(np.shape(result["history"])[1], 60)
        self.assertTrue(np.all(np.array(result["q05"]) <= np.array(result["q95"])))
        # History ends right before the first target minute of the same issue time.
        self.assertEqual(len(result["target"]), len(result["history"]))
        with self.assertRaises(KeyError):
            wave_viewer.forecast({"run": ["../x"], "model": ["gan_lstm"]})

    def test_seastate_ship_scale_and_profiles(self):
        result = wave_viewer.seastate("demo", {"frames": ["4"], "n": ["60"]})
        ship = result["ship"]
        # Hull proportions come from the MSS tanker (B/L = 46/246, T/L = 10/246).
        self.assertAlmostEqual(ship["beam_m"] / ship["length_m"], 46 / 246, places=6)
        self.assertEqual(np.shape(result["frames"]), (4, 60, 60))
        self.assertEqual(np.shape(result["transverse"])[0], 4)
        # lambda = L lies at a lower frequency than lambda = B, which lies below the peak here.
        self.assertLess(result["f_ship_length_hz"], result["f_ship_beam_hz"])
        self.assertTrue(0 < result["kept_variance_fraction"] <= 1)
        # Normal incidence: crests run along the quay, so the keel-line profile is uniform.
        config = self.sources["demo"]["config"]
        if config["waves"].get("spreading_s") is None and config["waves"].get("mean_direction_rad", 0) == 0:
            self.assertLess(np.ptp(result["centerline"][0]), 1e-2)

    def test_rejects_unknown_source_and_oversized_grid(self):
        with self.assertRaises(KeyError):
            wave_viewer.surface("../secret", {})
        with self.assertRaises(ValueError):
            wave_viewer.surface("demo", {"nx": ["801"], "nt": ["4001"]})


if __name__ == "__main__":
    unittest.main()
