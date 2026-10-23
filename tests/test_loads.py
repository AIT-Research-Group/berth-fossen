import unittest
import numpy as np
from berth_sim.loads import load_features


class LoadTests(unittest.TestCase):
    def test_statistics_and_cancellation(self):
        wind = np.tile([2., -3., 4.], (8, 1))
        rows = load_features(np.arange(8), wind, -wind, 4, 4)
        self.assertEqual(rows[0]['wind_Y_mean_N'], -3.)
        self.assertEqual(rows[0]['wind_Y_rms_N'], 3.)
        self.assertEqual(rows[0]['environment_N_absmax_Nm'], 0.)
        self.assertEqual(rows[0]['time_s'], 3.)

    def test_future_does_not_change_previous_window(self):
        wind = np.ones((8, 3))
        before = load_features(np.arange(8), wind, wind, 4, 4)[0]
        wind[4:] = 100
        after = load_features(np.arange(8), wind, wind, 4, 4)[0]
        self.assertEqual(before, after)

    def test_rms_of_sinusoid(self):
        t = np.arange(1000)/1000
        wave = np.tile((2*np.sin(2*np.pi*10*t))[:, None], (1, 3))
        row = load_features(t, np.zeros_like(wave), wave, 1000, 1000)[0]
        self.assertAlmostEqual(row['wave_Y_rms_N'], np.sqrt(2))
        self.assertAlmostEqual(row['wave_Y_mean_N'], 0.)
