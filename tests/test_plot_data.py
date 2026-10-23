"""Check that dashboard predictions are linked to the correct saved scenario."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.serve_plots import ROOT, catalog, context


class PlotDataTests(unittest.TestCase):
    def test_catalog_excludes_environment_only_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environmental = root / 'outputs/training_v1/dataset/scenario_000.csv'
            loads = root / 'outputs/force_training_v1/dataset/scenario_000.csv'
            for path in (environmental, loads):
                path.parent.mkdir(parents=True, exist_ok=True)
            environmental.write_text('time_s,wind_u_mps\n0,1\n', encoding='utf-8')
            loads.write_text('time_s,environment_X_rms_N\n0,1\n', encoding='utf-8')
            with patch('scripts.serve_plots.ROOT', root):
                files = catalog()
            self.assertEqual(list(files), ['outputs/force_training_v1/dataset/scenario_000.csv'])

    def test_forecast_quantiles_match_archive(self):
        run = ROOT / 'outputs/force_training_v1'
        if not run.exists():
            self.skipTest('Local training output not present')
        info = context(run / 'dataset/scenario_048.csv', 'gan_lstm')
        f = info['forecast']
        with np.load(run / 'evaluation/gan_lstm_test.npz') as raw:
            idx = np.flatnonzero(raw['scenario'] == 48)[0]
            samples = raw['ensemble'][idx] / 1000
            self.assertEqual(f['origins'][0], raw['origin_s'][idx])
            np.testing.assert_allclose(f['mean'][0], samples.mean(0), rtol=1e-6)
            np.testing.assert_allclose(f['lower'][0], np.quantile(samples, .05, axis=0))
            np.testing.assert_allclose(f['peak_samples'][0], samples.max(1))
        self.assertIsNone(context(run / 'dataset/scenario_000.csv', 'gan_lstm')['forecast'])

    def test_replay_uses_saved_motion(self):
        run = ROOT / 'outputs/loads_demo'
        if not run.exists():
            self.skipTest('Local demo not present')
        info = context(run / 'load_features.csv', 'gan_lstm')
        self.assertEqual(info['scene_mode'], 'replay')
        self.assertIsNone(info['forecast'])
        with np.load(run / 'vessel.npz') as raw:
            np.testing.assert_array_equal(info['motion']['state'], raw['state'])
