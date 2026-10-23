import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install training extras to test forecasting models")

from berth_sim.training.data import cut_windows
from berth_sim.training.model import Generator, fit_scaler, physical_to_latent, PhysicalTransform, energy_score
from berth_sim.training.evaluation import crps, event_labels, fit_calibration, apply_calibration
from berth_sim.training.train import load_model, predict


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.config = {"hidden_size": 8, "noise_size": 3, "horizon_steps": 5}
        rng = np.random.default_rng(3)
        self.physical = rng.uniform(1, 3, (30, 4)).astype(np.float32)
        self.physical[:, 2] = np.linalg.norm(self.physical[:, :2], axis=-1) + .5
        self.scaler = fit_scaler(self.physical)

    def test_physical_transform_round_trip(self):
        transform = PhysicalTransform(self.scaler)
        x = torch.from_numpy(self.physical)
        torch.testing.assert_close(transform.decode(transform.encode(x)), x)

    def test_generator_shapes_constraints_and_gradients(self):
        model = Generator(self.config, self.scaler)
        x = torch.from_numpy(self.physical[:12].reshape(2, 6, 4))
        ensemble = model(x, samples=4)
        self.assertEqual(ensemble.shape, (2, 4, 5, 4))
        self.assertTrue((ensemble[..., 3] > 0).all())
        self.assertTrue((ensemble[..., 2] >= torch.linalg.vector_norm(ensemble[..., :2], dim=-1)).all())
        self.assertGreater(float(ensemble.var(dim=1).sum().detach()), 0)
        target = x[:, :5]
        loss = energy_score(model.transform.score_space(ensemble), model.transform.score_space(target))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)

    def test_checkpoint_prediction_reproduces(self):
        model = Generator(self.config, self.scaler)
        history = self.physical[:6][None]
        before = predict(model, history, samples=4, seed=55)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            torch.save({"model_state": model.state_dict(), "config": self.config,
                        "scaler": self.scaler, "model_name": "gan_lstm"}, path)
            loaded, _ = load_model(path)
            after = predict(loaded, history, samples=4, seed=55)
        np.testing.assert_array_equal(before, after)

    def test_crps_matches_pairwise_definition(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(3, 10, 5, 4))
        targets = rng.normal(size=(3, 5, 4))
        expected = abs(samples-targets[:, None]).mean(axis=1) - .5*abs(samples[:, :, None]-samples[:, None, :]).mean(axis=(1, 2))
        np.testing.assert_allclose(crps(samples, targets), expected)

    def test_simultaneous_event_is_not_two_separate_exceedances(self):
        x = np.zeros((1, 2, 4))
        x[0, 0, 2] = 20
        x[0, 1, 3] = 2
        labels = event_labels(x, 2, {"gust_mps": 13, "reflected_hs_m": .7})
        self.assertTrue(labels["wind"][0] and labels["wave"][0] and labels["any"][0])
        self.assertFalse(labels["simultaneous"][0])

    def test_invalid_rows_break_windows(self):
        values = np.ones((12, 4))
        quality = np.ones(12, dtype=bool)
        quality[5] = False
        x, y, origins = cut_windows(values, np.arange(12)*60, quality, 3, 2, 1)
        self.assertEqual(origins.tolist(), [120., 480., 540.])

    def test_one_class_calibration_is_explicit(self):
        fit = fit_calibration(np.zeros(20), np.zeros(20), 100)
        self.assertEqual(fit["kind"], "constant")
        calibrated = apply_calibration(np.array([0, .5, 1]), fit)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated > 0) & (calibrated < 1)).all())


if __name__ == "__main__":
    unittest.main()
