import unittest
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Training extras required")
from berth_sim.training.model import Generator, Discriminator, energy_score, fit_scaler
from berth_sim.training.evaluation import conformal_offsets


def features(rng, n):
    wind = rng.normal(5, 2, (n, 2))
    gust = np.linalg.norm(wind, axis=1) + rng.uniform(.5, 3, n)
    hs = rng.uniform(.2, 1.2, n)
    return np.column_stack((wind, gust, hs)).astype(np.float32)


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.scaler = fit_scaler(features(self.rng, 200))
        self.config = {"horizon_steps": 6, "noise_size": 3, "hidden_size": 8, "local_noise_size": 2, "dropout": .2,
                       "critic": "multiscale", "spectral_norm": True}

    def test_generator_local_noise_and_constraints(self):
        model = Generator(self.config, self.scaler).eval()
        history = torch.from_numpy(features(self.rng, 5 * 12).reshape(5, 12, 4))
        samples = model(history, 4)
        self.assertEqual(tuple(samples.shape), (5, 4, 6, 4))
        # Physical constraints survive the upgrades: gust >= |mean wind|, Hm0 > 0.
        self.assertTrue((samples[..., 2] >= torch.linalg.vector_norm(samples[..., :2], dim=-1) - 1e-5).all())
        self.assertTrue((samples[..., 3] > 0).all())
        self.assertGreater(samples.std(dim=1).mean().item(), 0)

    def test_multiscale_spectral_norm_critic_trains(self):
        critic = Discriminator(self.config, self.scaler)
        history = torch.from_numpy(features(self.rng, 3 * 12).reshape(3, 12, 4))
        future = torch.from_numpy(features(self.rng, 3 * 6).reshape(3, 6, 4))
        score = critic(history, future)
        self.assertEqual(tuple(score.shape), (3,))
        score.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None))

    def test_fair_energy_score_prefers_true_spread(self):
        y = torch.randn(20000, 1, 1)
        ensemble = torch.randn(20000, 8, 1, 1)
        scores = {s: energy_score(ensemble * s, y, fair=True).item() for s in (.7, 1., 1.3)}
        self.assertLess(scores[1.], scores[.7])
        self.assertLess(scores[1.], scores[1.3])
        plain = {s: energy_score(ensemble * s, y).item() for s in (.8, 1.)}
        self.assertLess(plain[.8], plain[1.])  # the plain estimator rewards under-dispersion

    def test_conformal_restores_coverage_of_narrow_ensembles(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(4000, 5, 2))
        narrow = rng.normal(size=(4000, 50, 5, 2)) * .5  # half the true spread
        calibration, test = slice(0, 2000), slice(2000, 4000)
        offsets = conformal_offsets(narrow[calibration], truth[calibration])
        low, high = np.quantile(narrow[test], [.05, .95], axis=1)
        raw = ((truth[test] >= low) & (truth[test] <= high)).mean()
        fixed = ((truth[test] >= low - offsets) & (truth[test] <= high + offsets)).mean()
        self.assertLess(raw, .7)
        self.assertAlmostEqual(fixed, .9, delta=.02)


if __name__ == "__main__":
    unittest.main()
