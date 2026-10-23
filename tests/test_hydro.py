import copy
import json
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference/mss_tanker"
try:
    from berth_sim.hydrodynamics import TankerHydrodynamics
except ImportError:
    raise unittest.SkipTest("scipy (hydro extra) required")
from berth_sim.waves import WaveField
from berth_sim.vessel import first_order_force


@unittest.skipUnless((REFERENCE / "tanker.mat").exists(), "optional MSS reference data not present")
class HydroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hydro = TankerHydrodynamics(REFERENCE)
        cls.omega = np.array([.5, .8, 1.2])

    def test_provenance(self):
        info = self.hydro.provenance()
        self.assertEqual(info["length_m"], 246.0)
        self.assertEqual(len(info["sha256"]["tanker.mat"]), 64)

    def test_port_starboard_mirror_symmetry(self):
        # Mirroring y flips the signs of sway force and yaw moment only.
        port = self.hydro.force_rao(self.omega, np.pi / 2)
        starboard = self.hydro.force_rao(self.omega, -np.pi / 2)
        np.testing.assert_allclose(port * [1, -1, -1], starboard, rtol=1e-9, atol=1e-6)

    def test_following_and_head_seas_have_no_sway(self):
        for direction in (0., np.pi):
            rao = self.hydro.force_rao(self.omega, direction)
            self.assertLess(np.abs(rao[:, 1:]).max(), 1e-6 * np.abs(rao[:, 0]).min() + 1e3)

    def test_long_wave_sway_force_phase(self):
        # Froude-Krylov limit: F ~ -i k rho g V eta for exp(-i omega t), i.e. -90 degrees.
        rao = self.hydro.force_rao(np.array([.4]), np.pi / 2)
        self.assertAlmostEqual(np.degrees(np.angle(rao[0, 1])), -90., delta=15.)

    def test_berth_reflection_uses_opposite_direction(self):
        config = json.loads((ROOT / "configs/demo.json").read_text())["waves"]
        config.update(reflection=1., reflection_phase_rad=0., mean_direction_rad=.3)
        field = WaveField.create(config, np.random.default_rng(1))
        heading = np.pi / 2
        incident, reflected = self.hydro.berth_force_raos(field, heading)
        np.testing.assert_allclose(incident, self.hydro.force_rao(field.omega, .3 - heading))
        np.testing.assert_allclose(reflected, self.hydro.force_rao(field.omega, np.pi - .3 - heading))
        # A full standing wave doubles, or cancels, the incident-only load at x=0.
        time = np.arange(0, 600, .25)
        both = first_order_force(time, field, incident, reflected, 0., np.ones(len(time)))
        alone = first_order_force(time, field, incident, np.zeros_like(reflected), 0., np.ones(len(time)))
        self.assertFalse(np.allclose(both, alone, rtol=.05))


if __name__ == "__main__":
    unittest.main()
