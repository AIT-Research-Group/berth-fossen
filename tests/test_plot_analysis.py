import unittest
import numpy as np
from scripts.plot_analysis import confusion, events, analysis


class AnalysisTests(unittest.TestCase):
    def test_confusion_orientation_and_undefined_metrics(self):
        m = confusion([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual([m[k] for k in ('tn', 'fp', 'fn', 'tp')], [1, 1, 1, 1])
        self.assertEqual(m['recall'], .5)
        self.assertIsNone(confusion([0, 0], [0, 0])['precision'])
        self.assertIsNone(confusion([0, 0], [0, 0])['recall'])

    def test_joint_event_is_union_per_trajectory_and_strict_threshold(self):
        samples = np.zeros((1, 2, 2, 6))
        samples[0, 0, 0, :2] = 2000  # same sample exceeds two axes
        samples[0, 1, 1, 2] = 1000  # equality is not an exceedance
        target = np.zeros((1, 2, 6)); target[0, 1, 2] = 2000
        truth, p = events(samples, target, np.ones(3), 2)
        np.testing.assert_array_equal(p, [[.5, .5, 0, .5]])
        np.testing.assert_array_equal(truth, [[0, 0, 1, 1]])
        self.assertFalse(events(samples, target, np.ones(3), 1)[0].any())

    def test_invalid_options(self):
        for query in ({'x': ['nan']}, {'p': ['0']}, {'h': ['29']}, {'model': ['unknown']}):
            with self.assertRaises(ValueError):
                analysis(query)
