"""Aggregation across seeds and one end-to-end sweep job."""
import math
import statistics
import unittest

from fogids.bridge import compile_driver
from fogids.sweep import METRICS, aggregate, run_one, t95
from test_problem import ROOT_CONFIG


def row(policy, seed, miss):
    r = {key: 0.0 for key, _ in METRICS}
    r.update(config='c', load_scale=1, seed=seed, policy=policy, miss_all=miss)
    return r


class SweepTests(unittest.TestCase):
    def test_mean_and_t_interval(self):
        values = [0.1, 0.2, 0.4]
        summary = aggregate([row('a', s, v) for s, v in enumerate(values)])
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]['miss_all'], statistics.fmean(values))
        expected = 4.303 * statistics.stdev(values) / math.sqrt(3)
        self.assertAlmostEqual(summary[0]['miss_all_ci'], expected)

    def test_t_quantiles(self):
        self.assertAlmostEqual(t95(9), 2.262)
        self.assertAlmostEqual(t95(22), 2.086)  # falls back to the nearest smaller tabulated df
        self.assertAlmostEqual(t95(100), 1.96)

    def test_single_job_end_to_end(self):
        compile_driver()
        r = run_one((str(ROOT_CONFIG), 3, 'risk-split', 'none'))
        self.assertEqual((r['seed'], r['policy']), (3, 'risk-split'))
        self.assertEqual(r['unfinished'], 0)
        self.assertTrue(0 <= r['false_alert_rate'] <= 1)


if __name__ == '__main__':
    unittest.main()
