"""Fluid lower bound and estimate-fidelity helpers (M5)."""
import unittest

from fogids.oracle import fidelity, spearman
from fogids.problem import Instance, Model, Task
from test_problem import instance

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def task(i, release, flows, deadline=1.0):
    return Task(id=i, gateway='gw0', release_s=release, deadline_s=deadline, n_flows=flows,
                attack_flows=0, label='benign', input_bytes=500, prefilter_score=0.0, criticality=1.0)


def models(full_fpr=0.1):
    return [Model('light', 10, 0.1, 10, 0.0, recall={'ddos': 0.9}, false_positive_rate=0.3),
            Model('full', 10, 1.0, 10, 0.0, recall={'ddos': 0.99}, false_positive_rate=full_fpr)]


def with_models(p):
    return Instance(name='t', seed=0, horizon_s=10.0, window_s=1.0, epoch_s=0.1, nodes=p.nodes,
                    models=models(), tasks=p.tasks).validate()


@unittest.skipUnless(HAVE_SCIPY, 'scipy not installed')
class FluidBoundTests(unittest.TestCase):
    def test_uncongested_instance_runs_everything_with_the_best_model(self):
        from fogids.bounds import fluid_bound
        p = with_models(instance([task(0, 1.0, 100)]))
        b = fluid_bound(p)
        self.assertEqual(b['deadline_miss_rate'], 0.0)
        self.assertAlmostEqual(b['false_alert_rate'], 0.1)
        self.assertEqual(b['plan'][0][1], 'full')

    def test_burst_forces_the_cheaper_model_or_misses(self):
        from fogids.bounds import fluid_bound
        # Ten windows released together with a 0.12 s deadline: full (110 MI) fits alone
        # on the gateway (1000 MIPS) but not ten times; transfers to fog are too slow.
        p = with_models(instance([task(i, 1.0, 100, deadline=0.12) for i in range(10)]))
        b = fluid_bound(p)
        self.assertGreater(b['false_alert_rate'], 0.1 + 1e-6)

    def test_bound_is_below_every_simple_assignment(self):
        from fogids.bounds import fluid_bound
        p = with_models(instance([task(i, 1.0 + 0.05 * i, 200) for i in range(6)]))
        self.assertLessEqual(fluid_bound(p)['false_alert_rate'], 0.3 + 1e-9)


class FidelityTests(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(spearman([1, 2], [1, 2]))

    def test_broken_promises(self):
        p = with_models(instance([task(i, 1.0, 100) for i in range(4)]))
        predicted = {0: 1.5, 1: 1.6, 2: 2.5, 3: 1.4}
        end = {'results': [{'id': i, 'finish_s': f} for i, f in enumerate([1.5, 2.3, 2.6, 1.4])]}
        f = fidelity(p, predicted, end)
        self.assertAlmostEqual(f['broken_promises'], 1 / 3)  # task 1 promised 0.6 s, took 1.3 s
        self.assertAlmostEqual(f['deadline_agreement'], 0.75)


if __name__ == '__main__':
    unittest.main()
