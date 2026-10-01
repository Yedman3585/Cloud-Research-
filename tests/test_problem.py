from pathlib import Path
import unittest

from fogids.problem import Instance, Model, Node, Task

ROOT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/ids-place-small.json'


def node(name, tier, parent, mips=1000, bw=1000.0, lat=0.01, resident=('light',)):
    return Node(name=name, tier=tier, parent=parent, mips=mips, memory_mb=1024,
                uplink_bytes_s=None if parent is None else bw,
                uplink_latency_s=None if parent is None else lat,
                idle_w=1.0, busy_w=2.0, cost_per_gi=0.0, resident=resident)


def instance(tasks=None, nodes=None):
    nodes = nodes or [node('cloud', 'cloud', None, mips=8000, resident=('light', 'full')),
                      node('fog0', 'fog', 'cloud', mips=4000, bw=2000.0, lat=0.05),
                      node('gw0', 'edge', 'fog0'), node('gw1', 'edge', 'fog0')]
    models = [Model('light', fixed_mi=10, mi_per_flow=1, memory_mb=100, load_s=0.5, recall={'ddos': 0.9}),
              Model('full', fixed_mi=100, mi_per_flow=4, memory_mb=300, load_s=2.0, recall={'ddos': 0.99})]
    tasks = tasks if tasks is not None else [task(0, 'gw0', 1.0)]
    return Instance(name='t', seed=0, horizon_s=10.0, window_s=1.0, epoch_s=0.1,
                    nodes=nodes, models=models, tasks=tasks)


def task(i, gateway, release, flows=100, attack=0):
    return Task(id=i, gateway=gateway, release_s=release, deadline_s=1.0, n_flows=flows,
                attack_flows=attack, label='ddos' if attack else 'benign', input_bytes=500,
                prefilter_score=0.2, criticality=1.0)


class ProblemTests(unittest.TestCase):
    def test_eligibility_is_gateway_and_ancestors(self):
        p = instance().validate()
        self.assertEqual(p.eligible(p.tasks[0]), ['gw0', 'fog0', 'cloud'])

    def test_transfer_time_sums_traversed_uplinks(self):
        p = instance().validate()
        t = p.tasks[0]
        self.assertEqual(p.transfer_s(t, 'gw0'), 0)
        self.assertAlmostEqual(p.transfer_s(t, 'fog0'), 0.01 + 500 / 1000)
        self.assertAlmostEqual(p.transfer_s(t, 'cloud'), 0.01 + 0.5 + 0.05 + 500 / 2000)
        with self.assertRaises(ValueError):
            p.path_links('gw0', 'gw1')

    def test_compute_time_includes_cold_start_only_when_not_resident(self):
        p = instance().validate()
        t = p.tasks[0]
        self.assertAlmostEqual(p.compute_s(t, 'gw0', 'light'), 110 / 1000)
        self.assertAlmostEqual(p.compute_s(t, 'gw0', 'full'), 2.0 + 500 / 1000)
        self.assertAlmostEqual(p.compute_s(t, 'cloud', 'full'), 500 / 8000)

    def test_weight_grows_with_score_and_criticality(self):
        t = task(0, 'gw0', 1.0)
        self.assertAlmostEqual(t.weight, 0.3)

    def test_validation_rejects_bad_instances(self):
        bad_parent = [node('cloud', 'cloud', None), node('gw0', 'edge', 'missing')]
        two_roots = [node('cloud', 'cloud', None), node('fog0', 'fog', None)]
        inverted = [node('cloud', 'cloud', None), node('fog0', 'fog', 'gw0'), node('gw0', 'edge', 'cloud')]
        for nodes in (bad_parent, two_roots, inverted):
            with self.assertRaises(ValueError):
                instance(tasks=[], nodes=nodes).validate()
        with self.assertRaises(ValueError):  # task on a non-edge node
            instance(tasks=[task(0, 'fog0', 1.0)]).validate()
        with self.assertRaises(ValueError):  # releases out of order
            instance(tasks=[task(0, 'gw0', 2.0), task(1, 'gw0', 1.0)]).validate()
        with self.assertRaises(ValueError):  # attack class without recall
            instance(tasks=[Task(**{**task(0, 'gw0', 1.0, attack=5).__dict__, 'label': 'web'})]).validate()
        heavy = [node('cloud', 'cloud', None, resident=('light', 'full'))]
        heavy[0] = Node(**{**heavy[0].__dict__, 'memory_mb': 200})
        with self.assertRaises(ValueError):  # resident models exceed memory
            instance(tasks=[], nodes=heavy).validate()


if __name__ == '__main__':
    unittest.main()
