import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fogids.generator import Stream, _burst_overlap, generate, load_config
from fogids.problem import BENIGN, Instance

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / 'configs/ids-place-small.json'
# Fingerprint of configs/ids-place-small.json. A mismatch on another machine means
# the generator is not platform-reproducible; update only after a deliberate change.
SMALL_SHA256 = '7e3568e77fa2897ee397857d2b5b646fef740c74251d1986b5e273e29da54a2e'


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(SMALL)

    def test_same_seed_same_instance_and_platform_fingerprint(self):
        first, second = generate(self.c).to_json(), generate(copy.deepcopy(self.c)).to_json()
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256(first.encode()).hexdigest(), SMALL_SHA256)

    def test_different_seed_changes_traffic(self):
        other = copy.deepcopy(self.c)
        other['seed'] += 1
        self.assertNotEqual(generate(self.c).to_json(), generate(other).to_json())

    def test_json_round_trip(self):
        instance = generate(self.c)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'instance.json'
            instance.save(path)
            loaded = Instance.load(path)
        self.assertEqual(loaded.to_json(), instance.to_json())
        self.assertEqual(loaded.tasks, instance.tasks)

    def test_topology_shape(self):
        instance = generate(self.c)
        topo = self.c['topology']
        self.assertEqual(len(instance.tier('cloud')), 1)
        self.assertEqual(len(instance.tier('fog')), topo['fog_nodes'])
        self.assertEqual(len(instance.tier('edge')), topo['fog_nodes'] * topo['gateways_per_fog'])
        critical = [g for g in instance.tier('edge') if g.criticality > 1]
        self.assertEqual(len(critical), 2)  # ceil(0.25 * 8)

    def test_tasks_respect_chunking_and_flow_accounting(self):
        instance = generate(self.c)
        traffic = self.c['traffic']
        for t in instance.tasks:
            self.assertLessEqual(t.n_flows, traffic['max_flows_per_task'])
            self.assertEqual(t.input_bytes, traffic['header_bytes'] + traffic['bytes_per_flow'] * t.n_flows)
            self.assertEqual(t.label == BENIGN, t.attack_flows == 0)
            self.assertIn(t.deadline_s, (self.c['deadlines']['high_s'], self.c['deadlines']['low_s']))

    def test_attack_burst_raises_load_only_on_target_gateways(self):
        c = copy.deepcopy(self.c)
        c['attacks'] = [{'class': 'ddos', 'start_s': 10.0, 'duration_s': 10.0, 'ramp_s': 0.0,
                         'gateways': [0], 'flows_per_s': 3000}]
        instance = generate(c)

        def flows(gateway, lo, hi):
            return sum(t.n_flows for t in instance.tasks if t.gateway == gateway and lo <= t.release_s < hi)
        self.assertGreater(flows('gw0', 12, 20), 5 * flows('gw0', 1, 9))
        self.assertLess(flows('gw1', 12, 20), 2 * flows('gw1', 1, 9))
        attacked = [t for t in instance.tasks if t.attack_flows]
        self.assertTrue(attacked and all(t.gateway == 'gw0' for t in attacked))
        self.assertGreater(sum(t.deadline_s == 0.3 for t in attacked), 0.8 * len(attacked))

    def test_no_attacks_means_all_benign(self):
        c = copy.deepcopy(self.c)
        c['attacks'] = []
        self.assertTrue(all(t.label == BENIGN for t in generate(c).tasks))

    def test_burst_ramp_integral(self):
        burst = {'start_s': 10.0, 'duration_s': 4.0, 'ramp_s': 2.0}
        self.assertAlmostEqual(_burst_overlap(burst, 0, 100), 3.0)  # 4 s minus half the ramp
        self.assertAlmostEqual(_burst_overlap(burst, 10, 11), 0.25)
        self.assertAlmostEqual(_burst_overlap(burst, 12, 13), 1.0)
        self.assertEqual(_burst_overlap(burst, 20, 21), 0.0)

    def test_poisson_mean(self):
        stream = Stream(5)
        for mean in (3.0, 200.0):
            samples = [stream.poisson(mean) for _ in range(4000)]
            self.assertAlmostEqual(sum(samples) / len(samples), mean, delta=0.05 * mean)

    def test_shipped_configs_generate_valid_instances(self):
        for name in ('ids-place-small', 'ids-place-medium'):
            summary = generate(load_config(ROOT / f'configs/{name}.json')).summary()
            self.assertGreater(summary['tasks'], 0)
            self.assertGreater(summary['attack_flows'], 0)


if __name__ == '__main__':
    unittest.main()
