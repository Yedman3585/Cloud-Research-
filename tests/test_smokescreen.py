import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fogids.smokescreen import DROP, POLICIES, Engine, Option, fluid_bound, generate  # noqa: E402


def tiny_config(seed=3):
    c = json.loads((ROOT / 'configs/ids-place-smokescreen.json').read_text(encoding='utf-8'))
    c['seed'] = seed
    c['horizon_s'] = 6.0
    c['topology']['fog_nodes'] = 1
    c['topology']['gateways_per_fog'] = 2
    c['risk']['critical_gateways'] = [0]
    c['attacks'] = [
        {'class': 'ddos', 'start_s': 1.0, 'duration_s': 4.0, 'ramp_s': 0.0, 'gateways': [0], 'flows_per_s': 3000.0},
        {'class': 'web', 'start_s': 0.0, 'duration_s': 6.0, 'ramp_s': 0.0, 'gateways': [0, 1], 'flows_per_s': 15.0}]
    return c


class SmokescreenTests(unittest.TestCase):
    def test_generator_is_deterministic_and_splits_streams(self):
        a, b = generate(tiny_config()), generate(tiny_config())
        self.assertEqual([t.__dict__ for t in a.tasks], [t.__dict__ for t in b.tasks])
        flood = [t for t in a.tasks if t.stream == 'flood']
        self.assertTrue(flood and all(t.packets >= 50 * t.n_flows for t in flood))
        self.assertEqual(sorted(a.class_counts()), ['ddos', 'web'])

    def test_option_costs_and_detection(self):
        sc = generate(tiny_config())
        t = next(t for t in sc.tasks if t.stream == 'flood')
        full = sc.stages(t, Option(t.gateway, 'light', 1.0, 'gw'))
        tenth = sc.stages(t, Option(t.gateway, 'light', 0.1, 'gw'))
        self.assertLess(tenth[0][2], 0.2 * full[0][2])  # extraction scales with f (plus hashing)
        raw = sc.stages(t, Option('fog0', 'light', 0.1, 'node'))
        self.assertEqual([k for k, _, _ in raw], ['cpu', 'link', 'cpu'])
        self.assertEqual(sc.detection(t, DROP), {'ddos': 0.0})
        self.assertGreater(sc.detection(t, Option(t.gateway, 'light', 0.01, 'gw'))['ddos'], 0.99)

    def test_engine_matches_processor_sharing(self):
        from fogids.ps_model import ps_finish_times
        sc = generate(tiny_config())
        t = sc.tasks[0]
        e = Engine(sc)
        opt = Option(t.gateway, 'light', 1.0, 'gw')
        stages = sc.stages(t, opt)
        e._start(t, list(stages), 0.0)
        e._advance(float('inf'))
        expected = 0.0
        for _, node, work in stages:
            expected = ps_finish_times(sc.node(node).mips, expected, {}, [(expected, 'x', work)])['x']
        self.assertAlmostEqual(e.finish[t.id], expected, places=6)

    def test_bound_is_below_every_policy(self):
        sc = generate(tiny_config())
        bound = fluid_bound(sc)
        for name, cls in POLICIES.items():
            r = Engine(sc).run(cls())
            self.assertLessEqual(bound['objective'], r['objective'] + 1e-9, name)

    def test_processor_sharing_bound_is_below_every_policy(self):
        from fogids.smokescreen_bound import ps_window_bound
        sc = generate(tiny_config())
        bound = ps_window_bound(sc, time_limit_s=5.0)
        self.assertEqual(bound['not_optimal'], 0)
        for name, cls in POLICIES.items():
            r = Engine(sc).run(cls())
            self.assertLessEqual(bound['objective'], r['objective'] + 1e-9, name)


if __name__ == '__main__':
    unittest.main()
