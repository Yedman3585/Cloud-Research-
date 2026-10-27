import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fogids.families import FAMILIES  # noqa: E402
from fogids.smokescreen import Engine, POLICIES, generate  # noqa: E402

X3 = json.loads((ROOT / 'configs/ids-place-smokescreen-x3.json').read_text(encoding='utf-8'))


def config(seed, family=None, horizon=8.0):
    c = copy.deepcopy(X3)
    c['seed'], c['horizon_s'] = seed, horizon
    if family:
        FAMILIES[family](c)
    return c


class FamiliesTest(unittest.TestCase):
    def test_every_family_generates_and_runs(self):
        for name in FAMILIES:
            sc = generate(config(3, name))
            self.assertTrue(sc.tasks, name)
            r = Engine(sc).run(POLICIES['greedy-v2']())
            self.assertTrue(0.0 <= r['objective'] <= 2.0, name)

    def test_families_depend_on_the_seed(self):
        for name in ('f2-botnet', 'f3-evasive', 'f4-surges'):
            a, b = config(3, name, 60.0), config(4, name, 60.0)
            self.assertNotEqual(json.dumps(a['attacks']) + json.dumps(a.get('surges')),
                                json.dumps(b['attacks']) + json.dumps(b.get('surges')), name)

    def test_profiles_change_detector_rates(self):
        sc = generate(config(3, 'f1-detectors'))
        t = sc.tasks[0]
        p = sc.device_profile(t.gateway, t.release_s)
        factor = sc.profiles[0][p]['fpr']
        self.assertAlmostEqual(sc.fpr(t, 'light'), min(0.95, sc.model('light').false_positive_rate * factor))

    def test_without_family_rates_are_the_models(self):
        sc = generate(config(3))
        t = sc.tasks[0]
        self.assertIsNone(sc.device_profile(t.gateway, 0.0))
        self.assertEqual(sc.fpr(t, 'full'), sc.model('full').false_positive_rate)


if __name__ == '__main__':
    unittest.main()
