import importlib.util
import json
from pathlib import Path
import unittest

from fogids.generator import generate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('calibrate', ROOT / 'scripts/calibrate.py')
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class CalibrationTests(unittest.TestCase):
    def test_label_grouping(self):
        cases = {'BenignTraffic': 'benign', 'DDoS-ICMP_Flood': 'ddos', 'DoS-SYN_Flood': 'dos',
                 'Mirai-greeth_flood': 'mirai', 'Recon-PortScan': 'recon', 'VulnerabilityScan': 'recon',
                 'MITM-ArpSpoofing': 'spoofing', 'DNS_Spoofing': 'spoofing', 'XSS': 'web',
                 'Uploading_Attack': 'web', 'DictionaryBruteForce': 'bruteforce'}
        for label, group in cases.items():
            self.assertEqual(calibrate.label_group(label), group)
        with self.assertRaises(ValueError):
            calibrate.label_group('Unknown')

    def test_linear_fit(self):
        xs = [1, 10, 100, 1000]
        a, b, r2 = calibrate.fit_linear(xs, [5e-5 + 2e-6 * x for x in xs])
        self.assertAlmostEqual(a, 5e-5, places=12)
        self.assertAlmostEqual(b, 2e-6, places=12)
        self.assertAlmostEqual(r2, 1.0)
        a, b, _ = calibrate.fit_linear([1, 2, 3], [0.5, 2.0, 3.5])  # negative intercept: through origin
        self.assertEqual(a, 0.0)
        self.assertGreater(b, 0)

    def test_calibrated_config_generates_valid_instance(self):
        base = json.loads((ROOT / 'configs/ids-place-small.json').read_text(encoding='utf-8'))
        recall = {c: 0.9 for c in calibrate.ATTACKS}
        profile = {'profile': 'test', 'ref_mips': 8000,
                   'models': [{'name': n, 'fixed_mi': 0.5, 'mi_per_flow': 0.01, 'memory_mb': 20,
                               'load_s': 0.05, 'recall': recall} for n in ('light', 'full')]}
        config = calibrate.calibrated_config(base, profile, n_features=46)
        self.assertTrue(config['calibrated'])
        self.assertEqual(config['traffic']['bytes_per_flow'], 184)
        self.assertFalse(base.get('calibrated'))  # base config untouched
        instance = generate(config)
        self.assertTrue(instance.meta['calibrated'])
        self.assertEqual(instance.model('full').mi_per_flow, 0.01)


if __name__ == '__main__':
    unittest.main()
