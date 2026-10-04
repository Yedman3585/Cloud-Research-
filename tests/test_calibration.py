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


try:
    import numpy as np
except ImportError:  # optional calibration dependency
    np = None


@unittest.skipIf(np is None, 'numpy not installed')
class ContextWindowTests(unittest.TestCase):
    def test_blocks_stay_whole_and_windows_stay_inside_blocks(self):
        block = np.repeat(np.arange(30), 20)
        y = np.repeat(np.arange(30) % 3, 20)
        X = np.arange(len(block), dtype=np.float32)[:, None].repeat(2, 1)
        parts = calibrate._split_blocks(y, block, seed=1)
        self.assertTrue((sum(p.astype(int) for p in parts) == 1).all())
        for p in parts:
            for b in np.unique(block[p]):
                self.assertTrue(p[block == b].all())  # a block never straddles two splits
        windows = calibrate.Windows(X, block, 4, pool=np.arange(len(block)))
        self.assertFalse(windows.eligible[[0, 1, 2, 20, 22]].any())
        self.assertTrue(windows.eligible[[3, 23, 39]].all())
        S = windows(np.array([3, 39]))
        self.assertEqual(S.shape, (2, 4, 2))
        self.assertEqual(S[1, :, 0].tolist(), [36, 37, 38, 39])
        mixed = windows(np.array([39]), rho=1.0, rng=np.random.default_rng(0))
        self.assertEqual(mixed[0, -1, 0], 39)  # the flow itself is never replaced


if __name__ == '__main__':
    unittest.main()
