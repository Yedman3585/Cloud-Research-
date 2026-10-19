from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

from fogids.smokescreen import POLICIES, Engine, generate  # noqa: E402
from test_smokescreen import tiny_config  # noqa: E402


@unittest.skipIf(shutil.which('java') is None, 'Java not installed')
class SmokescreenIFogSimTests(unittest.TestCase):
    def test_ifogsim_reproduces_the_python_simulator(self):
        from fogids.smokescreen_ifogsim import simulate_ifogsim
        sc = generate(tiny_config())
        for name in ('greedy-v2', 'no-shed'):
            reference = simulate_ifogsim(sc, POLICIES[name]())
            python = Engine(sc).run(POLICIES[name]())
            self.assertLess(reference['finish_error_max_s'], 0.005, name)
            self.assertAlmostEqual(reference['objective'], python['objective'], places=3, msg=name)


class FastAnnealTests(unittest.TestCase):
    def test_kernel_energy_matches_python_model(self):
        import random
        try:
            from fogids.fast_anneal import FastAnnealer, _get_kernel
        except ImportError:
            self.skipTest('numpy missing')
        kernel = _get_kernel()
        if kernel is None:
            self.skipTest('numba not installed')
        import numpy as np
        from fogids.smokescreen import Estimator
        from fogids.smokescreen_qubo import PSAnneal
        sc = generate(tiny_config())
        policy = PSAnneal(fast=False)
        policy.reset(sc)
        tasks = [t for t in sc.tasks if t.release_s <= sc.tasks[0].release_s + 1.0][:8]
        now = tasks[-1].release_s
        m = policy.build_ps(Estimator(Engine(sc), now), tasks, now)
        packed = FastAnnealer.pack(m)
        rng = random.Random(3)
        for _ in range(5):
            choice = [rng.randrange(len(o)) for o in m.options]
            _, e = kernel(*packed, np.array(choice, np.int64), 0, 0.5, 0.005, m.kappa, 1)
            m._setup(choice)
            self.assertAlmostEqual(e, m.energy(), places=6)
        best, e = kernel(*packed, np.zeros(len(tasks), np.int64), 20000, 0.5, 0.005, m.kappa, 1)
        m._setup([int(x) for x in best])
        self.assertAlmostEqual(e, m.energy(), places=6)


if __name__ == '__main__':
    unittest.main()
