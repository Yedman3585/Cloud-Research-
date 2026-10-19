from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

from fogids.smokescreen import Engine, Estimator, generate  # noqa: E402
from fogids.smokescreen_qubo import AlertEvidence, AlertPSAnneal, PSAnneal, QuboSmoke, ValueGreedyAlerts  # noqa: E402
from test_smokescreen import tiny_config  # noqa: E402


def epoch_tasks(sc):
    first = sc.tasks[0].release_s
    return [t for t in sc.tasks if t.release_s <= first + 1.0][:8]


class SmokescreenQuboTests(unittest.TestCase):
    def test_qubo_matrix_matches_energy(self):
        sc = generate(tiny_config())
        policy = QuboSmoke(gamma=1.0)
        policy.reset(sc)
        tasks = epoch_tasks(sc)[:3]
        est = Estimator(Engine(sc), tasks[-1].release_s)
        q = policy.build(est, tasks, tasks[-1].release_s)
        matrix, offset = q.to_qubo(penalty=100.0)
        index, k = {}, 0
        for i, opts in enumerate(q.options):
            for j in range(len(opts)):
                index[(i, j)] = k
                k += 1
        rng = random.Random(1)
        for _ in range(20):
            choice = [rng.randrange(len(opts)) for opts in q.options]
            bits = {index[(i, c)] for i, c in enumerate(choice)}
            e = offset + sum(v for (u, w), v in matrix.items() if u in bits and w in bits)
            self.assertAlmostEqual(e, q.energy(choice), places=6)

    def test_epoch_ps_move_delta_matches_energy(self):
        sc = generate(tiny_config())
        policy = PSAnneal()
        policy.reset(sc)
        tasks = epoch_tasks(sc)
        now = tasks[-1].release_s
        m = policy.build_ps(Estimator(Engine(sc), now), tasks, now)
        rng = random.Random(2)
        m._setup([0] * len(tasks))
        e = m.energy()
        for _ in range(200):
            i = rng.randrange(len(tasks))
            e += m.move(i, rng.randrange(len(m.options[i])))
            self.assertAlmostEqual(e, m.energy(), places=6)

    def test_policies_run(self):
        sc = generate(tiny_config())
        for policy in (QuboSmoke(budget_s=0.005), PSAnneal(budget_s=0.005), AlertPSAnneal(budget_s=0.005),
                       ValueGreedyAlerts()):
            r = Engine(sc).run(policy)
            self.assertLessEqual(r['objective'], 1.0 + 1e-9)
            if hasattr(policy, 'stats'):
                self.assertGreater(policy.stats['epochs'], 0)

    def test_alerts_reveal_low_rate_attacks(self):
        sc = generate(tiny_config())
        engine = Engine(sc)
        engine.run(PSAnneal(budget_s=0.005))
        self.assertTrue(engine.alerts)
        self.assertTrue(all(a.time_s >= 0 and a.quiet_alerts >= 0 for a in engine.alerts))
        evidence = AlertEvidence(sc, horizon_s=10.0)
        evidence.update(engine, sc.horizon_s)
        # Both gateways carry a 15 flows/s web attack: the excess over the false alerts
        # expected from benign traffic is positive.
        self.assertGreater(max(evidence.share(g) for g in ('gw0', 'gw1')), 0.0)


if __name__ == '__main__':
    unittest.main()
