"""Epoch QUBO (qubo_place.py): coefficients, one-hot encoding and a simulated run."""
import itertools
import unittest

from fogids.ps_model import PSModel, ps_sum_of_finish_times
from fogids.qubo_place import QuboPlace, build_epoch
from test_problem import instance, task


def epoch(gamma=1.0, lam=1.0):
    tasks = [task(i, 'gw0', 1.0, flows=100 * (i + 1)) for i in range(3)]
    p = instance(tasks).validate()
    q = build_epoch(PSModel(p), p, p.tasks, 1.0, lam, gamma, 0.0, 'light')
    return p, q


class EpochQuboTests(unittest.TestCase):
    def test_cpu_pairs_follow_processor_sharing(self):
        # Same node, weights 1/deadline = 1: the quadratic part of the CPU term of the
        # chosen options plus their linear CPU parts is the exact sum of finish times.
        p, q = epoch(gamma=1.0, lam=0.0)
        mips = p.node('gw0').mips
        chosen = [next(o for o in group if q.options[o][1:] == ('gw0', 'light')) for group in q.by_task]
        works = [p.model('light').work_mi(t.n_flows) for t in p.tasks]
        quadratic = sum(c for (a, b), c in q.pairs.items() if a in chosen and b in chosen)
        linear = sum(q.linear[o] for o in chosen)
        self.assertAlmostEqual((quadratic + linear) * mips, ps_sum_of_finish_times(works), places=6)

    def test_plain_qubo_matches_one_hot_energy_and_penalises_violations(self):
        _, q = epoch()
        penalty = 10 * (sum(abs(v) for v in q.linear) + sum(abs(v) for v in q.pairs.values()))
        qubo, offset = q.to_qubo(penalty)
        n = len(q.options)

        def energy(bits):
            return offset + sum(c * bits[u] * bits[v] for (u, v), c in qubo.items())

        for choice in itertools.product(*q.by_task):
            bits = [0] * n
            for o in choice:
                bits[o] = 1
            self.assertAlmostEqual(energy(bits), q.energy(list(choice)), places=6)
        self.assertGreater(energy([0] * n), max(q.energy(list(c)) for c in itertools.product(*q.by_task)))

    def test_anneal_finds_the_exhaustive_optimum_on_a_small_epoch(self):
        _, q = epoch(gamma=5.0)
        best = min(itertools.product(*q.by_task), key=lambda c: q.energy(list(c)))
        found, e = q.anneal([g[0] for g in q.by_task], 0.05, seed=1)
        self.assertAlmostEqual(e, q.energy(list(best)), places=6)
        self.assertAlmostEqual(q.energy(found), e, places=6)


class QuboPlaceSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fogids.bridge import compile_driver
        try:
            compile_driver()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))

    def test_every_task_is_placed_once(self):
        from fogids.bridge import simulate
        p = instance([task(i, f'gw{i % 2}', 1.0 + 0.05 * i, flows=200) for i in range(8)]).validate()
        end = simulate(p, QuboPlace(budget_s=0.005))
        self.assertTrue(all(r['finish_s'] is not None for r in end['results']))


if __name__ == '__main__':
    unittest.main()
