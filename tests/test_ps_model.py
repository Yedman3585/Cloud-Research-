"""Processor-sharing replica (ps_model.py) against analytic values and the simulator."""
import random
import unittest

from fogids.policies import Policy
from fogids.ps_model import PSModel, ps_finish_times, ps_sum_of_finish_times
from fogids.ps_policies import PSGreedy
from test_problem import instance, task


class ProcessorSharingTests(unittest.TestCase):
    def test_late_short_task_overtakes_running_long_task(self):
        # 1000 MIPS: A (1000 MI) alone for 0.1 s, then A and B (200 MI) share the CPU.
        f = ps_finish_times(1000, 0.0, {}, [(0.0, 'A', 1000), (0.1, 'B', 200)])
        self.assertAlmostEqual(f['B'], 0.5)
        self.assertAlmostEqual(f['A'], 1.2)

    def test_running_tasks_are_slowed_by_an_arrival(self):
        f = ps_finish_times(100, 0.0, {'A': 100}, [(0.5, 'B', 100)])
        self.assertAlmostEqual(f['A'], 1.5)  # 50 MI alone, then 50 MI at half speed
        self.assertAlmostEqual(f['B'], 2.0)

    def test_closed_form_matches_simulation(self):
        rng = random.Random(3)
        for _ in range(50):
            works = [rng.uniform(1, 100) for _ in range(rng.randint(1, 8))]
            f = ps_finish_times(1.0, 0.0, {}, [(0.0, i, w) for i, w in enumerate(works)])
            self.assertAlmostEqual(sum(f.values()), ps_sum_of_finish_times(works), places=6)

    def test_closed_form_is_a_quadratic_in_placement_bits(self):
        # sum_i a_i y_i + 2 sum_{i<j} min(a_i, a_j) y_i y_j for every subset of tasks.
        works = [5.0, 2.0, 7.0, 3.0]
        for mask in range(16):
            y = [(mask >> i) & 1 for i in range(4)]
            quad = sum(a * b for a, b in zip(works, y)) + 2 * sum(
                min(works[i], works[j]) * y[i] * y[j] for i in range(4) for j in range(i + 1, 4))
            self.assertAlmostEqual(quad, ps_sum_of_finish_times([a for a, b in zip(works, y) if b]))

    def test_model_advances_and_reports_the_delay_it_causes(self):
        p = instance([task(0, 'gw0', 1.0, flows=990), task(1, 'gw0', 1.1, flows=190)]).validate()
        m = PSModel(p)
        first = m.commit(p.tasks[0], 'gw0', 'light', 1.0)
        self.assertAlmostEqual(first, 2.0)  # 1000 MI on 1000 MIPS
        own, _, _, others = m.estimate(p.tasks[1], 'gw0', 'light', 1.1)
        self.assertAlmostEqual(own, 1.5)
        self.assertAlmostEqual(others[0], 2.2)  # the earlier task is pushed back


class Scripted(Policy):
    name = 'scripted'

    def __init__(self, table):
        self.table = table

    def decide(self, snapshot):
        return [(t.id, *self.table[t.id]) for t in snapshot.tasks]


class SimulatorAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fogids.bridge import compile_driver
        try:
            compile_driver()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))

    def test_model_predicts_simulated_finish_times(self):
        from fogids.bridge import simulate
        tasks = [task(0, 'gw0', 1.0, flows=990), task(1, 'gw1', 1.0, flows=300),
                 task(2, 'gw0', 1.1, flows=190), task(3, 'gw0', 1.2, flows=50)]
        table = {0: ('gw0', 'light'), 1: ('fog0', 'light'), 2: ('gw0', 'light'), 3: ('fog0', 'light')}
        p = instance(tasks).validate()
        end = simulate(p, Scripted(table))
        m = PSModel(p)
        by_epoch = {}
        for t in p.tasks:
            by_epoch.setdefault(round(t.release_s, 1), []).append(t)
        for now in sorted(by_epoch):
            for t in by_epoch[now]:
                m.commit(t, *table[t.id], now)
        predicted = {}
        for node in ('gw0', 'fog0'):
            predicted.update(m.finishes(node, 1.2))
        for r in end['results']:
            self.assertAlmostEqual(predicted[r['id']], r['finish_s'], delta=0.011)

    def test_ps_greedy_assigns_every_task_once(self):
        from fogids.bridge import simulate
        p = instance([task(i, f'gw{i % 2}', 1.0 + 0.05 * i, flows=200) for i in range(8)]).validate()
        end = simulate(p, PSGreedy())
        self.assertTrue(all(r['finish_s'] is not None for r in end['results']))


if __name__ == '__main__':
    unittest.main()
