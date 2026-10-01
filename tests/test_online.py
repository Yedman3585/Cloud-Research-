"""Epoch-based co-simulation checked against hand-computed (analytic) timings."""
import unittest

from fogids.bridge import SimulationError, compile_driver, simulate
from fogids.generator import generate, load_config
from fogids.metrics import summarize
from fogids.policies import Fixed, GreedyFinish, Policy
from test_problem import ROOT_CONFIG, instance, task

DELTA = 0.011  # CPU update interval 0.01 s


class Scripted(Policy):
    """Assigns tasks from a fixed table {task_id: (node, model)}."""
    name = 'scripted'

    def __init__(self, table):
        self.table = table
        self.snapshots = []

    def decide(self, snapshot):
        self.snapshots.append(snapshot)
        return [(t.id, *self.table[t.id]) for t in snapshot.tasks]


def latencies(p, end):
    return {r['id']: r['finish_s'] - p.tasks[r['id']].release_s for r in end['results']}


class OnlineSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compile_driver()

    def run_table(self, tasks, table, **kwargs):
        p = instance(tasks).validate()
        policy = Scripted(table)
        end = simulate(p, policy, **kwargs)
        return p, policy, end

    def test_local_execution_on_epoch_boundary(self):
        p, policy, end = self.run_table([task(0, 'gw0', 1.0)], {0: ('gw0', 'light')})
        self.assertAlmostEqual(latencies(p, end)[0], 110 / 1000, delta=DELTA)
        self.assertAlmostEqual(policy.snapshots[0].t, 1.0)  # released on the boundary: same epoch

    def test_release_waits_for_next_epoch(self):
        p, _, end = self.run_table([task(0, 'gw0', 1.05)], {0: ('gw0', 'light')})
        self.assertAlmostEqual(latencies(p, end)[0], 0.05 + 0.11, delta=DELTA)

    def test_cloud_latency_is_transfer_plus_compute(self):
        p, _, end = self.run_table([task(0, 'gw0', 1.0)], {0: ('cloud', 'full')})
        t = p.tasks[0]
        expected = p.transfer_s(t, 'cloud') + p.compute_s(t, 'cloud', 'full')
        self.assertAlmostEqual(latencies(p, end)[0], expected, delta=DELTA)

    def test_cold_start_is_paid_once(self):
        tasks = [task(0, 'gw0', 1.0), task(1, 'gw0', 5.0)]
        p, policy, end = self.run_table(tasks, {0: ('gw0', 'full'), 1: ('gw0', 'full')})
        lat = latencies(p, end)
        self.assertAlmostEqual(lat[0], 2.0 + 0.5, delta=DELTA)  # load_s + 500 MI / 1000 MIPS
        self.assertAlmostEqual(lat[1], 0.5, delta=DELTA)
        self.assertEqual([r['cold_start'] for r in end['results']], [True, False])
        self.assertIn('full', policy.snapshots[1].nodes['gw0']['resident'])

    def test_tasks_share_a_node_cpu(self):
        tasks = [task(0, 'gw0', 1.0), task(1, 'gw1', 1.0)]
        p, _, end = self.run_table(tasks, {0: ('fog0', 'light'), 1: ('fog0', 'light')})
        lat = latencies(p, end)
        # Equal transfers end together; two 110 MI tasks then share 4000 MIPS.
        transfer = p.transfer_s(p.tasks[0], 'fog0')
        for i in (0, 1):
            self.assertAlmostEqual(lat[i], transfer + 2 * 110 / 4000, delta=DELTA)

    def test_fixed_decision_delay_is_charged(self):
        p, _, end = self.run_table([task(0, 'gw0', 1.0)], {0: ('gw0', 'light')}, decision_time=0.25)
        self.assertAlmostEqual(latencies(p, end)[0], 0.25 + 0.11, delta=DELTA)

    def test_snapshot_reports_completions_and_outstanding_work(self):
        tasks = [task(0, 'gw0', 1.0), task(1, 'gw0', 3.0)]
        _, policy, _ = self.run_table(tasks, {0: ('gw0', 'light'), 1: ('gw0', 'light')})
        second = policy.snapshots[1]
        self.assertEqual([i for i, _ in second.completed], [0])
        self.assertEqual(second.nodes['gw0']['outstanding'], 0)

    def test_ineligible_assignment_is_rejected(self):
        with self.assertRaises(SimulationError):
            self.run_table([task(0, 'gw0', 1.0)], {0: ('gw1', 'light')})

    def test_generated_instance_is_deterministic_and_completes(self):
        p = generate(load_config(ROOT_CONFIG))
        first = simulate(p, GreedyFinish())
        second = simulate(p, GreedyFinish())
        self.assertEqual(first['results'], second['results'])
        summary = summarize(p, first)
        self.assertEqual(summary['all']['tasks'], len(p.tasks))
        self.assertEqual(summary['all']['unfinished'], 0)
        local = summarize(p, simulate(p, Fixed('edge', 'light')))
        self.assertEqual(local['placement'], {'edge/light': len(p.tasks)})
        self.assertEqual(local['cloud_cost'], 0.0)


if __name__ == '__main__':
    unittest.main()
