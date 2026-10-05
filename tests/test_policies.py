"""Scheduler-side queue model and the reference policies (no simulator needed)."""
import unittest

from fogids.bridge import Snapshot
from fogids.policies import NAMED, QueueModel, by_name
from test_problem import instance, task


def snapshot(p, tasks, t):
    return Snapshot(t=t, epoch=int(round(t / p.epoch_s)), tasks=tasks, completed=[],
                    nodes={n.name: {'outstanding': 0, 'resident': list(n.resident)} for n in p.nodes})


class QueueModelTests(unittest.TestCase):
    def test_single_task_matches_queue_free_times(self):
        p = instance().validate()
        t = p.tasks[0]
        q = QueueModel(p)
        for node in p.eligible(t):
            finish, _, _ = q.estimate(t, node, 'light', t.release_s)
            expected = t.release_s + p.transfer_s(t, node) + p.compute_s(t, node, 'light')
            self.assertAlmostEqual(finish, expected)

    def test_second_transfer_waits_for_the_link(self):
        p = instance([task(0, 'gw0', 1.0), task(1, 'gw0', 1.0)]).validate()
        q = QueueModel(p)
        first = q.commit(p.tasks[0], 'fog0', 'light', 1.0)
        second, _, _ = q.estimate(p.tasks[1], 'fog0', 'light', 1.0)
        serialization = p.tasks[1].input_bytes / p.node('gw0').uplink_bytes_s
        self.assertGreaterEqual(second - first, serialization - 1e-9)

    def test_cpu_backlog_drains_over_time(self):
        p = instance([task(0, 'gw0', 1.0), task(1, 'gw0', 5.0)]).validate()
        q = QueueModel(p)
        q.commit(p.tasks[0], 'gw0', 'light', 1.0)
        later, _, _ = q.estimate(p.tasks[1], 'gw0', 'light', 5.0)
        self.assertAlmostEqual(later, 5.0 + p.compute_s(p.tasks[1], 'gw0', 'light'))

    def test_cold_start_predicted_once(self):
        p = instance([task(0, 'gw0', 1.0), task(1, 'gw0', 9.0)]).validate()
        q = QueueModel(p)
        cold = q.commit(p.tasks[0], 'gw0', 'full', 1.0) - 1.0
        warm, _, _ = q.estimate(p.tasks[1], 'gw0', 'full', 9.0)
        self.assertAlmostEqual(cold - (warm - 9.0), p.model('full').load_s)


class PolicyTests(unittest.TestCase):
    def test_every_policy_assigns_each_task_to_an_eligible_node(self):
        p = instance([task(i, 'gw0' if i % 2 else 'gw1', 1.0 + i * 0.01) for i in range(6)]).validate()
        names = [*NAMED, 'edge-light', 'fog-full', 'cloud-light']
        for name in names:
            policy = by_name(name)
            policy.reset(p)
            out = policy.decide(snapshot(p, p.tasks, 1.1))
            self.assertEqual(sorted(i for i, _, _ in out), [t.id for t in p.tasks], name)
            for i, node, model in out:
                self.assertIn(node, p.eligible(p.tasks[i]), name)
                self.assertIn(model, ('light', 'full'), name)

    def test_edge_adaptive_never_offloads(self):
        p = instance([task(i, 'gw0', 1.0) for i in range(5)]).validate()
        policy = by_name('edge-adaptive')
        policy.reset(p)
        self.assertTrue(all(node == 'gw0' for _, node, _ in policy.decide(snapshot(p, p.tasks, 1.0))))

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            by_name('nonsense')


if __name__ == '__main__':
    unittest.main()
