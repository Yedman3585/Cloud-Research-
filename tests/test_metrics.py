"""Detection-quality metrics computed from a hand-made end message (no simulator needed)."""
import unittest

from fogids.metrics import summarize, task_rows
from fogids.problem import Model, Task
from test_problem import instance


def task(i, release, flows, attack, label):
    return Task(id=i, gateway='gw0', release_s=release, deadline_s=1.0, n_flows=flows,
                attack_flows=attack, label=label, input_bytes=500, prefilter_score=0.5, criticality=1.0)


def end_message(assignments, finish):
    return {'t': 10.0, 'energy_j': {}, 'results': [
        {'id': i, 'node': node, 'model': model, 'dispatch_s': 1.0, 'finish_s': finish.get(i), 'cold_start': False}
        for i, (node, model) in assignments.items()]}


class MetricsTests(unittest.TestCase):
    def setUp(self):
        p = instance([task(0, 1.0, 100, 90, 'ddos'), task(1, 2.0, 100, 10, 'web'), task(2, 3.0, 50, 0, 'benign')])
        models = [Model('light', 1, 0.01, 10, 0.1, recall={'ddos': 1.0, 'web': 0.5}, false_positive_rate=0.2),
                  Model('full', 2, 0.10, 20, 0.1, recall={'ddos': 1.0, 'web': 0.9}, false_positive_rate=0.1)]
        self.p = type(p)(name=p.name, seed=0, horizon_s=10.0, window_s=1.0, epoch_s=0.1,
                         nodes=p.nodes, models=models, tasks=p.tasks).validate()

    def test_false_alerts_and_macro_missed_detection(self):
        end = end_message({0: ('gw0', 'light'), 1: ('gw0', 'light'), 2: ('gw0', 'full')},
                          {0: 1.1, 1: 2.1, 2: 3.1})
        s = summarize(self.p, end)
        # Benign flows: 10 + 90 + 50; false alerts 10*0.2 + 90*0.2 + 50*0.1 = 25.
        self.assertAlmostEqual(s['false_alert_flows'], 25.0)
        self.assertAlmostEqual(s['false_alert_rate'], 25.0 / 150)
        self.assertEqual(s['missed_attack_by_class'], {'ddos': 0.0, 'web': 0.5})
        self.assertAlmostEqual(s['missed_attack_macro'], 0.25)
        self.assertAlmostEqual(s['missed_attack_flow_fraction'], 5 / 100)

    def test_heavier_model_reduces_both_error_kinds(self):
        light = summarize(self.p, end_message({i: ('gw0', 'light') for i in range(3)}, {0: 1.1, 1: 2.1, 2: 3.1}))
        full = summarize(self.p, end_message({i: ('gw0', 'full') for i in range(3)}, {0: 1.1, 1: 2.1, 2: 3.1}))
        self.assertLess(full['false_alert_rate'], light['false_alert_rate'])
        self.assertLess(full['missed_attack_macro'], light['missed_attack_macro'])

    def test_unfinished_task_misses_attacks_and_raises_no_alerts(self):
        rows = task_rows(self.p, end_message({i: ('gw0', 'full') for i in range(3)}, {0: 1.1, 2: 3.1}))
        unfinished = rows[1]
        self.assertTrue(unfinished['missed'])
        self.assertEqual(unfinished['missed_attack_flows'], 10)
        self.assertEqual(unfinished['false_alert_flows'], 0.0)


if __name__ == '__main__':
    unittest.main()
