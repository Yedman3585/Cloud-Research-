"""Policies built on ``PSModel`` (processor-sharing replica of the simulator).

* ``ps-greedy``: tasks by decreasing risk weight; for each, the best model on any
  eligible node such that the task is predicted to meet its deadline *and* no task
  already committed to that node, predicted to meet its own deadline, is pushed past
  it. Otherwise the option that creates the fewest new misses, then finishes first.
* ``Shadow``: runs another policy unchanged and replays its decisions through a
  ``PSModel``, so two latency estimators can be compared on identical decisions.

Both record ``predicted`` (task id -> finish time predicted at decision time) for the
fidelity analysis in ``oracle.py``.
"""
from .policies import Policy, _models_by_quality
from .ps_model import PSModel

TOLERANCE_S = 1e-5


class PSGreedy(Policy):
    name = 'ps-greedy'

    def __init__(self, slack=0.0):
        # A model other than the fastest must leave this share of the deadline unused,
        # so that tasks of later epochs still fit next to it.
        self.slack = slack
        if slack:
            self.name = f'ps-greedy-s{slack:g}'

    def reset(self, instance):
        super().reset(instance)
        self.model = PSModel(instance)
        self.quality = [m.name for m in _models_by_quality(instance)]
        self.fastest = min(instance.models, key=lambda m: m.mi_per_flow).name
        self.predicted = {}

    def _deadline(self, task_id):
        t = self.instance.tasks[task_id]
        return t.release_s + t.deadline_s + TOLERANCE_S

    def _option(self, task, node, model, now, before):
        own, _, _, after = self.model.estimate(task, node, model, now)
        budget = task.deadline_s * (1.0 if model == self.fastest else 1.0 - self.slack)
        meets = own <= task.release_s + budget + TOLERANCE_S
        broken = sum(1 for k, f in after.items()
                     if before.get(k, f) <= self._deadline(k) < f)
        return own, meets, broken

    def decide(self, snapshot):
        self.model.sync(snapshot)
        now = snapshot.t
        out = []
        for t in sorted(snapshot.tasks, key=lambda t: (-t.weight, t.id)):
            best, fallback = None, None
            for node in self.instance.eligible(t):
                before = self.model.finishes(node, now)
                for rank, model in enumerate(self.quality):
                    own, meets, broken = self._option(t, node, model, now, before)
                    if meets and not broken:
                        key = (rank, own)
                        if best is None or key < best[0]:
                            best = (key, node, model)
                    key = (broken + (not meets), own)
                    if fallback is None or key < fallback[0]:
                        fallback = (key, node, model)
            _, node, model = best or fallback
            self.predicted[t.id] = self.model.commit(t, node, model, now)
            out.append((t.id, node, model))
        return out


class Shadow(Policy):
    """Decisions of ``inner``; predictions of both ``inner`` and a PSModel."""

    def __init__(self, inner):
        self.inner = inner
        self.name = f'{inner.name}+ps'

    def reset(self, instance):
        super().reset(instance)
        self.inner.reset(instance)
        self.model = PSModel(instance)
        self.predicted = {}

    def decide(self, snapshot):
        self.model.sync(snapshot)
        out = self.inner.decide(snapshot)
        tasks = {t.id: t for t in snapshot.tasks}
        for task_id, node, model in out:  # same order in which the inner policy committed
            self.predicted[task_id] = self.model.commit(tasks[task_id], node, model, snapshot.t)
        return out
