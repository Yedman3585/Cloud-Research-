"""Reference placement policies for IDS-PLACE.

A policy sees only scheduler-visible information: the released tasks (without
ground-truth labels), the node state in the snapshot, and the static instance.
"""


class Policy:
    name = 'policy'

    def reset(self, instance):
        self.instance = instance

    def decide(self, snapshot):
        """Return [(task_id, node_name, model_name)] for every task in the snapshot."""
        raise NotImplementedError


def _models_by_quality(instance):
    """Models ordered from best to worst detection quality: mean recall minus false-positive rate."""
    return sorted(instance.models,
                  key=lambda m: -(sum(m.recall.values()) / max(1, len(m.recall)) - m.false_positive_rate))


class Fixed(Policy):
    """Every task to one tier (its gateway, fog parent, or the cloud) with one model."""

    def __init__(self, tier, model):
        self.tier, self.model = tier, model
        self.name = f'{tier}-{model}'

    def decide(self, snapshot):
        out = []
        for t in snapshot.tasks:
            nodes = self.instance.eligible(t)
            node = next(n for n in nodes if self.instance.node(n).tier == self.tier)
            out.append((t.id, node, self.model))
        return out


class GreedyFinish(Policy):
    """Highest risk weight first; best model whose estimated latency meets the deadline.

    Latency estimate: queue-free transfer + (backlog of assigned, unfinished work +
    own work) / MIPS + cold start. Falls back to the fastest (node, model) pair when
    no option meets the deadline. A deliberately simple, transparent baseline.
    """
    name = 'greedy-finish'

    def reset(self, instance):
        super().reset(instance)
        self.backlog = {n.name: 0.0 for n in instance.nodes}
        self.pending = {}
        self.quality = [m.name for m in _models_by_quality(instance)]

    def decide(self, snapshot):
        for task_id, _ in snapshot.completed:
            node, work = self.pending.pop(task_id)
            self.backlog[node] = max(0.0, self.backlog[node] - work)
        p = self.instance
        out = []
        for t in sorted(snapshot.tasks, key=lambda t: (-t.weight, t.id)):
            options = []
            for node in p.eligible(t):
                spec = p.node(node)
                resident = snapshot.nodes[node]['resident']
                for rank, model in enumerate(self.quality):
                    m = p.model(model)
                    cold = 0.0 if model in resident else m.load_s
                    work = m.work_mi(t.n_flows) + cold * spec.mips
                    latency = p.transfer_s(t, node) + (self.backlog[node] + work) / spec.mips
                    options.append((latency <= t.deadline_s, rank, latency, node, model, work))
            meeting = [o for o in options if o[0]]
            best = min(meeting, key=lambda o: (o[1], o[2])) if meeting else min(options, key=lambda o: o[2])
            _, _, _, node, model, work = best
            self.backlog[node] += work
            self.pending[t.id] = (node, work)
            out.append((t.id, node, model))
        return out


def by_name(name):
    if name == 'greedy-finish':
        return GreedyFinish()
    tier, _, model = name.partition('-')
    if tier in ('edge', 'fog', 'cloud') and model:
        return Fixed(tier, model)
    raise ValueError(f'Unknown policy {name!r}; use greedy-finish or <edge|fog|cloud>-<model>')
