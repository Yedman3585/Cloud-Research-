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


class QueueModel:
    """A scheduler-side estimate of when a task would finish, from the policy's own view.

    Uplinks are FIFO queues: a transfer starts when the link is free, occupies it for
    bytes / bandwidth and then adds the propagation latency, as in the simulator.
    A node's CPU is a fluid backlog of MI that drains at the node's MIPS; a task that
    arrives finds the backlog left at its arrival time and finishes after that backlog
    plus its own work. Resident models are tracked so cold starts are predicted.
    This is an estimate built only from decisions the policy made; it ignores
    processor sharing among tasks that arrive later and is not the simulator's truth.
    """

    def __init__(self, instance):
        self.p = instance
        self.link_free = {n.name: 0.0 for n in instance.nodes}
        self.backlog = {n.name: 0.0 for n in instance.nodes}
        self.backlog_t = {n.name: 0.0 for n in instance.nodes}
        self.resident = {n.name: set(n.resident) for n in instance.nodes}

    def sync(self, snapshot):
        for name, state in snapshot.nodes.items():
            self.resident[name] = set(state['resident'])

    def _backlog_at(self, node, t):
        mips = self.p.node(node).mips
        return max(0.0, self.backlog[node] - mips * (t - self.backlog_t[node]))

    def estimate(self, task, node, model, now):
        """Return (finish_time, work_mi, per-link transfer completion times)."""
        t, done = now, []
        for link in self.p.path_links(task.gateway, node):
            spec = self.p.node(link)
            start = max(t, self.link_free[link])
            end = start + task.input_bytes / spec.uplink_bytes_s
            done.append((link, end))
            t = end + spec.uplink_latency_s
        m, mips = self.p.model(model), self.p.node(node).mips
        work = m.work_mi(task.n_flows) + (0.0 if model in self.resident[node] else m.load_s * mips)
        finish = t + (self._backlog_at(node, t) + work) / mips
        return finish, work, done

    def commit(self, task, node, model, now):
        finish, work, done = self.estimate(task, node, model, now)
        for link, end in done:
            self.link_free[link] = end
        self.backlog[node] = self._backlog_at(node, now) + work
        self.backlog_t[node] = now
        self.resident[node].add(model)
        return finish


class QueuePolicy(Policy):
    """Base class: tasks in order of decreasing risk weight, finish times from QueueModel."""

    def reset(self, instance):
        super().reset(instance)
        self.queues = QueueModel(instance)
        self.quality = [m.name for m in _models_by_quality(instance)]

    def decide(self, snapshot):
        self.queues.sync(snapshot)
        out = []
        for t in sorted(snapshot.tasks, key=lambda t: (-t.weight, t.id)):
            node, model = self.choose(t, snapshot.t)
            self.queues.commit(t, node, model, snapshot.t)
            out.append((t.id, node, model))
        return out

    def latency(self, task, node, model, now):
        return self.queues.estimate(task, node, model, now)[0] - task.release_s

    def choose(self, task, now):
        raise NotImplementedError


class EdgeAdaptive(QueuePolicy):
    """Never offloads: the best model at the gateway that is predicted to meet the deadline,
    otherwise the fastest model there."""
    name = 'edge-adaptive'

    def choose(self, task, now):
        gw = task.gateway
        for model in self.quality:
            if self.latency(task, gw, model, now) <= task.deadline_s:
                return gw, model
        return gw, min(self.quality, key=lambda m: self.latency(task, gw, m, now))


class QueueGreedy(QueuePolicy):
    """Over all eligible nodes: the best model predicted to meet the deadline (fastest node
    for it); otherwise the fastest (node, model) pair. Like greedy-finish, but with uplink
    queues and draining CPU backlogs in the estimate."""
    name = 'queue-greedy'

    def choose(self, task, now):
        options = [(self.latency(task, node, model, now), rank, node, model)
                   for node in self.instance.eligible(task) for rank, model in enumerate(self.quality)]
        meeting = [o for o in options if o[0] <= task.deadline_s]
        best = min(meeting, key=lambda o: (o[1], o[0])) if meeting else min(options)
        return best[2], best[3]


class RiskSplit(QueuePolicy):
    """Short-deadline (high-risk) windows run the fastest model at their gateway; all other
    windows run the best model on the eligible node predicted to finish first."""
    name = 'risk-split'

    def reset(self, instance):
        super().reset(instance)
        self.shortest = min((t.deadline_s for t in instance.tasks), default=0.0)

    def choose(self, task, now):
        if task.deadline_s <= self.shortest:
            gw = task.gateway
            return gw, min(self.quality, key=lambda m: self.latency(task, gw, m, now))
        best = self.quality[0]
        node = min(self.instance.eligible(task), key=lambda n: self.latency(task, n, best, now))
        return node, best


class RandomPolicy(Policy):
    """Uniformly random eligible node and model; seeded by the instance seed."""
    name = 'random'

    def reset(self, instance):
        import random
        super().reset(instance)
        self.rng = random.Random(instance.seed)

    def decide(self, snapshot):
        out = []
        for t in snapshot.tasks:
            nodes = self.instance.eligible(t)
            node = nodes[int(self.rng.random() * len(nodes))]
            models = self.instance.models
            out.append((t.id, node, models[int(self.rng.random() * len(models))].name))
        return out


NAMED = {'greedy-finish': GreedyFinish, 'edge-adaptive': EdgeAdaptive, 'queue-greedy': QueueGreedy,
         'risk-split': RiskSplit, 'random': RandomPolicy}


def by_name(name):
    if name in NAMED:
        return NAMED[name]()
    tier, _, model = name.partition('-')
    if tier in ('edge', 'fog', 'cloud') and model:
        return Fixed(tier, model)
    raise ValueError(f'Unknown policy {name!r}; use one of {sorted(NAMED)} or <edge|fog|cloud>-<model>')
