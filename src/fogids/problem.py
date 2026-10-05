"""IDS-PLACE problem model: topology, IDS model variants, inference tasks, instances.

An instance is a fully specified, simulator-independent description of one
experiment: a tree of edge, fog and cloud nodes, the available IDS model
variants, and the stream of feature-window inference tasks. Instances are
plain JSON (schema ``ids-place/1``) so that Python solvers, the Java simulator
and later analyses all read the same file.

Units: seconds, MI (million instructions), MIPS, bytes, bytes/s, watts, MB.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

SCHEMA = 'ids-place/1'
TIERS = ('edge', 'fog', 'cloud')
BENIGN = 'benign'
DIGITS = 9  # stored float precision; keeps JSON identical across platforms


def _round(value):
    return round(value, DIGITS) if isinstance(value, float) else value


@dataclass(frozen=True)
class Node:
    name: str
    tier: str
    parent: str | None
    mips: float
    memory_mb: int
    uplink_bytes_s: float | None  # link to parent; None for the root
    uplink_latency_s: float | None
    idle_w: float
    busy_w: float
    cost_per_gi: float  # monetary cost per 1000 MI executed (cloud billing)
    resident: tuple = ()  # model names loaded at time zero
    criticality: float = 1.0  # weight of the devices behind an edge gateway


@dataclass(frozen=True)
class Model:
    name: str
    fixed_mi: float
    mi_per_flow: float
    memory_mb: int
    load_s: float  # cold-start time when the model is not resident on a node
    recall: dict = field(default_factory=dict)  # attack class -> recall in [0, 1]
    false_positive_rate: float = 0.0  # fraction of benign flows flagged as attacks

    def work_mi(self, n_flows):
        return self.fixed_mi + self.mi_per_flow * n_flows


@dataclass(frozen=True)
class Task:
    """One feature window (or chunk of it) that is ready for IDS inference."""
    id: int
    gateway: str
    release_s: float  # absolute time the features are ready at the gateway
    deadline_s: float  # relative to release
    n_flows: int
    attack_flows: int
    label: str  # dominant ground-truth class; never visible to schedulers
    input_bytes: int
    prefilter_score: float  # noisy attack score visible to schedulers, [0, 1]
    criticality: float

    @property
    def estimated_benign_flows(self):
        """Scheduler-side estimate of benign flows, from the pre-filter score."""
        return self.n_flows * (1.0 - self.prefilter_score)

    @property
    def weight(self):
        """Risk weight used in objectives: device criticality x (floor + score)."""
        return self.criticality * (0.1 + self.prefilter_score)


@dataclass
class Instance:
    name: str
    seed: int
    horizon_s: float
    window_s: float
    epoch_s: float
    nodes: list
    models: list
    tasks: list
    meta: dict = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self):
        self._nodes = {n.name: n for n in self.nodes}
        self._models = {m.name: m for m in self.models}

    # Lookups -------------------------------------------------------------
    def node(self, name):
        return self._nodes[name]

    def model(self, name):
        return self._models[name]

    def tier(self, tier):
        return [n for n in self.nodes if n.tier == tier]

    def ancestors(self, name):
        """The node itself followed by its parents up to the root."""
        chain = []
        while name is not None:
            chain.append(name)
            name = self._nodes[name].parent
        return chain

    def eligible(self, task):
        """Nodes allowed to run a task: its gateway and the gateway's ancestors.

        v1 uses the simulator's tree topology; lateral fog-to-fog offloading
        would need links that the tree does not have.
        """
        return self.ancestors(task.gateway)

    def path_links(self, gateway, target):
        """Uplinks (named by their child node) traversed from gateway to target."""
        chain = self.ancestors(gateway)
        if target not in chain:
            raise ValueError(f'{target} is not on the path from {gateway} to the root')
        return chain[:chain.index(target)]

    def transfer_s(self, task, target):
        """Queue-free transfer time of the task input from its gateway to target."""
        return sum(self._nodes[l].uplink_latency_s + task.input_bytes / self._nodes[l].uplink_bytes_s
                   for l in self.path_links(task.gateway, target))

    def compute_s(self, task, target, model):
        """Exclusive-CPU execution time, including a cold start if not resident."""
        node, m = self._nodes[target], self._models[model]
        cold = 0.0 if model in node.resident else m.load_s
        return cold + m.work_mi(task.n_flows) / node.mips

    # Validation ----------------------------------------------------------
    def validate(self):
        if self.schema != SCHEMA:
            raise ValueError(f'Unsupported schema {self.schema!r}')
        if len(self._nodes) != len(self.nodes) or len(self._models) != len(self.models):
            raise ValueError('Node and model names must be unique')
        if min(self.horizon_s, self.window_s, self.epoch_s) <= 0:
            raise ValueError('Horizon, window and epoch must be positive')
        roots = [n for n in self.nodes if n.parent is None]
        if len(roots) != 1 or roots[0].tier != 'cloud':
            raise ValueError('Topology must have exactly one root, of tier cloud')
        for n in self.nodes:
            if n.tier not in TIERS:
                raise ValueError(f'Unknown tier {n.tier!r}')
            if n.mips <= 0 or n.memory_mb <= 0 or n.criticality <= 0:
                raise ValueError(f'Node {n.name}: MIPS, memory and criticality must be positive')
            if min(n.idle_w, n.busy_w, n.cost_per_gi) < 0:
                raise ValueError(f'Node {n.name}: power and cost must be nonnegative')
            if n.parent is not None:
                if n.parent not in self._nodes:
                    raise ValueError(f'Node {n.name}: unknown parent {n.parent}')
                if TIERS.index(self._nodes[n.parent].tier) <= TIERS.index(n.tier):
                    raise ValueError(f'Node {n.name}: parent must be in a higher tier')
                if not n.uplink_bytes_s or n.uplink_bytes_s <= 0 or n.uplink_latency_s is None or n.uplink_latency_s < 0:
                    raise ValueError(f'Node {n.name}: invalid uplink')
            unknown = set(n.resident) - set(self._models)
            if unknown:
                raise ValueError(f'Node {n.name}: unknown resident models {sorted(unknown)}')
            if sum(self._models[m].memory_mb for m in n.resident) > n.memory_mb:
                raise ValueError(f'Node {n.name}: resident models exceed memory')
        for m in self.models:
            if min(m.fixed_mi, m.mi_per_flow, m.load_s) < 0 or m.memory_mb <= 0:
                raise ValueError(f'Model {m.name}: invalid cost parameters')
            if not all(0 <= r <= 1 for r in m.recall.values()):
                raise ValueError(f'Model {m.name}: recall must lie in [0, 1]')
            if not 0 <= m.false_positive_rate <= 1:
                raise ValueError(f'Model {m.name}: false-positive rate must lie in [0, 1]')
        previous = None
        for t in self.tasks:
            if t.gateway not in self._nodes or self._nodes[t.gateway].tier != 'edge':
                raise ValueError(f'Task {t.id}: gateway must be an edge node')
            if previous is not None and (t.id <= previous.id or t.release_s < previous.release_s):
                raise ValueError('Tasks must have increasing ids and nondecreasing release times')
            if t.deadline_s <= 0 or t.n_flows <= 0 or t.input_bytes <= 0:
                raise ValueError(f'Task {t.id}: deadline, flows and input size must be positive')
            if not 0 <= t.attack_flows <= t.n_flows or not 0 <= t.prefilter_score <= 1:
                raise ValueError(f'Task {t.id}: invalid attack flows or prefilter score')
            if (t.label == BENIGN) != (t.attack_flows == 0):
                raise ValueError(f'Task {t.id}: label must be benign exactly when there are no attack flows')
            if t.label != BENIGN and any(t.label not in m.recall for m in self.models):
                raise ValueError(f'Task {t.id}: no recall defined for class {t.label}')
            previous = t
        return self

    # Serialization -------------------------------------------------------
    def to_dict(self):
        def clean(obj):
            return {k: (list(v) if isinstance(v, tuple) else _round(v)) for k, v in obj.__dict__.items()}
        return {'schema': self.schema, 'name': self.name, 'seed': self.seed,
                'horizon_s': _round(self.horizon_s), 'window_s': _round(self.window_s),
                'epoch_s': _round(self.epoch_s), 'meta': self.meta,
                'nodes': [clean(n) for n in self.nodes],
                'models': [{**clean(m), 'recall': {k: _round(v) for k, v in sorted(m.recall.items())}}
                           for m in self.models],
                'tasks': [clean(t) for t in self.tasks]}

    @classmethod
    def from_dict(cls, d):
        return cls(name=d['name'], seed=d['seed'], horizon_s=d['horizon_s'], window_s=d['window_s'],
                   epoch_s=d['epoch_s'], meta=d.get('meta', {}), schema=d['schema'],
                   nodes=[Node(**{**n, 'resident': tuple(n.get('resident', ()))}) for n in d['nodes']],
                   models=[Model(**m) for m in d['models']],
                   tasks=[Task(**t) for t in d['tasks']])

    def to_json(self):
        return json.dumps(self.to_dict(), indent=1, sort_keys=True) + '\n'

    def save(self, path):
        Path(path).write_text(self.to_json(), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8'))).validate()

    # Description ---------------------------------------------------------
    def summary(self):
        """Offered load and problem-size statistics used to sanity-check instances."""
        tasks = self.tasks
        epochs = max(1, int(round(self.horizon_s / self.epoch_s)))
        per_epoch = [0] * epochs
        for t in tasks:
            per_epoch[min(epochs - 1, int(t.release_s / self.epoch_s))] += 1
        capacity_mi = sum(n.mips for n in self.nodes) * self.horizon_s
        edge_fog_mi = sum(n.mips for n in self.nodes if n.tier != 'cloud') * self.horizon_s
        load = {m.name: sum(m.work_mi(t.n_flows) for t in tasks) for m in self.models}
        labels = {}
        for t in tasks:
            labels[t.label] = labels.get(t.label, 0) + 1
        eligible = sum(len(self.eligible(t)) for t in tasks) / max(1, len(tasks))
        return {
            'tasks': len(tasks), 'labels': dict(sorted(labels.items())),
            'short_deadline_tasks': sum(t.deadline_s < max((u.deadline_s for u in tasks), default=0) for t in tasks),
            'flows': sum(t.n_flows for t in tasks), 'attack_flows': sum(t.attack_flows for t in tasks),
            'tasks_per_epoch_mean': len(tasks) / epochs, 'tasks_per_epoch_peak': max(per_epoch),
            'eligible_nodes_per_task': eligible,
            'binary_variables_per_epoch_peak': int(max(per_epoch) * eligible * len(self.models)),
            'utilization_all_nodes': {k: v / capacity_mi for k, v in load.items()},
            'utilization_edge_fog_only': {k: v / edge_fog_mi for k, v in load.items()},
        }
