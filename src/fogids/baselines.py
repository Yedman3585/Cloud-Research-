"""Baselines from the literature, adapted to the smokescreen scenario.

``SplitPolicy`` follows SA-TOCA (Sudyana et al., "ML-Based Intrusion Detection as a
Service: Traffic Split Offloading and Cost Allocation in a Multi-Tier Architecture",
IEEE TSC 2025): the share of IDS traffic processed at each tier is fixed offline and
chosen by simulated annealing; at run time traffic is split by those shares, with no
feedback from the system state. Here a traffic class is (stream, deadline class); for
each class the offline parameters are a route split over

* ``local``      features and inference at the gateway,
* ``fog``        features at the gateway, inference at the fog node,
* ``fog-raw``    sampled raw packets to the fog node, features and inference there,
* ``cloud``      features at the gateway, inference in the cloud,
* ``drop``       not analysed,

plus one model and one sampling fraction per class. Tasks of a class are routed by
smooth weighted round robin over the shares (deterministic). ``tune_splits`` anneals
the parameters on training seeds with **our** objective (macro missed detection +
false alerts), which favours the baseline: the original optimises a cost model.
"""
import json
import math
import random

from .smokescreen import DROP, Option, SPolicy

ROUTES = ('local', 'fog', 'fog-raw', 'cloud', 'drop')
CLASSES = ('flood/high', 'flood/low', 'residual/high', 'residual/low')
SHARE_STEPS = 10  # shares are multiples of 1/10


def _class(sc, task):
    high = task.deadline_s <= sc.config['deadlines']['high_s']
    return f"{task.stream}/{'high' if high else 'low'}"


def default_params(sc):
    """Everything local with the fastest model, no sampling (the no-shed start)."""
    fastest = min(sc.models, key=lambda m: m.mi_per_flow).name
    return {c: {'shares': {'local': SHARE_STEPS, 'fog': 0, 'fog-raw': 0, 'cloud': 0, 'drop': 0},
                'model': fastest, 'f': 1.0} for c in CLASSES}


class SplitPolicy(SPolicy):
    name = 'sa-toca'

    def __init__(self, params=None, path=None):
        if path is not None:
            params = json.loads(open(path, encoding='utf-8').read())['params']
        self.params = params

    def reset(self, sc):
        super().reset(sc)
        if self.params is None:
            self.params = default_params(sc)
        self.credit = {}  # (gateway, class) -> {route: credit} for smooth weighted round robin

    def route(self, gateway, cls):
        shares = self.params[cls]['shares']
        credit = self.credit.setdefault((gateway, cls), {r: 0.0 for r in ROUTES})
        total = sum(shares.values())
        for r in ROUTES:
            credit[r] += shares[r]
        best = max(ROUTES, key=lambda r: (credit[r], -ROUTES.index(r)))
        credit[best] -= total
        return best

    def decide(self, engine, now, tasks):
        out = {}
        for t in sorted(tasks, key=lambda t: t.id):
            cls = _class(self.sc, t)
            p = self.params[cls]
            r = self.route(t.gateway, cls)
            if r == 'drop':
                out[t.id] = DROP
                continue
            fog = self.sc.node(t.gateway).parent
            node = {'local': t.gateway, 'fog': fog, 'fog-raw': fog, 'cloud': 'cloud'}[r]
            out[t.id] = Option(node, p['model'], p['f'], 'node' if r == 'fog-raw' else 'gw')
        return out


def _neighbour(params, rng, models, fractions):
    new = json.loads(json.dumps(params))
    cls = rng.choice(CLASSES)
    p = new[cls]
    kind = rng.random()
    if kind < 0.6:  # move one to five share steps between two routes
        src = rng.choice([r for r in ROUTES if p['shares'][r] > 0])
        dst = rng.choice([r for r in ROUTES if r != src])
        k = rng.randint(1, min(5, p['shares'][src]))
        p['shares'][src] -= k
        p['shares'][dst] += k
    elif kind < 0.8:
        p['f'] = rng.choice([f for f in fractions if f != p['f']])
    else:
        p['model'] = rng.choice([m for m in models if m != p['model']] or models)
    return new


def tune_splits(configs, iterations=120, seed=0, log=print):
    """Simulated annealing over the split parameters; ``configs`` are scenario
    configurations (one per training seed). Returns (best params, best objective,
    history)."""
    from .smokescreen import Engine, generate
    scenarios = [generate(c) for c in configs]
    models = [m.name for m in scenarios[0].models]
    fractions = list(scenarios[0].fractions)
    rng = random.Random(seed)

    def score(params):
        return sum(Engine(sc).run(SplitPolicy(params))['objective'] for sc in scenarios) / len(scenarios)

    current = default_params(scenarios[0])
    e = score(current)
    best, best_e = current, e
    history = [(0, e, best_e)]
    t0, t1 = 0.01, 0.0002
    for k in range(1, iterations + 1):
        temp = t0 * (t1 / t0) ** (k / iterations)
        cand = _neighbour(current, rng, models, fractions)
        ce = score(cand)
        if ce <= e or rng.random() < math.exp(-(ce - e) / temp):
            current, e = cand, ce
            if e < best_e:
                best, best_e = current, e
        history.append((k, ce, best_e))
        log(f'iteration {k}: candidate {ce:.4f}, current {e:.4f}, best {best_e:.4f}')
    return best, best_e, history


class QIEAPolicy(SPolicy):
    """FOG-QIEA-style per-epoch placement: a quantum-inspired evolutionary algorithm
    over the epoch's options with a standard additive load model.

    Each task carries a probability vector over its options (the multi-valued analogue
    of Q-bit amplitudes); every generation observes ``population`` solutions from them,
    scores each, and rotates the vectors toward the best solution found so far
    (p <- (1 - rate) p + rate * onehot(best)). The score uses the same option values as
    ``ps-anneal-alerts`` (alert evidence) and the same deadline rule, but the usual
    additive contention model: a task's predicted latency on a resource is the backlog
    already there plus the full load of every other task chosen on it (no processor
    sharing). ``model='ps'`` scores with the processor-sharing epoch model of
    ps-anneal instead (an optimizer-only swap, as an ablation). Runs within the same
    per-epoch time budget; the result is checked and
    repaired against the estimator like the other policies.
    """
    name = 'fog-qiea'

    def __init__(self, population=12, rate=0.2, budget_s=0.05, kappa=2.0, slack=0.1, seed=0, floor=3.0,
                 model='additive'):
        self.model = model
        if model == 'ps':
            self.name = 'qiea-ps'
        self.population, self.rate, self.budget_s = population, rate, budget_s
        self.kappa, self.slack, self.seed, self.floor = kappa, slack, seed, floor

    def reset(self, sc):
        from .smokescreen_qubo import AlertPSAnneal
        super().reset(sc)
        # Reuse the model construction and alert-based values of ps-anneal-alerts.
        self.inner = AlertPSAnneal(floor=self.floor, slack=self.slack, kappa=self.kappa, fast=False)
        self.inner.reset(sc)
        self.rng = random.Random(self.seed)
        self.stats = {'epochs': 0, 'tasks': 0, 'decision_s': 0.0, 'generations': 0}

    def _score(self, m, backlog, choice):
        if self.model == 'ps':  # the processor-sharing epoch model of ps-anneal
            m._setup(choice)
            return m.energy()
        load = dict(backlog)
        for i, k in enumerate(choice):
            for res, a in m.stages[i][k].items():
                load[res] = load.get(res, 0.0) + a
        e = 0.0
        for i, k in enumerate(choice):
            stages = m.stages[i][k]
            if not stages:
                continue
            F = sum(load[res] for res in stages)
            late = F - m.budgets[i]
            if late <= 1e-9:
                e -= m.values[i][k]
            else:
                e += self.kappa * late / m.horizons[i]
        return e

    def decide(self, engine, now, tasks):
        import time
        from .smokescreen import Estimator
        started = time.perf_counter()
        tasks = sorted(tasks, key=lambda t: (-t.weight, t.id))
        self.inner.evidence.update(engine, now)
        est = Estimator(engine, now)
        m = self.inner.build_ps(est, tasks, now)
        backlog = {}
        for _, loads, _, _ in m.fixed:
            for res, a in loads.items():
                backlog[res] = backlog.get(res, 0.0) + a
        probs = [[1.0 / len(row)] * len(row) for row in m.options]
        best, best_e = None, float('inf')
        deadline = started + self.budget_s
        gens = 0
        while time.perf_counter() < deadline or best is None:
            gens += 1
            for _ in range(self.population):
                choice = [self.rng.choices(range(len(p)), weights=p)[0] for p in probs]
                e = self._score(m, backlog, choice)
                if e < best_e:
                    best, best_e = choice, e
            for i, p in enumerate(probs):
                for k in range(len(p)):
                    p[k] = (1 - self.rate) * p[k] + (self.rate if k == best[i] else 0.0)
        out = {}
        for i, t in enumerate(tasks):
            order = [best[i]] + sorted(range(len(m.options[i])), key=lambda k: -m.values[i][k])
            chosen = DROP
            for k in order:
                o = m.options[i][k]
                if o.node is None:
                    break
                good, _ = self.meets(est, t, o, self.slack)
                if good:
                    chosen = o
                    break
            if chosen.node is not None:
                est.commit(t, chosen)
            out[t.id] = chosen
        self.stats['epochs'] += 1
        self.stats['tasks'] += len(tasks)
        self.stats['generations'] += gens
        self.stats['decision_s'] += time.perf_counter() - started
        return out
