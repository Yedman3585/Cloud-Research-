"""Epoch QUBO for the smokescreen scenario (``smokescreen.py``).

At an epoch, every released task i chooses one option o = (node, model, fraction f,
extraction site) or the drop option; x_o = 1 if chosen, exactly one per task.

Energy (lower is better), built from scheduler-visible quantities only:

* value: ``-V_o`` with ``V_o = c_i (1 - (1 - f r_k)^(a_i))``, where a_i is the attack
  flow count estimated from the pre-filter score, but at least ``min_attack`` (a
  low-rate attack hides below the score's noise, so every task is assumed to possibly
  contain one of that size), r_k the model's mean recall and c_i
  the gateway criticality, shared equally among the chunks of a flood window (a window
  counts once, however many tasks it was cut into); minus a small false-alert cost on the estimated benign
  flows; the drop option has value 0;
* contention, exact for processor sharing of jobs that start together: on every CPU,
  ``gamma / MIPS * [sum_o w_o a_o x_o + sum_{o<p} (w_o + w_p) min(a_o, a_p) x_o x_p]``
  with a_o the option's work on that CPU (extraction at the gateway, raw-packet
  extraction plus inference at the fog or cloud), plus linear terms
  ``(w_o + w_j) min(a_o, r_j)`` for every job j already there (remaining work r_j); on
  every uplink the FIFO analogue with bytes, plus the current wait. Weights are
  ``1 / deadline`` of the task, ``1 / (time to deadline)`` of a job already running, so
  the term is a sum of finish times normalised by the deadline budgets.

The pairwise terms are not stored: the annealer evaluates them per resource from the
current assignment (``EpochQubo.delta``), which keeps an epoch with thousands of
options cheap; ``to_qubo`` materialises the matrix for small epochs.

The decoded assignment is checked task by task against the processor-sharing
estimator (own deadline, no committed task pushed past its deadline); a task that
fails takes the lowest-energy option that passes, or is dropped.
"""
import math
import random
import time

from .smokescreen import DROP, TOL, Estimator, SPolicy, ValueGreedy


class EpochQubo:
    def __init__(self, tasks, options, linear, loads, gamma):
        self.tasks = tasks
        self.options = options  # per task: [Option]
        self.linear = linear  # per task: [float]
        self.loads = loads  # per task: [ {resource: (amount / rate, weight)} ]
        self.gamma = gamma
        self.on = {}  # resource -> {task index: (amount / rate, weight)}

    def _pair(self, a, b):
        return self.gamma * (a[1] + b[1]) * min(a[0], b[0])

    def place(self, i, k):
        for res, load in self.loads[i][k].items():
            self.on.setdefault(res, {})[i] = load

    def remove(self, i, k):
        for res in self.loads[i][k]:
            self.on[res].pop(i, None)

    def delta(self, i, old, new):
        d = self.linear[i][new] - self.linear[i][old]
        for res, load in self.loads[i][old].items():
            for j, other in self.on.get(res, {}).items():
                if j != i:
                    d -= self._pair(load, other)
        for res, load in self.loads[i][new].items():
            for j, other in self.on.get(res, {}).items():
                if j != i:
                    d += self._pair(load, other)
        return d

    def energy(self, choice):
        e = sum(self.linear[i][k] for i, k in enumerate(choice))
        on = {}
        for i, k in enumerate(choice):
            for res, load in self.loads[i][k].items():
                for other in on.get(res, []):
                    e += self._pair(load, other)
                on.setdefault(res, []).append(load)
        return e

    def to_qubo(self, penalty):
        """Plain QUBO {(u, v): coef} over all option bits and an offset, with the
        one-hot constraint as penalty * (sum - 1)^2 per task."""
        index, q = {}, {}
        for i, opts in enumerate(self.options):
            for k in range(len(opts)):
                u = index[(i, k)] = len(index)
                q[(u, u)] = self.linear[i][k] - penalty
                for k2 in range(k):
                    q[(index[(i, k2)], u)] = 2 * penalty
        keys = list(index)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                (i, k), (j, l) = keys[a], keys[b]
                if i == j:
                    continue
                c = sum(self._pair(load, self.loads[j][l][res])
                        for res, load in self.loads[i][k].items() if res in self.loads[j][l])
                if c:
                    q[(index[keys[a]], index[keys[b]])] = q.get((index[keys[a]], index[keys[b]]), 0.0) + c
        return q, penalty * len(self.options)

    def anneal(self, start, budget_s, seed):
        rng = random.Random(seed)
        choice = list(start)
        self.on = {}
        for i, k in enumerate(choice):
            self.place(i, k)
        movable = [i for i, opts in enumerate(self.options) if len(opts) > 1]
        e = self.energy(choice)
        best, best_e = list(choice), e
        if not movable:
            return best, best_e
        scale = max(1e-9, max(abs(v) for row in self.linear for v in row))
        t0, t1 = 0.3 * scale, 1e-3 * scale
        started = time.perf_counter()
        deadline = started + budget_s
        k = 0
        while True:
            if (k & 31) == 0:
                now = time.perf_counter()
                if now > deadline:
                    break
                frac = (deadline - now) / budget_s
                temp = t1 * (t0 / t1) ** frac
            k += 1
            i = movable[int(rng.random() * len(movable))]
            old = choice[i]
            new = int(rng.random() * len(self.options[i]))
            if new == old:
                continue
            d = self.delta(i, old, new)
            if d <= 0 or rng.random() < math.exp(-d / temp):
                self.remove(i, old)
                self.place(i, new)
                choice[i] = new
                e += d
                if e < best_e - 1e-12:
                    best, best_e = list(choice), e
        return best, best_e


class QuboSmoke(SPolicy):
    """Epoch QUBO with processor-sharing contention, annealed within a time budget,
    checked and repaired against the estimator."""
    name = 'qubo-smoke'

    def __init__(self, gamma=1.0, fa_weight=1e-4, slack=0.1, budget_s=0.05, seed=0, min_attack=20.0):
        self.min_attack = min_attack
        self.gamma, self.fa_weight, self.slack, self.budget_s, self.seed = gamma, fa_weight, slack, budget_s, seed

    def reset(self, sc):
        super().reset(sc)
        self.epoch = 0
        self.stats = {'epochs': 0, 'tasks': 0, 'options': 0, 'repaired': 0, 'decision_s': 0.0}

    def value(self, t, o, share=1.0):
        if o.node is None:
            return 0.0
        a = max(self.min_attack, t.estimated_attack_flows)
        p = 1.0 - (1.0 - o.f * self.recall[o.model]) ** a
        fa = o.f * t.n_flows * (1.0 - t.prefilter_score) * self.sc.model(o.model).false_positive_rate
        return share * t.criticality * p - self.fa_weight * fa

    def build(self, est, tasks, now):
        sc = self.sc
        jobs = {}  # resource -> [(remaining / rate, weight)]
        for node, active in est.active.items():
            rate = sc.node(node).mips
            for key, rem in active.items():
                w = 1.0 / max(0.05, est.deadline.get(key, now + 2.0) - now)
                jobs.setdefault(('cpu', node), []).append((rem / rate, w))
            for _, key, work in est.pending[node]:
                w = 1.0 / max(0.05, est.deadline.get(key, now + 2.0) - now)
                jobs.setdefault(('cpu', node), []).append((work / rate, w))
        options, linear, loads = [], [], []
        # A flood window counts once: its value is shared among its chunks.
        chunks = {}
        for t in tasks:
            if t.stream == 'flood':
                chunks[(t.gateway, t.release_s)] = chunks.get((t.gateway, t.release_s), 0) + 1
        for t in tasks:
            share = 1.0 / chunks[(t.gateway, t.release_s)] if t.stream == 'flood' else 1.0
            w = 1.0 / t.deadline_s
            row_o, row_l, row_r = [], [], []
            for o in sc.options(t):
                if o.node is not None and sc.standalone_s(t, o) > t.deadline_s * (1 - self.slack) + TOL:
                    continue
                load = {}
                lin = -self.value(t, o, share)
                for kind, res, amount in sc.stages(t, o):
                    spec = sc.node(res)
                    if kind == 'cpu':
                        key, a = ('cpu', res), amount / spec.mips
                        lin += self.gamma * w * a
                        for r, wj in jobs.get(key, []):
                            lin += self.gamma * (w + wj) * min(a, r)
                    else:
                        key, a = ('link', res), amount / spec.uplink_bytes_s
                        lin += self.gamma * w * (a + est.link_free[res] - now if est.link_free[res] > now else a)
                    prev = load.get(key)
                    load[key] = (a + (prev[0] if prev else 0.0), w)
                row_o.append(o)
                row_l.append(lin)
                row_r.append(load)
            options.append(row_o)
            linear.append(row_l)
            loads.append(row_r)
        return EpochQubo(tasks, options, linear, loads, self.gamma)

    def decide(self, engine, now, tasks):
        started = time.perf_counter()
        tasks = sorted(tasks, key=lambda t: (-t.weight, t.id))
        est = Estimator(engine, now)
        q = self.build(est, tasks, now)
        start = [min(range(len(row)), key=lambda k: row[k]) for row in q.linear]
        choice, _ = q.anneal(start, self.budget_s, self.seed + self.epoch)
        self.epoch += 1
        out = {}
        for i, t in enumerate(tasks):
            order = [choice[i]] + sorted(range(len(q.options[i])), key=lambda k: q.linear[i][k])
            chosen = DROP
            for k in order:
                o = q.options[i][k]
                if o.node is None:
                    chosen = DROP
                    break
                good, _ = self.meets(est, t, o, self.slack)
                if good:
                    chosen = o
                    break
            if chosen is not q.options[i][choice[i]]:
                self.stats['repaired'] += 1
            if chosen.node is not None:
                est.commit(t, chosen)
            out[t.id] = chosen
        self.stats['epochs'] += 1
        self.stats['tasks'] += len(tasks)
        self.stats['options'] += sum(len(r) for r in q.options)
        self.stats['decision_s'] += time.perf_counter() - started
        return out


class EpochPS:
    """Deadline-exact epoch model for annealing.

    Every chosen option o puts a load a_{o,r} (seconds of exclusive use) on each resource
    r it uses. Under processor sharing of jobs that start together, the time job o
    spends on CPU r is ``sum_p min(a_{o,r}, a_{p,r})`` over all jobs on r, itself
    included: the row of the QUBO's contention term. Uplinks use the same form
    (shortest-first FIFO). A job's predicted latency F_o is the sum over its stages, and
    it earns its value only if F_o fits its budget (time to its deadline). Energy:
    ``-sum_o V_o [F_o <= B_o] + kappa * sum_o max(0, F_o - B_o) / D_o``, over the new
    options and the jobs already committed (fixed, with their own budgets and values).
    """

    def __init__(self, options, values, stages, budgets, horizons, fixed, kappa):
        self.options, self.values, self.stages = options, values, stages  # per task, per option
        self.budgets, self.horizons = budgets, horizons  # per task
        self.fixed = fixed  # [(value, {res: load}, budget, horizon)] already committed
        self.kappa = kappa

    def _setup(self, choice):
        n = len(self.options)
        self.choice = list(choice)
        self.on = {}  # res -> {job: load}; jobs: task index or ('f', k)
        self.load = {}  # job -> {res: load}
        self.t = {}  # job -> {res: time on res}
        for k, (_, loads, _, _) in enumerate(self.fixed):
            self._add(('f', k), loads)
        for i in range(n):
            self._add(i, self.stages[i][choice[i]])

    def _add(self, job, loads):
        self.load[job] = loads
        self.t[job] = {}
        for res, a in loads.items():
            members = self.on.setdefault(res, {})
            total = a
            for other, b in members.items():
                m = min(a, b)
                total += m
                self.t[other][res] += m
            members[job] = a
            self.t[job][res] = total

    def _remove(self, job):
        for res, a in self.load[job].items():
            members = self.on[res]
            del members[job]
            for other, b in members.items():
                self.t[other][res] -= min(a, b)
        del self.load[job], self.t[job]

    def _job_cost(self, job):
        F = sum(self.t[job].values())
        if isinstance(job, tuple):
            value, _, B, D = self.fixed[job[1]]
        else:
            value, B, D = self.values[job][self.choice[job]], self.budgets[job], self.horizons[job]
        if not self.load[job]:
            return 0.0
        late = F - B
        return (-value if late <= TOL else 0.0) + (self.kappa * late / D if late > 0 else 0.0)

    def _affected(self, resources):
        jobs = set()
        for res in resources:
            jobs.update(self.on.get(res, {}))
        return jobs

    def energy(self):
        return sum(self._job_cost(j) for j in self.load)

    def move(self, i, new):
        """Apply the move and return the energy change."""
        old = self.choice[i]
        res = set(self.stages[i][old]) | set(self.stages[i][new])
        jobs = self._affected(res) | {i}
        before = sum(self._job_cost(j) for j in jobs if j in self.load)
        self._remove(i)
        self.choice[i] = new
        self._add(i, self.stages[i][new])
        jobs = self._affected(res) | {i}
        after = sum(self._job_cost(j) for j in jobs)
        return after - before

    def anneal(self, start, budget_s, seed, t0=0.5, t1=0.005):
        rng = random.Random(seed)
        self._setup(start)
        e = self.energy()
        best, best_e = list(self.choice), e
        movable = [i for i, opts in enumerate(self.options) if len(opts) > 1]
        if not movable:
            return best, best_e
        deadline = time.perf_counter() + budget_s
        k = 0
        temp = t0
        while True:
            if (k & 15) == 0:
                now = time.perf_counter()
                if now > deadline:
                    break
                temp = t1 * (t0 / t1) ** ((deadline - now) / budget_s)
            k += 1
            i = movable[int(rng.random() * len(movable))]
            old = self.choice[i]
            new = int(rng.random() * len(self.options[i]))
            if new == old:
                continue
            d = self.move(i, new)
            if d <= 0 or rng.random() < math.exp(-d / temp):
                e += d
                if e < best_e - 1e-12:
                    best, best_e = list(self.choice), e
            else:
                self.move(i, old)
        return best, best_e


def _pareto(options, values, loads):
    """Drop dominated options: o is dominated by p if p is worth at least as much and
    puts no more load on any resource (processor-sharing and FIFO delays are monotone
    in every load, so p is never worse for o's task or for any other job). The drop
    option (no load) is kept."""
    keep = []
    for a in range(len(options)):
        dominated = False
        for b in range(len(options)):
            if a == b or values[b] < values[a] - 1e-12:
                continue
            la, lb = loads[a], loads[b]
            if any(res not in la for res in lb):
                continue
            if all(lb.get(res, 0.0) <= x + 1e-12 for res, x in la.items()):
                strict = values[b] > values[a] + 1e-12 or any(lb.get(res, 0.0) < x - 1e-12 for res, x in la.items())
                if strict or b < a:
                    dominated = True
                    break
        if not dominated:
            keep.append(a)
    return [options[k] for k in keep], [values[k] for k in keep], [loads[k] for k in keep]


class PSAnneal(QuboSmoke):
    """Epoch placement by annealing the deadline-exact processor-sharing model
    (``EpochPS``), checked and repaired against the estimator."""
    name = 'ps-anneal'

    def __init__(self, kappa=2.0, fa_weight=1e-4, slack=0.1, budget_s=0.05, seed=0, min_attack=5.0,
                 flood_share=True, oracle_value=False, prune=True, t0=0.5, t1=0.005, fast=True, repair=True):
        super().__init__(gamma=0.0, fa_weight=fa_weight, slack=slack, budget_s=budget_s, seed=seed,
                         min_attack=min_attack)
        self.prune, self.t0, self.t1, self.fast, self.repair = prune, t0, t1, fast, repair
        self.kappa, self.flood_share, self.oracle_value = kappa, flood_share, oracle_value
        if oracle_value:
            self.name = 'ps-anneal-oracle-value'

    def reset(self, sc):
        super().reset(sc)
        self.counts = sc.class_counts()
        if self.fast:
            from .fast_anneal import FastAnnealer
            self.annealer = FastAnnealer()

    def value(self, t, o, share=1.0):
        if not self.oracle_value:
            return super().value(t, o, share)
        # Reference only: the task's exact contribution to the objective (uses ground
        # truth), scaled so that a low-rate attack task is worth about 1.
        if o.node is None:
            return 0.0
        det = self.sc.detection(t, o)
        scale = 300.0
        return scale * sum(det[c] / self.counts[c] for c, _ in t.attacks) - self.fa_weight * self.sc.false_alerts(t, o)

    def build_ps(self, est, tasks, now):
        sc = self.sc
        fixed = []
        for node, active in est.active.items():
            rate = sc.node(node).mips
            items = list(active.items()) + [(key, work) for _, key, work in est.pending[node]]
            for key, rem in items:
                dl = est.deadline.get(key)
                if dl is None or dl <= now:
                    continue
                fixed.append((1.0, {('cpu', node): rem / rate}, dl - now, dl - now))
        chunks = {}
        for t in tasks:
            if t.stream == 'flood':
                chunks[(t.gateway, t.release_s)] = chunks.get((t.gateway, t.release_s), 0) + 1
        options, values, stages, budgets, horizons = [], [], [], [], []
        for t in tasks:
            share = 1.0 / chunks[(t.gateway, t.release_s)] if (t.stream == 'flood' and self.flood_share) else 1.0
            row_o, row_v, row_s = [], [], []
            for o in sc.options(t):
                if o.node is not None and sc.standalone_s(t, o) > t.deadline_s * (1 - self.slack) + TOL:
                    continue
                loads = {}
                for kind, res, amount in sc.stages(t, o):
                    spec = sc.node(res)
                    if kind == 'cpu':
                        key, a = ('cpu', res), amount / spec.mips
                    else:
                        key = ('link', res)
                        a = amount / spec.uplink_bytes_s + spec.uplink_latency_s
                        if est.link_free[res] > now:
                            a += est.link_free[res] - now
                    loads[key] = loads.get(key, 0.0) + a
                row_o.append(o)
                row_v.append(self.value(t, o, share))
                row_s.append(loads)
            if self.prune:
                row_o, row_v, row_s = _pareto(row_o, row_v, row_s)
            options.append(row_o)
            values.append(row_v)
            stages.append(row_s)
            budget = (t.release_s + t.deadline_s * (1 - self.slack)) - now - est.delay
            budgets.append(budget)
            horizons.append(t.deadline_s)
        return EpochPS(options, values, stages, budgets, horizons, fixed, self.kappa)

    def decide(self, engine, now, tasks):
        started = time.perf_counter()
        tasks = sorted(tasks, key=lambda t: (-t.weight, t.id))
        est = Estimator(engine, now)
        m = self.build_ps(est, tasks, now)
        # Start: the most valuable option of every task that fits on its own.
        start = []
        for i, t in enumerate(tasks):
            fits = [k for k, s in enumerate(m.stages[i]) if sum(s.values()) <= m.budgets[i]] or [0]
            start.append(max(fits, key=lambda k: m.values[i][k]))
        if self.fast:
            choice, _ = self.annealer.anneal(m, start, self.budget_s, self.seed + self.epoch, self.t0, self.t1)
        else:
            choice, _ = m.anneal(start, self.budget_s, self.seed + self.epoch, self.t0, self.t1)
        self.epoch += 1
        out = {}
        for i, t in enumerate(tasks):
            order = [choice[i]] + sorted(range(len(m.options[i])), key=lambda k: -m.values[i][k])
            if not self.repair:
                order = order[:1]
            chosen = DROP if self.repair else m.options[i][choice[i]]
            for k in (order if self.repair else []):
                o = m.options[i][k]
                if o.node is None:
                    break
                good, _ = self.meets(est, t, o, self.slack)
                if good:
                    chosen = o
                    break
            if chosen is not m.options[i][choice[i]]:
                self.stats['repaired'] += 1
            if chosen.node is not None:
                est.commit(t, chosen)
            out[t.id] = chosen
        self.stats['epochs'] += 1
        self.stats['tasks'] += len(tasks)
        self.stats['options'] += sum(len(r) for r in m.options)
        self.stats['decision_s'] += time.perf_counter() - started
        return out


class AlertEvidence:
    """Scheduler-side evidence of low-rate attacks from the alerts already raised.

    For a gateway, over the residual-stream alerts of the last ``horizon_s`` seconds:
    A low-rate alerts observed, E the false alerts expected if every analysed flow had
    been benign (analysed flows x the model's false-positive rate), N the analysed flows
    x the model's mean recall. The estimated share of low-rate attack flows in the
    gateway's residual traffic is max(0, A - E) / N, and z = (A - E) / sqrt(E + 1).
    """

    def __init__(self, sc, horizon_s=40.0):
        self.sc, self.horizon_s = sc, horizon_s
        self.fpr = {m.name: m.false_positive_rate for m in sc.models}
        self.recall = {m.name: sum(m.recall.values()) / len(m.recall) for m in sc.models}

    def update(self, engine, now):
        """Incremental: add the alerts raised since the last call, drop the expired ones."""
        if not hasattr(self, 'acc') or getattr(self, '_engine', None) is not engine:
            self._engine, self._seen, self.acc, self._window = engine, 0, {}, {}
        from collections import deque
        alerts = engine.alerts
        while self._seen < len(alerts) and alerts[self._seen].time_s <= now:
            a = alerts[self._seen]
            self._seen += 1
            if a.stream != 'residual':
                continue
            item = (a.time_s, a.quiet_alerts, a.analysed_flows * self.fpr[a.model],
                    a.analysed_flows * self.recall[a.model])
            self._window.setdefault(a.gateway, deque()).append(item)
            s = self.acc.setdefault(a.gateway, [0.0, 0.0, 0.0])
            for k in range(3):
                s[k] += item[k + 1]
        for gw, q in self._window.items():
            s = self.acc[gw]
            while q and q[0][0] < now - self.horizon_s:
                item = q.popleft()
                for k in range(3):
                    s[k] -= item[k + 1]

    def share(self, gateway):
        A, E, N = self.acc.get(gateway, (0.0, 0.0, 0.0))
        return max(0.0, A - E) / N if N else 0.0

    def z(self, gateway):
        A, E, _ = self.acc.get(gateway, (0.0, 0.0, 0.0))
        return (A - E) / math.sqrt(E + 1.0)


class AlertPSAnneal(PSAnneal):
    """ps-anneal whose option values estimate the low-rate attack flows of a residual
    task from the gateway's recent alerts (``AlertEvidence``) instead of a fixed prior:
    a = max(floor, share x flows); flood tasks keep the pre-filter estimate."""
    name = 'ps-anneal-alerts'

    def __init__(self, horizon_s=40.0, floor=1.0, **kw):
        super().__init__(**kw)
        self.horizon_s, self.floor = horizon_s, floor

    def reset(self, sc):
        super().reset(sc)
        self.evidence = AlertEvidence(sc, self.horizon_s)

    def attack_estimate(self, t):
        if t.stream == 'flood':
            return max(self.min_attack, t.estimated_attack_flows)
        return max(self.floor, self.evidence.share(t.gateway) * t.n_flows)

    def value(self, t, o, share=1.0):
        if o.node is None:
            return 0.0
        a = self.attack_estimate(t)
        p = 1.0 - (1.0 - o.f * self.recall[o.model]) ** a
        fa = o.f * t.n_flows * (1.0 - t.prefilter_score) * self.sc.model(o.model).false_positive_rate
        return share * t.criticality * p - self.fa_weight * fa

    def decide(self, engine, now, tasks):
        self.evidence.update(engine, now)
        return super().decide(engine, now, tasks)


class ValueGreedyAlerts(ValueGreedy):
    """value-greedy with the same alert evidence as ps-anneal-alerts (fair baseline:
    same information, myopic one-task-at-a-time decisions)."""
    name = 'value-greedy-alerts'

    def __init__(self, horizon_s=40.0, floor=3.0, min_attack=5.0, **kw):
        super().__init__(**kw)
        self.horizon_s, self.floor, self.min_attack = horizon_s, floor, min_attack

    def reset(self, sc):
        super().reset(sc)
        self.evidence = AlertEvidence(sc, self.horizon_s)

    def value(self, t, o):
        if o.node is None:
            return 0.0
        if t.stream == 'flood':
            a = max(self.min_attack, t.estimated_attack_flows)
        else:
            a = max(self.floor, self.evidence.share(t.gateway) * t.n_flows)
        p = 1.0 - (1.0 - o.f * self.recall[o.model]) ** a
        fa = o.f * t.n_flows * (1 - t.prefilter_score) * self.sc.model(o.model).false_positive_rate
        return t.criticality * p - 1e-4 * fa

    def decide(self, engine, now, tasks):
        self.evidence.update(engine, now)
        return super().decide(engine, now, tasks)
