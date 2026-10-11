"""Epoch QUBO for IDS-PLACE built on the simulator's resource model.

At an epoch, the released tasks are placed jointly. Each task i has options
o = (node, model); binary x_o = 1 if option o is chosen, exactly one per task.

Objective (lower is better):

* false alerts: ``lam * benign_i * fpr_k`` per option, with benign flows estimated from
  the pre-filter score (scheduler-visible);
* CPU contention, exact for processor sharing: for tasks that start together on one
  node with works a and weights w, the weighted sum of finish times is
  ``sum_i w_i a_i + sum_{i<j} (w_i + w_j) min(a_i, a_j)`` (divided by the node's MIPS).
  Tasks already on the node (remaining work r_j, weight w_j) add the linear terms
  ``(w_o + w_j) min(a_o, r_j)``. Weights are 1 / deadline, so the term is a sum of
  normalised latencies;
* uplinks: the current wait of every link on the option's path (linear) and, for two
  new tasks crossing the same link, ``(w_o + w_p) min(b_o, b_p) / bandwidth`` (FIFO in
  shortest-first order), plus the transfer itself;
* one option per task: a penalty ``P (sum_o x_o - 1)^2`` in the QUBO form; the annealer
  instead moves between options of a task, which keeps the constraint exactly.

Options that cannot meet their deadline even with the present load (``PSModel``
estimate; models other than the fastest must leave ``slack`` of the deadline) are
dropped before the QUBO is built; the fastest model at the gateway is always kept.
The decoded assignment is checked task by task against ``PSModel`` (own deadline, no
task already on the node pushed past its deadline); a task that fails the check takes
the ``ps-greedy`` choice instead.
"""
import math
import random
import time

from .policies import Policy, _models_by_quality
from .ps_model import PSModel

TOLERANCE_S = 1e-5


class EpochQubo:
    """Options, linear and pairwise coefficients of one epoch."""

    def __init__(self, options, linear, pairs, by_task):
        self.options = options  # [(task, node, model)]
        self.linear = linear  # [float]
        self.pairs = pairs  # {(o, p): coef} with o < p, o and p of different tasks
        self.by_task = by_task  # [[option indices]] per task
        self.neighbours = [[] for _ in options]
        for (o, p), c in pairs.items():
            self.neighbours[o].append((p, c))
            self.neighbours[p].append((o, c))

    def energy(self, choice):
        """Energy of one option index per task."""
        chosen = set(choice)
        e = sum(self.linear[o] for o in choice)
        e += sum(c for (o, p), c in self.pairs.items() if o in chosen and p in chosen)
        return e

    def to_qubo(self, penalty):
        """Plain QUBO {(u, v): coef} over all option bits (u <= v) and an offset, with
        the one-hot constraint as penalty * (sum - 1)^2 per task."""
        q = {(o, o): c for o, c in enumerate(self.linear)}
        for key, c in self.pairs.items():
            q[key] = q.get(key, 0.0) + c
        offset = 0.0
        for group in self.by_task:
            for a in group:
                q[(a, a)] = q.get((a, a), 0.0) - penalty  # P x^2 - 2P x = -P x
                for b in group:
                    if a < b:
                        q[(a, b)] = q.get((a, b), 0.0) + 2 * penalty
            offset += penalty
        return q, offset

    def anneal(self, start, budget_s, seed, t0=None, t1=None):
        """Simulated annealing over one-hot-preserving moves (change one task's option).
        Returns the best choice found within ``budget_s`` seconds of wall time."""
        rng = random.Random(seed)
        choice = list(start)
        n = len(choice)
        movable = [i for i in range(n) if len(self.by_task[i]) > 1]
        if not movable:
            return choice, self.energy(choice)
        chosen = [0] * len(self.options)
        for o in choice:
            chosen[o] = 1
        e = self.energy(choice)
        best, best_e = list(choice), e
        scale = max(1e-9, max(abs(v) for v in self.linear))
        t0 = t0 or scale
        t1 = t1 or scale * 1e-3
        deadline = time.perf_counter() + budget_s
        steps, k = 0, 0
        while True:
            if (k & 63) == 0 and time.perf_counter() > deadline:
                break
            k += 1
            frac = min(1.0, (deadline - time.perf_counter()) / budget_s) if budget_s > 0 else 0.0
            temp = t1 + (t0 - t1) * max(0.0, frac)
            i = movable[int(rng.random() * len(movable))]
            group = self.by_task[i]
            old = choice[i]
            new = group[int(rng.random() * len(group))]
            if new == old:
                continue
            delta = self.linear[new] - self.linear[old]
            for p, c in self.neighbours[old]:
                if chosen[p]:
                    delta -= c
            for p, c in self.neighbours[new]:
                if chosen[p] and p != old:
                    delta += c
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                chosen[old], chosen[new] = 0, 1
                choice[i] = new
                e += delta
                steps += 1
                if e < best_e - 1e-12:
                    best, best_e = list(choice), e
        return best, best_e


def build_epoch(model, instance, tasks, now, lam, gamma, slack, fastest, urgency=0.0):
    """Build the epoch QUBO from the current ``PSModel`` state.

    Weights of the latency terms: ``1 / deadline`` when ``urgency`` is 0; otherwise
    ``1 / (margin + urgency)``, where the margin is how far the task's predicted finish
    is from its deadline. Latency then costs little where deadlines are comfortable and
    a lot where they are tight, which approximates the deadline-miss objective; tasks
    already predicted to miss get weight 0 (further delay changes nothing).
    """
    def weight(deadline_abs, deadline_s, finish):
        if not urgency:
            return 1.0 / deadline_s
        margin = deadline_abs - finish
        return 0.0 if margin < 0 else 1.0 / (margin + urgency)

    options, linear, by_task = [], [], []
    deadline_w = {}
    for i, t in enumerate(tasks):
        group = []
        for node in instance.eligible(t):
            for m in instance.models:
                own, work, _, _ = model.estimate(t, node, m.name, now)
                budget = t.deadline_s * (1.0 if m.name == fastest else 1.0 - slack)
                keep = own <= t.release_s + budget + TOLERANCE_S or (node == t.gateway and m.name == fastest)
                if not keep:
                    continue
                group.append(len(options))
                options.append((t, node, m.name, work, weight(t.release_s + t.deadline_s, t.deadline_s, own)))
        by_task.append(group)
    # Linear terms.
    jobs = {node.name: model.jobs(node.name, now) for node in instance.nodes}
    finish = {node.name: model.finishes(node.name, now) for node in instance.nodes} if urgency else {}
    job_w = {}
    for name, items in jobs.items():
        for task_id, _ in items:
            tj = instance.tasks[task_id]
            job_w[task_id] = weight(tj.release_s + tj.deadline_s, tj.deadline_s,
                                    finish.get(name, {}).get(task_id, now))
    for o, (t, node, mname, work, w) in enumerate(options):
        m, spec = instance.model(mname), instance.node(node)
        fa = lam * t.estimated_benign_flows * m.false_positive_rate
        cpu = w * work
        for task_id, remaining in jobs[node]:
            wj = job_w[task_id]
            cpu += (w + wj) * min(work, remaining)
        transfer = 0.0
        for link in instance.path_links(t.gateway, node):
            lspec = instance.node(link)
            transfer += model.link_wait(link, now) + t.input_bytes / lspec.uplink_bytes_s + lspec.uplink_latency_s
        linear.append(fa + gamma * (cpu / spec.mips + w * transfer))
    # Pairwise terms between options of different tasks.
    pairs = {}
    by_node, by_link = {}, {}
    for o, (t, node, mname, work, w) in enumerate(options):
        by_node.setdefault(node, []).append(o)
        for link in instance.path_links(t.gateway, node):
            by_link.setdefault(link, []).append(o)
    task_of = {}
    for i, group in enumerate(by_task):
        for o in group:
            task_of[o] = i
    for node, group in by_node.items():
        mips = instance.node(node).mips
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                o, p = group[a], group[b]
                if task_of[o] == task_of[p]:
                    continue
                c = (options[o][4] + options[p][4]) * min(options[o][3], options[p][3]) / mips
                pairs[(o, p)] = pairs.get((o, p), 0.0) + gamma * c
    for link, group in by_link.items():
        bw = instance.node(link).uplink_bytes_s
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                o, p = group[a], group[b]
                if task_of[o] == task_of[p]:
                    continue
                to, tp = options[o][0], options[p][0]
                c = (options[o][4] + options[p][4]) * min(to.input_bytes, tp.input_bytes) / bw
                key = (min(o, p), max(o, p))
                pairs[key] = pairs.get(key, 0.0) + gamma * c
    return EpochQubo([(t, node, m) for t, node, m, _, _ in options], linear, pairs, by_task)


class QuboPlace(Policy):
    """Joint epoch placement by annealing the epoch QUBO, checked against PSModel."""

    def __init__(self, lam=1.0, gamma=10.0, slack=0.1, budget_s=0.02, seed=0, urgency=0.0):
        self.lam, self.gamma, self.slack, self.budget_s, self.seed = lam, gamma, slack, budget_s, seed
        self.urgency = urgency
        self.name = 'qubo-ps'
        self.stats = {'epochs': 0, 'repaired': 0, 'tasks': 0, 'variables': 0}

    def reset(self, instance):
        super().reset(instance)
        self.model = PSModel(instance)
        self.quality = [m.name for m in _models_by_quality(instance)]
        self.fastest = min(instance.models, key=lambda m: m.mi_per_flow).name
        self.predicted = {}
        self.epoch = 0
        self.stats = {'epochs': 0, 'repaired': 0, 'tasks': 0, 'variables': 0}

    def _deadline(self, task_id):
        t = self.instance.tasks[task_id]
        return t.release_s + t.deadline_s + TOLERANCE_S

    def _acceptable(self, task, node, model, now):
        before = self.model.finishes(node, now)
        own, _, _, after = self.model.estimate(task, node, model, now)
        budget = task.deadline_s * (1.0 if model == self.fastest else 1.0 - self.slack)
        meets = own <= task.release_s + budget + TOLERANCE_S
        broken = sum(1 for k, f in after.items() if before.get(k, f) <= self._deadline(k) < f)
        return meets and not broken, (broken + (not meets), own)

    def _greedy_choice(self, task, now):
        best, fallback = None, None
        for node in self.instance.eligible(task):
            for rank, model in enumerate(self.quality):
                ok, key = self._acceptable(task, node, model, now)
                if ok and (best is None or (rank, key[1]) < best[0]):
                    best = ((rank, key[1]), node, model)
                if fallback is None or key < fallback[0]:
                    fallback = (key, node, model)
        _, node, model = best or fallback
        return node, model

    def decide(self, snapshot):
        self.model.sync(snapshot)
        now = snapshot.t
        tasks = sorted(snapshot.tasks, key=lambda t: (-t.weight, t.id))
        q = build_epoch(self.model, self.instance, tasks, now, self.lam, self.gamma, self.slack, self.fastest,
                        self.urgency)
        # Start from the cheapest-alert option of every task that survived pruning.
        start = [min(group, key=lambda o: q.linear[o]) for group in q.by_task]
        choice, _ = q.anneal(start, self.budget_s, self.seed + self.epoch)
        self.epoch += 1
        self.stats['epochs'] += 1
        self.stats['tasks'] += len(tasks)
        self.stats['variables'] += len(q.options)
        out = []
        for i, t in enumerate(tasks):
            _, node, model = q.options[choice[i]]
            ok, _ = self._acceptable(t, node, model, now)
            if not ok:
                node, model = self._greedy_choice(t, now)
                self.stats['repaired'] += 1
            self.predicted[t.id] = self.model.commit(t, node, model, now)
            out.append((t.id, node, model))
        return out
