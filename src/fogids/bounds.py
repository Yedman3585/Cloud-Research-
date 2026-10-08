"""Offline fluid lower bound for IDS-PLACE (milestone M5).

How good can any policy be on an instance? This module solves a linear-programming
relaxation over the whole horizon, with full knowledge of the future:

* variable x[i, m, k] in [0, 1]: fraction of task i run on node m with model k,
  only for options whose queue-free transfer plus exclusive compute time fits the
  deadline; variable z[i] in [0, 1]: fraction of task i given up (a deadline miss);
* every task is fully placed or given up: sum x[i, ., .] + z[i] = 1;
* capacity: for every node (CPU, MI) and every uplink (bytes) and every checked time
  interval [a, b] (a grid of lengths, every task window, and windows grown from each
  window start), the work of the tasks that must be served entirely inside the
  interval is at most capacity x L. A task's CPU window on node m is
  [release + transfer, release + deadline]; its window on a link is
  [release, release + deadline - compute].

These are necessary conditions for meeting the deadlines, so the optimum is a valid
lower bound on any policy's cost; it is optimistic because work is fluid, queues are
ignored and only a subset of intervals is checked. The objective is lexicographic
through weights: deadline misses first, then expected false alerts on benign flows,
then expected missed attack flows.
"""
import math

MISS_WEIGHT = 1e6  # per task given up; dominates every quality term


def _intervals(starts, lengths):
    return [(a, a + L) for L in lengths for a in starts]


def fluid_bound(instance, lengths=(0.3, 0.5, 1.0, 2.0, 4.0), grid_s=None, time_limit_s=600):
    """Return a dict with the bound on deadline misses, false alerts and missed attacks."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    p = instance
    grid = grid_s or p.epoch_s
    tasks = p.tasks
    options = []  # (task index, node, model, cost, work_mi, cpu_window, link_windows)
    for i, t in enumerate(tasks):
        benign = t.n_flows - t.attack_flows
        for node in p.eligible(t):
            spec = p.node(node)
            transfer = p.transfer_s(t, node)
            for m in p.models:
                work = m.work_mi(t.n_flows)
                compute = work / spec.mips
                if transfer + compute > t.deadline_s + 1e-9:
                    continue
                recall = m.recall.get(t.label, 0.0) if t.attack_flows else 0.0
                cost = benign * m.false_positive_rate + 1e-3 * t.attack_flows * (1 - recall)
                cpu = (t.release_s + transfer, t.release_s + t.deadline_s)
                links = []
                t0 = t.release_s
                for link in p.path_links(t.gateway, node):
                    # The input must cross every link before the compute window ends.
                    links.append((link, t0, t.release_s + t.deadline_s - compute))
                options.append((i, node, m.name, cost, work, cpu, links))
    n_x, n_z = len(options), len(tasks)
    c = np.concatenate([np.array([o[3] for o in options]), np.full(n_z, MISS_WEIGHT)])

    # Equality: each task placed once (or given up).
    rows = [o[0] for o in options] + list(range(n_z))
    cols = list(range(n_x)) + [n_x + i for i in range(n_z)]
    a_eq = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_z, n_x + n_z)).tocsr()
    b_eq = np.ones(n_z)

    end = max((t.release_s + t.deadline_s for t in tasks), default=0.0)
    starts = [k * grid for k in range(int(math.ceil(end / grid)) + 1)]
    ub_rows, ub_cols, ub_vals, b_ub = [], [], [], []
    resources = {}  # (kind, name) -> list of (window_start, window_end, amount, column)
    for col, (i, node, model, cost, work, cpu, links) in enumerate(options):
        resources.setdefault(('cpu', node), []).append((cpu[0], cpu[1], work, col))
        for link, a, b in links:
            resources.setdefault(('link', link), []).append((a, b, tasks[i].input_bytes, col))
    for (kind, name), items in resources.items():
        spec = p.node(name)
        rate = spec.mips if kind == 'cpu' else spec.uplink_bytes_s
        items.sort()
        window_starts = np.array([w[0] for w in items])
        # Intervals: the grid, plus every task window itself and windows grown from each
        # window start, so that tasks released together (chunks of one window) are checked.
        own = {(round(w[0], 9), round(w[1], 9)) for w in items}
        grown = {(a, round(a + L, 9)) for a, _ in own for L in lengths}
        for a, b in sorted(set(_intervals(starts, lengths)) | own | grown):
            lo = np.searchsorted(window_starts, a - 1e-9)
            members = []
            for s, e, amount, col in items[lo:]:
                if s >= b:
                    break
                if e <= b + 1e-9:
                    members.append((amount, col))
            if not members:
                continue
            capacity = rate * (b - a)
            if sum(amount for amount, _ in members) <= capacity:
                continue  # cannot bind
            row = len(b_ub)
            for amount, col in members:
                ub_rows.append(row)
                ub_cols.append(col)
                ub_vals.append(amount)
            b_ub.append(capacity)
    a_ub = coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(len(b_ub), n_x + n_z)).tocsr() if b_ub else None
    res = linprog(c, A_ub=a_ub, b_ub=np.array(b_ub) if b_ub else None, A_eq=a_eq, b_eq=b_eq,
                  bounds=(0, 1), method='highs', options={'time_limit': time_limit_s})
    if res.status != 0:
        raise RuntimeError(f'LP not solved: {res.message}')
    x, z = res.x[:n_x], res.x[n_x:]
    benign = sum(t.n_flows - t.attack_flows for t in tasks)
    attack = sum(t.attack_flows for t in tasks)
    false_alerts = sum(x[col] * (tasks[o[0]].n_flows - tasks[o[0]].attack_flows) * p.model(o[2]).false_positive_rate
                       for col, o in enumerate(options))
    missed = 0.0
    for col, o in enumerate(options):
        t = tasks[o[0]]
        if t.attack_flows:
            missed += x[col] * t.attack_flows * (1 - p.model(o[2]).recall.get(t.label, 0.0))
    missed += sum(z[i] * tasks[i].attack_flows for i in range(n_z))
    plan = {}  # task id -> (node, model) with the largest fraction; given-up tasks keep none
    for col, o in enumerate(options):
        tid = tasks[o[0]].id
        if x[col] > plan.get(tid, (None, None, 0.0))[2]:
            plan[tid] = (o[1], o[2], x[col])
    full = {o[2] for o in options}
    share = {name: sum(x[col] for col, o in enumerate(options) if o[2] == name) / max(1, n_z) for name in full}
    return {
        'tasks': n_z, 'options': n_x, 'constraints': len(b_ub),
        'deadline_miss_rate': float(z.sum() / max(1, n_z)),
        'false_alert_rate': float(false_alerts / benign) if benign else 0.0,
        'missed_attack': float(missed / attack) if attack else 0.0,
        'model_share': {k: float(v) for k, v in sorted(share.items())},
        'plan': {tid: (node, model) for tid, (node, model, _) in plan.items()},
    }
