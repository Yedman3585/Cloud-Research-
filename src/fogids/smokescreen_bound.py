"""A lower bound for the smokescreen scenario that respects processor sharing.

The fluid bound (``smokescreen.fluid_bound``) lets work be served in any order, so on a
flooded gateway it assumes the window's chunks can be finished one after another. Under
processor sharing they cannot: jobs that start together on one CPU slow each other,
and job i, starting with the other jobs of its window, finishes after
``sum_j min(a_i, a_j)`` seconds of that CPU (a_j = work / MIPS). Any additional job only
delays it further (processor-sharing finish times are monotone in the set of jobs).

Relaxation, per gateway window (the tasks of one gateway released at the same time):
every chosen option's first stage runs on the gateway with only the other tasks of its
window (jobs of earlier windows, of other gateways and the decision delay are left out),
and everything after the first stage takes its idle time (no contention on uplinks, fog
or cloud). A chosen option must then satisfy

    sum_{(j, p) chosen} min(a_io, a_jp) <= deadline_i - downstream_io,

a linear constraint in the binaries x_jp once activated by x_io (big-M). The window's
mixed-integer program minimises the exact objective contribution of its tasks (ground
truth values, as in the fluid bound); HiGHS's dual bound is used, so a window not solved
to optimality within its time limit still gives a valid bound. The sum over windows is a
lower bound on the objective of every policy.
"""
from .smokescreen import FLOOD, TOL


def ps_window_bound(sc, time_limit_s=10.0, verbose=False):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    counts = sc.class_counts()
    windows = {}
    for t in sc.tasks:
        windows.setdefault((t.gateway, t.release_s), []).append(t)
    total, gap_windows, solved = 0.0, 0, 0
    det = {c: 0.0 for c in counts}
    fa = 0.0
    for (gw, _), tasks in windows.items():
        mips = sc.node(gw).mips
        opts, cost, load, budget, owner = [], [], [], [], []
        for i, t in enumerate(tasks):
            for o in sc.options(t):
                stages = sc.stages(t, o)
                if o.node is not None and sc.standalone_s(t, o) > t.deadline_s + TOL:
                    continue
                first = stages[0][2] / mips if stages else 0.0
                downstream = sc.standalone_s(t, o) - first if stages else 0.0
                opts.append((t, o))
                cost.append(sc.cost(t, o, counts))
                load.append(first)
                budget.append(t.deadline_s - downstream)
                owner.append(i)
        n = len(opts)
        # One option per task.
        A_eq = lil_matrix((len(tasks), n))
        for k, i in enumerate(owner):
            A_eq[i, k] = 1.0
        # Processor sharing on the gateway, for every option with a first stage.
        rows = [k for k in range(n) if load[k] > 0]
        A = lil_matrix((len(rows), n))
        ub = []
        for r, k in enumerate(rows):
            a = load[k]
            coef = np.minimum(a, np.array(load))
            coef[[q for q in range(n) if owner[q] == owner[k]]] = 0.0
            coef[k] = a  # its own work
            # max over the other tasks' options of min(a, a_jp), for the big-M
            worst = a
            for j in set(owner) - {owner[k]}:
                worst += max(coef[q] for q in range(n) if owner[q] == j)
            M = max(0.0, worst - budget[k])
            for q in np.flatnonzero(coef):
                A[r, q] = coef[q]
            A[r, k] = a + M
            ub.append(budget[k] + M)
        constraints = [LinearConstraint(A_eq.tocsr(), 1.0, 1.0)]
        if rows:
            constraints.append(LinearConstraint(A.tocsr(), -np.inf, np.array(ub)))
        res = milp(np.array(cost), constraints=constraints, integrality=np.ones(n),
                   bounds=Bounds(0, 1), options={'time_limit': time_limit_s, 'disp': False})
        if res.x is None and getattr(res, 'mip_dual_bound', None) is None:
            raise RuntimeError(f'Window {gw}: {res.message}')
        if res.status == 0:
            solved += 1
            bound = res.fun
        else:  # time limit: the dual bound is still a valid lower bound
            gap_windows += 1
            bound = res.mip_dual_bound
        total += bound
        if verbose and len(tasks) > 4:
            print(f'{gw} {len(tasks)} tasks {n} options: {"optimal" if res.status == 0 else "dual bound"}', flush=True)
        if res.x is not None:
            for k in np.flatnonzero(res.x > 0.5):
                t, o = opts[k]
                for c, p in sc.detection(t, o).items():
                    det[c] += p
                fa += sc.false_alerts(t, o)
    detected = {c: det[c] / counts[c] for c in counts}
    benign = sum(t.benign_flows for t in sc.tasks)
    quiet = [c for c in counts if c not in FLOOD]
    flood = [c for c in counts if c in FLOOD]
    return {
        'objective': total,
        'windows': len(windows), 'solved': solved, 'not_optimal': gap_windows,
        # Of the optimal per-window plans (when all windows are solved to optimality,
        # these describe the bound's solution):
        'missed_quiet': 1.0 - sum(detected[c] for c in quiet) / max(1, len(quiet)),
        'missed_flood': 1.0 - sum(detected[c] for c in flood) / max(1, len(flood)),
        'false_alert_rate': fa / max(1, benign),
    }
