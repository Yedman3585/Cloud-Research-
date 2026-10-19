"""Compiled annealer for ``EpochPS`` (``smokescreen_qubo.py``).

Same model and moves as ``EpochPS.anneal``, on dense arrays and compiled with numba:
for every resource, the load of each job (0 = not on it); for every job, its
predicted latency F = sum over its resources of sum_k min(a_j, a_k). Adding or removing
a job changes F of every job sharing a resource with it by min(a_j, a_k), so a move is
O(jobs on the resources involved). The temperature follows the same geometric
schedule over a fixed number of moves; the wrapper sets that number from the measured
move rate so that one call takes about the time budget.

Without numba the pure-Python ``EpochPS.anneal`` is used.
"""
import math
import time

import numpy as np

try:
    from numba import njit
except ImportError:  # optional dependency
    njit = None

TOL = 1e-6


def _kernel(nopt, nres, ores, oload, val, budget, horizon, fres, fload, fval,
            start, iters, t0, t1, kappa, seed):
    T = nopt.shape[0]
    Fx = fres.shape[0]
    J = T + Fx
    R = 0
    for i in range(T):
        for k in range(nopt[i]):
            for s in range(nres[i, k]):
                if ores[i, k, s] + 1 > R:
                    R = ores[i, k, s] + 1
    for f in range(Fx):
        if fres[f] + 1 > R:
            R = fres[f] + 1
    load = np.zeros((R, J))
    F = np.zeros(J)
    value = np.zeros(J)
    on = np.zeros(J, np.bool_)
    choice = start.copy()

    # Place fixed jobs and the start assignment.
    for f in range(Fx):
        j = T + f
        r = fres[f]
        a = fload[f]
        tot = a
        for k in range(J):
            b = load[r, k]
            if b > 0.0:
                m = a if a < b else b
                tot += m
                F[k] += m
        load[r, j] = a
        F[j] = tot
        value[j] = fval[f]
        on[j] = True
    for i in range(T):
        k0 = choice[i]
        value[i] = val[i, k0]
        on[i] = nres[i, k0] > 0
        for s in range(nres[i, k0]):
            r = ores[i, k0, s]
            a = oload[i, k0, s]
            tot = a
            for k in range(J):
                b = load[r, k]
                if b > 0.0:
                    m = a if a < b else b
                    tot += m
                    F[k] += m
            load[r, i] = a
            F[i] += tot

    np.random.seed(seed)
    mark = np.zeros(J, np.int64)
    stamp = 0
    affected = np.zeros(J, np.int64)
    best = choice.copy()
    e = 0.0
    for j in range(J):
        if on[j]:
            late = F[j] - budget[j]
            c = (-value[j] if late <= TOL else 0.0)
            if late > 0.0:
                c += kappa * late / horizon[j]
            e += c
    best_e = e
    movable = np.zeros(T, np.int64)
    nm = 0
    for i in range(T):
        if nopt[i] > 1:
            movable[nm] = i
            nm += 1
    if nm == 0:
        return best, best_e
    for it in range(iters):
        temp = t1 * (t0 / t1) ** (1.0 - it / iters)
        i = movable[np.random.randint(nm)]
        old = choice[i]
        new = np.random.randint(nopt[i])
        if new == old:
            continue
        # Affected jobs: those on the resources of the old and new option, and i.
        stamp += 1
        na = 0
        mark[i] = stamp
        affected[na] = i
        na += 1
        for which in range(2):
            kk = old if which == 0 else new
            for s in range(nres[i, kk]):
                r = ores[i, kk, s]
                for k in range(J):
                    if load[r, k] > 0.0 and mark[k] != stamp:
                        mark[k] = stamp
                        affected[na] = k
                        na += 1
        before = 0.0
        for q in range(na):
            j = affected[q]
            if on[j]:
                late = F[j] - budget[j]
                c = (-value[j] if late <= TOL else 0.0)
                if late > 0.0:
                    c += kappa * late / horizon[j]
                before += c
        # Remove old, add new.
        for which in range(2):
            kk = old if which == 0 else new
            sign = -1.0 if which == 0 else 1.0
            for s in range(nres[i, kk]):
                r = ores[i, kk, s]
                a = oload[i, kk, s]
                if which == 0:
                    load[r, i] = 0.0
                tot = a
                for k in range(J):
                    b = load[r, k]
                    if b > 0.0 and k != i:
                        m = a if a < b else b
                        tot += m
                        F[k] += sign * m
                F[i] += sign * tot
                if which == 1:
                    load[r, i] = a
        value[i] = val[i, new]
        on[i] = nres[i, new] > 0
        after = 0.0
        for q in range(na):
            j = affected[q]
            if on[j]:
                late = F[j] - budget[j]
                c = (-value[j] if late <= TOL else 0.0)
                if late > 0.0:
                    c += kappa * late / horizon[j]
                after += c
        d = after - before
        if d <= 0.0 or np.random.random() < math.exp(-d / temp):
            choice[i] = new
            e += d
            if e < best_e - 1e-12:
                best_e = e
                for q in range(T):
                    best[q] = choice[q]
        else:
            # Undo: remove new, add old.
            for which in range(2):
                kk = new if which == 0 else old
                sign = -1.0 if which == 0 else 1.0
                for s in range(nres[i, kk]):
                    r = ores[i, kk, s]
                    a = oload[i, kk, s]
                    if which == 0:
                        load[r, i] = 0.0
                    tot = a
                    for k in range(J):
                        b = load[r, k]
                        if b > 0.0 and k != i:
                            m = a if a < b else b
                            tot += m
                            F[k] += sign * m
                    F[i] += sign * tot
                    if which == 1:
                        load[r, i] = a
            value[i] = val[i, old]
            on[i] = nres[i, old] > 0
    return best, best_e


_compiled = None


def _get_kernel():
    global _compiled
    if _compiled is None and njit is not None:
        src = _kernel
        _compiled = njit(cache=True)(src)
    return _compiled


class FastAnnealer:
    """Anneals ``EpochPS`` models within a wall-time budget per call."""

    def __init__(self):
        self.rate = None  # moves per second, measured
        self.calls = 0

    @staticmethod
    def pack(m):
        res_id = {}
        T = len(m.options)
        K = max(len(o) for o in m.options)
        S = max((len(s) for row in m.stages for s in row), default=1) or 1
        nopt = np.array([len(o) for o in m.options], np.int64)
        nres = np.zeros((T, K), np.int64)
        ores = np.zeros((T, K, S), np.int64)
        oload = np.zeros((T, K, S))
        val = np.zeros((T, K))
        for i in range(T):
            for k, loads in enumerate(m.stages[i]):
                nres[i, k] = len(loads)
                for s, (res, a) in enumerate(loads.items()):
                    ores[i, k, s] = res_id.setdefault(res, len(res_id))
                    oload[i, k, s] = max(a, 1e-12)
                val[i, k] = m.values[i][k]
        Fx = len(m.fixed)
        fres = np.zeros(Fx, np.int64)
        fload = np.zeros(Fx)
        fval = np.zeros(Fx)
        fb, fh = [], []
        for f, (v, loads, B, D) in enumerate(m.fixed):
            (res, a), = loads.items()
            fres[f] = res_id.setdefault(res, len(res_id))
            fload[f] = max(a, 1e-12)
            fval[f] = v
            fb.append(B)
            fh.append(D)
        budget = np.array(list(m.budgets) + fb, float)
        horizon = np.array(list(m.horizons) + fh, float)
        return nopt, nres, ores, oload, val, budget, horizon, fres, fload, fval

    def anneal(self, m, start, budget_s, seed, t0, t1):
        kernel = _get_kernel()
        if kernel is None:
            return m.anneal(start, budget_s, seed, t0, t1)
        packed = self.pack(m)
        nopt, nres, ores, oload, val, budget, horizon, fres, fload, fval = packed
        iters = 2000 if self.rate is None else max(200, int(self.rate * budget_s))
        if self.calls == 0:
            iters = 200  # compile on a short run
        started = time.perf_counter()
        best, best_e = kernel(nopt, nres, ores, oload, val, budget, horizon, fres, fload, fval,
                              np.array(start, np.int64), iters, t0, t1, m.kappa, seed)
        elapsed = time.perf_counter() - started
        self.calls += 1
        if self.calls > 1 and elapsed > 1e-4 and iters >= 200:  # the first call compiles
            rate = iters / elapsed
            self.rate = rate if self.rate is None else 0.7 * self.rate + 0.3 * rate
        return [int(x) for x in best], float(best_e)
