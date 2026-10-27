"""Train the decision-focused value weights of ``ps-anneal-dfl`` (``fogids.dfl``).

Data: the epoch models recorded by ``scripts/collect_values.py`` (every option of every
task with its loads, the value the policy used and the exact value from ground truth).
For each epoch the annealer is the solver: x*(c) is its solution with the exact values c,
and the SPO+ loss of predicted values v is

    L(v) = max_x [(2v - c) . x - P(x)] - 2v . x*(c) + c . x*(c),

with P the deadline penalty of the epoch model; its subgradient in v is 2(x~ - x*), x~ the
solution with values 2v - c, and x the on-time chosen options. The attack-flow estimate in
v comes from gradient-boosted trees cross-fitted over seeds (even/odd), so that it is not
evaluated on its own training data. Starting point: the values of ``ps-anneal-learned``
with false-alert weight 1e-3.

    python scripts/train_dfl.py                       # every condition
    python scripts/train_dfl.py --exclude drift       # all but one

Output: artifacts/models/dfl.json or dfl-without-<variant>.json (dfl2... with --features v2).
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

FLOOR, FA_WEIGHT, ITERS, T0, T1 = 3.0, 1e-3, 6000, 0.5, 0.005


def finish_times():
    from numba import njit

    @njit(cache=True)
    def _finish(nopt, nres, ores, oload, fres, fload, choice):
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
        for i in range(T):
            k0 = choice[i]
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
        return F[:T]
    return _finish


class Epoch:
    """One recorded epoch, packed for the annealer kernel."""

    def __init__(self, rec, rows, a_hat, recall, fpr):
        from fogids.fast_anneal import FastAnnealer
        from fogids.smokescreen_qubo import EpochPS
        tasks = [rows[i] for i in rec['tasks']]
        chunks = {}
        for t in tasks:
            if t['stream'] == 'flood':
                chunks[(t['gateway'], t['release_s'])] = chunks.get((t['gateway'], t['release_s']), 0) + 1
        by_task = [[] for _ in tasks]
        for o in rec['options']:
            by_task[o['task']].append(o)
        T, K = len(tasks), max(len(r) for r in by_task)
        self.P, self.FA, self.CS = np.zeros((T, K)), np.zeros((T, K)), np.zeros((T, K))
        self.C, self.REC = np.zeros((T, K)), np.zeros((T, K))
        self.fits = np.zeros((T, K), bool)
        stages = []
        for i, (t, opts) in enumerate(zip(tasks, by_task)):
            a = max(FLOOR, a_hat[t['id']])
            share = 1.0 / chunks[(t['gateway'], t['release_s'])] if t['stream'] == 'flood' else 1.0
            row = []
            for k, o in enumerate(opts):
                loads = {(r[0], r[1]): x for r, x in o['loads']}
                row.append(loads)
                self.fits[i, k] = sum(loads.values()) <= rec['budgets'][i]
                self.C[i, k], self.REC[i, k] = o['oracle_value'], o['value']
                if o['node'] is None:
                    continue
                self.P[i, k] = 1.0 - (1.0 - o['f'] * recall[o['model']]) ** a
                self.FA[i, k] = o['f'] * t['n_flows'] * (1.0 - t['prefilter_score']) * fpr[o['model']]
                self.CS[i, k] = share * t['criticality']
            stages.append(row)
            self.fits[i, 0] = True if not self.fits[i].any() else self.fits[i, 0]
        fixed = [(v, {(r[0][0], r[0][1]): r[1] for r in loads}, B, D) for v, loads, B, D in rec['fixed']]
        m = EpochPS([[None] * len(r) for r in stages], [[0.0] * len(r) for r in stages], stages,
                    rec['budgets'], rec['horizons'], fixed, rec['kappa'])
        packed = FastAnnealer.pack(m)
        self.nopt, self.nres, self.ores, self.oload, _, self.budget, self.horizon, self.fres, self.fload, self.fval = packed
        self.kappa, self.T, self.K = rec['kappa'], T, K
        self.task_ids = [t['id'] for t in tasks]

    def solve(self, kernel, values, seed, iters=ITERS, start=None):
        if start is None:
            masked = np.where(self.fits, values, -np.inf)
            start = np.argmax(masked[:, :], axis=1)
            start = np.minimum(start, self.nopt - 1)
        best, e = kernel(self.nopt, self.nres, self.ores, self.oload, np.ascontiguousarray(values), self.budget,
                         self.horizon, self.fres, self.fload, self.fval, np.asarray(start, np.int64), iters,
                         T0, T1, self.kappa, seed)
        return best, e

    def energy(self, kernel, values, choice):
        return self.solve(kernel, values, 0, iters=0, start=choice)[1]

    def x(self, finish, choice):
        F = finish(self.nopt, self.nres, self.ores, self.oload, self.fres, self.fload, np.asarray(choice, np.int64))
        X = np.zeros((self.T, self.K))
        for i, k in enumerate(choice):
            if self.nres[i, k] > 0 and F[i] <= self.budget[i] + 1e-6:
                X[i, k] = 1.0
        return X


def crossfit(runs, names):
    """Attack-flow estimate of every task from trees trained on the other seed parity."""
    from fogids.learned import feature_row
    from train_values import fit, target
    out = {}
    for parity in (0, 1):
        train = [(feature_row(r, names), target(r)) for (v, s), (rows, _) in runs.items() if s % 2 != parity
                 for r in rows.values()]
        model = fit([x for x, _ in train], [y for _, y in train])
        for (v, s), (rows, _) in runs.items():
            if s % 2 == parity:
                ids = list(rows)
                pred = model.predict([feature_row(rows[i], names) for i in ids])
                out[(v, s)] = dict(zip(ids, (float(p) for p in pred)))
    return out


def load_runs(data, variants, every):
    runs = {}
    for path in sorted(data.rglob('seed-*.epochs.jsonl.gz')):
        variant = 'base' if path.parent == data else path.parent.name
        if variant not in variants:
            continue
        seed = int(path.name.split('-')[1].split('.')[0])
        rows = {}
        with gzip.open(path.with_name(f'seed-{seed}.tasks.jsonl.gz'), 'rt', encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                if 'prefilter_score' in r:
                    rows[r['id']] = r
        recs = []
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for k, line in enumerate(fh):
                if k % every == 0:
                    rec = json.loads(line)
                    if len(rec['tasks']) >= 2:
                        recs.append(rec)
        runs[(variant, seed)] = (rows, recs)
    return runs


def main():
    from fogids.dfl import PHI, PHI_V2, phi_raw
    from fogids.learned import FEATURES, FEATURES_V2
    from fogids.fast_anneal import _get_kernel
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts/value-data')
    parser.add_argument('--variants', nargs='+', default=['base', 'load-x1', 'load-x5', 'separation-5', 'drift'])
    parser.add_argument('--exclude', default=None)
    parser.add_argument('--features', choices=('v1', 'v2'), default='v1',
                        help='v2: feature set v2 for the attack-flow model and the weights (scenario families)')
    parser.add_argument('--seeds-per-condition', type=int, default=6)
    parser.add_argument('--every', type=int, default=3, help='use every n-th epoch')
    parser.add_argument('--passes', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--l2', type=float, default=1e-2)
    parser.add_argument('--method', choices=('es', 'spo'), default='es',
                        help='es: zeroth-order descent on the regret (default); spo: SPO+ subgradient')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--batch', type=int, default=150)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--step-size', type=float, default=0.05)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    variants = [v for v in args.variants if v != args.exclude]
    prefix = 'dfl2' if args.features == 'v2' else 'dfl'
    out = args.output or ROOT / 'artifacts/models' / (f'{prefix}-without-{args.exclude}.json' if args.exclude
                                                      else f'{prefix}.json')
    names, phi_names = (FEATURES_V2, PHI_V2) if args.features == 'v2' else (FEATURES, PHI)
    started = time.perf_counter()
    runs = load_runs(args.data, variants, args.every)
    keep = {}
    for v in variants:
        seeds = sorted(s for vv, s in runs if vv == v)[:args.seeds_per_condition]
        keep.update({(v, s): runs[(v, s)] for s in seeds})
    runs = keep
    print(f'{sum(len(r[1]) for r in runs.values())} epochs from {len(runs)} runs ({", ".join(variants)}); '
          f'{time.perf_counter() - started:.0f} s', flush=True)
    a_hat = crossfit(runs, names)
    print(f'cross-fitted attack-flow estimates; {time.perf_counter() - started:.0f} s', flush=True)

    config = json.loads((ROOT / 'configs/ids-place-smokescreen-x3.json').read_text(encoding='utf-8'))
    recall = {m['name']: sum(m['recall'].values()) / len(m['recall']) for m in config['models']}
    fpr = {m['name']: m['false_positive_rate'] for m in config['models']}
    raw = [phi_raw(r, a_hat[key][i], phi_names) for key, (rows, _) in runs.items() for i, r in rows.items()]
    mean, std = np.mean(raw, axis=0), np.std(raw, axis=0) + 1e-9
    mean[0], std[0] = 0.0, 1.0

    kernel, finish = _get_kernel(), finish_times()
    samples = []
    for key, (rows, recs) in runs.items():
        last = max(s for v, s in runs if v == key[0])
        for rec in recs:
            e = Epoch(rec, rows, a_hat[key], recall, fpr)
            e.phi = (np.array([phi_raw(rows[i], a_hat[key][i], phi_names) for i in e.task_ids]) - mean) / std
            e.phi[:, 0] = 1.0
            e.val = key[1] == last
            e.key = key
            samples.append(e)
    rng = np.random.default_rng(0)
    for e in samples:
        best, _ = e.solve(kernel, e.C, 1)
        e.xstar, e.estar = e.x(finish, best), e.energy(kernel, e.C, best)
    train = [e for e in samples if not e.val]
    val = [e for e in samples if e.val]
    print(f'{len(train)} training and {len(val)} validation epochs; x* solved; '
          f'{time.perf_counter() - started:.0f} s', flush=True)

    n = len(phi_names)
    theta0_s, theta0_l = np.zeros(n), np.zeros(n)
    theta0_l[0] = math.log(FA_WEIGHT)
    theta = np.concatenate([theta0_s, theta0_l])
    theta0 = theta.copy()

    def values(e, th):
        s = np.exp(np.clip(e.phi @ th[:n], -20, 20))
        lam = np.exp(np.clip(e.phi @ th[n:], -30, 10))
        return s[:, None] * e.CS * e.P - lam[:, None] * e.FA, s, lam

    def regret(batch, th=None, use=None):
        # The annealer is a heuristic: average over three solver seeds.
        out = {}
        for e in batch:
            v = e.REC if use == 'formula' else values(e, th)[0]
            for seed in (2, 3, 4):
                choice, _ = e.solve(kernel, v, seed)
                out.setdefault(e.key[0], []).append(e.energy(kernel, e.C, choice) - e.estar)
        return {k: float(np.mean(x)) for k, x in out.items()}

    def show(label, r):
        print(f'  {label:28s} ' + '  '.join(f'{k} {v:.4f}' for k, v in sorted(r.items())), flush=True)

    print('validation regret in the epoch model (exact values, lower is better):')
    show('alert formula (as recorded)', regret(val, use='formula'))
    r0 = regret(val, theta)
    show('learned, fa 1e-3 (start)', r0)
    best, best_r = theta.copy(), float(np.mean(list(r0.values())))
    def batch_regret(batch, th, seed):
        return float(np.mean([e.energy(kernel, e.C, e.solve(kernel, values(e, th)[0], seed)[0]) - e.estar
                              for e in batch]))

    if args.method == 'es':
        # Zeroth-order descent on the regret itself: antithetic Gaussian perturbations of
        # theta, the same epochs and solver seed for both members of a pair.
        for step in range(1, args.steps + 1):
            batch = [train[i] for i in rng.choice(len(train), min(args.batch, len(train)), replace=False)]
            grad = np.zeros_like(theta)
            for _ in range(args.pairs):
                eps = rng.standard_normal(theta.shape)
                seed = int(rng.integers(1 << 30))
                diff = batch_regret(batch, theta + args.sigma * eps, seed) - batch_regret(batch, theta - args.sigma * eps, seed)
                grad += diff / (2 * args.sigma) * eps
            grad = grad / args.pairs + args.l2 * (theta - theta0)
            norm = np.linalg.norm(grad)
            if norm > 0:
                theta -= args.step_size * (1 - 0.5 * step / args.steps) * grad / norm
            if step % args.eval_every == 0 or step == args.steps:
                r = regret(val, theta)
                show(f'step {step}', r)
                if float(np.mean(list(r.values()))) < best_r:
                    best, best_r = theta.copy(), float(np.mean(list(r.values())))
                print(f'    {time.perf_counter() - started:.0f} s', flush=True)
    else:
        m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
        step = 0
        for p in range(args.passes):
            order = rng.permutation(len(train))
            loss_sum, loss_n = 0.0, 0
            for b in range(0, len(order), 16):
                grad = np.zeros_like(theta)
                for idx in order[b:b + 16]:
                    e = train[idx]
                    v, s, lam = values(e, theta)
                    choice, emax = e.solve(kernel, 2 * v - e.C, int(rng.integers(1 << 30)))
                    xt = e.x(finish, choice)
                    # SPO+ loss: energies are negated objectives
                    loss_sum += -emax - 2 * (v * e.xstar).sum() + (e.C * e.xstar).sum()
                    loss_n += 1
                    g = 2.0 * (xt - e.xstar)
                    grad[:n] += e.phi.T @ ((g * e.CS * e.P).sum(axis=1) * s)
                    grad[n:] += e.phi.T @ (-(g * e.FA).sum(axis=1) * lam)
                grad = grad / 16 + args.l2 * (theta - theta0)
                step += 1
                m1 = 0.9 * m1 + 0.1 * grad
                m2 = 0.999 * m2 + 0.001 * grad ** 2
                theta -= args.lr * (m1 / (1 - 0.9 ** step)) / (np.sqrt(m2 / (1 - 0.999 ** step)) + 1e-8)
            r = regret(val, theta)
            show(f'pass {p + 1}, SPO+ {loss_sum / max(loss_n, 1):.4f}', r)
            if float(np.mean(list(r.values()))) < best_r:  # early stopping on validation regret
                best, best_r = theta.copy(), float(np.mean(list(r.values())))
            print(f'    {time.perf_counter() - started:.0f} s', flush=True)
    theta = best
    print(f'kept the weights with the lowest validation regret ({best_r:.4f})')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'phi': phi_names, 'features': names, 'mean': mean.tolist(), 'std': std.tolist(),
                               'theta_s': theta[:n].tolist(), 'theta_l': theta[n:].tolist(),
                               'trained_on': variants, 'fa_weight_start': FA_WEIGHT}, indent=1), encoding='utf-8')
    print('weights:', ', '.join(f'{k} s={a:+.2f} l={b:+.2f}' for k, a, b in zip(phi_names, theta[:n], theta[n:])))
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
