"""Diagnostic: where is the bottleneck — GBM prediction or DFL weights?

Runs F3-evasive with 4 policies to decompose the formula→oracle gap:
  1. formula       (ps-anneal-alerts)     — baseline, A/E heuristic
  2. gbm-specialist (ps-anneal-learned)   — specialist GBM, fixed weights s=1 λ=1e-3
  3. dfl-specialist (ps-anneal-dfl2-specialist) — specialist GBM + learned s(t), λ(t)
  4. oracle        (ps-anneal-true-a)     — true attack flows, fixed weights

If gbm ≈ formula >> oracle  → prediction is the bottleneck (GBM ≈ A/E)
If gbm ≈ oracle >> formula  → GBM is good; DFL doesn't help beyond GBM
If dfl >> gbm                → DFL weights add value on top of GBM

Also reports per-task prediction error (GBM vs formula vs true) on each run.

Usage:
    python scripts/diagnose_gap.py [--seeds 2 3 4 5 6]
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

VARIANT = 'f3-evasive'
BASE = ROOT / 'configs' / 'ids-place-smokescreen-x3.json'

PHASES = [
    ('early', 0.0, 10.0),
    ('mid',  10.0, 30.0),
    ('late', 30.0, 60.0),
    ('full', 0.0, 60.0),
]


def evaluate_subset(sc, tasks_subset, decisions, finish):
    """Compute objective for a subset of tasks."""
    from fogids.smokescreen import FLOOD, TOL, DROP
    if not tasks_subset:
        return None

    counts = {}
    for t in tasks_subset:
        for c, _ in t.attacks:
            counts[c] = counts.get(c, 0) + 1

    det = {c: 0.0 for c in counts}
    fa = 0.0
    benign = 0

    for t in tasks_subset:
        opt = decisions.get(t.id, DROP)
        on_time = (opt.node is not None and finish.get(t.id) is not None
                   and finish[t.id] <= t.release_s + t.deadline_s + TOL)
        for c, p in sc.detection(t, opt, on_time).items():
            if c in det:
                det[c] += p
        fa += sc.false_alerts(t, opt, on_time)
        benign += t.benign_flows

    if not counts:
        return None

    detected = {c: det[c] / counts[c] for c in counts}
    missed = 1.0 - sum(detected.values()) / max(1, len(detected))
    return missed + sc.p['fa_weight'] * fa / max(1, benign)


def make_policy(name, variant):
    """Create policy by name."""
    kw = {}
    if name == 'formula':
        from fogids.smokescreen_qubo import AlertPSAnneal
        return AlertPSAnneal(**kw)
    elif name == 'gbm-specialist':
        from fogids.learned import LearnedPSAnneal
        return LearnedPSAnneal(ROOT / f'artifacts/models/value-gbm-v2-only-{variant}.pkl', **kw)
    elif name == 'dfl-specialist':
        from fogids.dfl import DFLPSAnneal
        return DFLPSAnneal(ROOT / f'artifacts/models/value-gbm-v2-only-{variant}.pkl',
                           ROOT / f'artifacts/models/dfl2-only-{variant}.json', **kw)
    elif name == 'oracle':
        from fogids.learned import TrueAttackPSAnneal
        return TrueAttackPSAnneal(**{'floor': 3.0, **kw})
    else:
        raise ValueError(f'unknown policy: {name}')


def prediction_errors(sc, policy, tasks):
    """Compute MAE of the attack-flow estimate vs true for each task."""
    from fogids.smokescreen import FLOOD
    errors = []
    for t in tasks:
        true_a = sum(a for c, a in t.attacks if c not in FLOOD) if t.stream != 'flood' \
                 else sum(a for _, a in t.attacks)
        if hasattr(policy, 'predicted') and t.id in policy.predicted:
            est = policy.predicted[t.id]
        elif hasattr(policy, 'evidence'):
            if t.stream == 'flood':
                est = max(getattr(policy, 'min_attack', 5.0), t.estimated_attack_flows)
            else:
                est = max(getattr(policy, 'floor', 1.0),
                          policy.evidence.share(t.gateway) * t.n_flows)
        else:
            continue
        errors.append({'true': true_a, 'est': est, 'err': abs(est - true_a),
                       'stream': t.stream, 'gateway': t.gateway})
    return errors


def run_one(seed, policies_names):
    """Run one scenario with all policies, return per-policy results."""
    from fogids.smokescreen import Engine, generate
    from fogids.families import FAMILIES

    variants = dict(FAMILIES)

    c = json.loads(BASE.read_text(encoding='utf-8'))
    c = copy.deepcopy(c)
    c['seed'] = seed
    variants[VARIANT](c)
    sc = generate(c)

    results = {}
    for pname in policies_names:
        policy = make_policy(pname, VARIANT)
        engine = Engine(sc)
        engine.run(policy)

        phase_results = {}
        for phase, t0, t1 in PHASES:
            subset = [t for t in sc.tasks if t0 <= t.release_s < t1]
            obj = evaluate_subset(sc, subset, engine.decisions, engine.finish)
            if obj is not None:
                phase_results[phase] = round(obj, 4)

        # Prediction errors (only for formula and gbm-specialist)
        pred_err = prediction_errors(sc, policy, sc.tasks)
        mae = sum(e['err'] for e in pred_err) / max(1, len(pred_err)) if pred_err else None
        mae_residual = [e for e in pred_err if e['stream'] != 'flood']
        mae_residual = sum(e['err'] for e in mae_residual) / max(1, len(mae_residual)) if mae_residual else None

        results[pname] = {
            'phases': phase_results,
            'mae': round(mae, 2) if mae is not None else None,
            'mae_residual': round(mae_residual, 2) if mae_residual is not None else None,
            'n_tasks': len(sc.tasks),
        }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[2, 3, 4, 5, 6])
    args = ap.parse_args()

    policies = ['formula', 'gbm-specialist', 'dfl-specialist', 'oracle']

    print(f'=== Diagnostic: decomposing formula->oracle gap on {VARIANT} ===')
    print(f'Seeds: {args.seeds}\n')

    all_results = {}
    for seed in args.seeds:
        print(f'seed {seed} ...', end='', flush=True)
        results = run_one(seed, policies)
        all_results[seed] = results
        objs = [f'{p}: {results[p]["phases"].get("full", "?")}' for p in policies]
        print(f'  {", ".join(objs)}')

    # Aggregate
    print(f'\n{"="*70}')
    print(f'{"AGGREGATE RESULTS":^70}')
    print(f'{"="*70}\n')

    # Per-phase table
    print('Objective by phase (mean over seeds, lower is better):')
    print(f'  {"policy":<18} {"early":>8} {"mid":>8} {"late":>8} {"full":>8}')
    print(f'  {"-"*18} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for p in policies:
        row = [f'  {p:<18}']
        for phase in ['early', 'mid', 'late', 'full']:
            vals = [all_results[s][p]['phases'].get(phase) for s in args.seeds]
            vals = [v for v in vals if v is not None]
            if vals:
                row.append(f'{sum(vals)/len(vals):>8.4f}')
            else:
                row.append(f'{"---":>8}')
        print(''.join(row))

    # Prediction error table
    print(f'\nAttack-flow prediction MAE (lower = better estimate):')
    print(f'  {"policy":<18} {"MAE (all)":>10} {"MAE (residual)":>15}')
    print(f'  {"-"*18} {"-"*10} {"-"*15}')
    for p in policies:
        mae_vals = [all_results[s][p]['mae'] for s in args.seeds if all_results[s][p]['mae'] is not None]
        mae_r_vals = [all_results[s][p]['mae_residual'] for s in args.seeds if all_results[s][p]['mae_residual'] is not None]
        mae_str = f'{sum(mae_vals)/len(mae_vals):>10.2f}' if mae_vals else f'{"N/A":>10}'
        mae_r_str = f'{sum(mae_r_vals)/len(mae_r_vals):>15.2f}' if mae_r_vals else f'{"N/A":>15}'
        print(f'  {p:<18} {mae_str} {mae_r_str}')

    # Gap decomposition
    print(f'\n{"="*70}')
    print('Gap decomposition (full scenario, mean):')
    means = {}
    for p in policies:
        vals = [all_results[s][p]['phases']['full'] for s in args.seeds]
        means[p] = sum(vals) / len(vals)

    total_gap = means['formula'] - means['oracle']
    gbm_gain = means['formula'] - means['gbm-specialist']
    dfl_gain = means['gbm-specialist'] - means['dfl-specialist']
    remaining = means['dfl-specialist'] - means['oracle']

    print(f'  Total gap (formula -> oracle):     {total_gap:+.4f}  (100%)')
    print(f'  GBM gain  (formula -> gbm):        {gbm_gain:+.4f}  ({100*gbm_gain/total_gap if total_gap else 0:.0f}%)')
    print(f'  DFL gain  (gbm -> dfl):            {dfl_gain:+.4f}  ({100*dfl_gain/total_gap if total_gap else 0:.0f}%)')
    print(f'  Remaining (dfl -> oracle):          {remaining:+.4f}  ({100*remaining/total_gap if total_gap else 0:.0f}%)')

    # Save raw results
    out = ROOT / 'artifacts' / 'diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'gap_analysis.json', 'w') as f:
        json.dump({'variant': VARIANT, 'seeds': args.seeds,
                   'results': {str(s): r for s, r in all_results.items()},
                   'means': means, 'gap': {'total': total_gap, 'gbm': gbm_gain,
                                           'dfl': dfl_gain, 'remaining': remaining}},
                  f, indent=2)
    print(f'\nResults saved to {out}/gap_analysis.json')


if __name__ == '__main__':
    main()
