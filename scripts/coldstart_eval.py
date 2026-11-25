"""Cold-start experiment: compare policies in early vs late epochs.

Hypothesis: in the first T seconds of a scenario the adaptive method hasn't
accumulated enough A/E signal (alert evidence), so DFL-specialist — which uses
a learned prior requiring no warm-up — should perform better early on.

We split each 60-second scenario into three phases:
  - early:  [0, 10) s  — adaptive has little signal (100 epochs)
  - mid:    [10, 30) s — adaptive is warming up
  - late:   [30, 60) s — adaptive has converged

For each phase we evaluate only the tasks released in that window.

Usage:
    python scripts/coldstart_eval.py [--families f1-detectors f3-evasive]
                                      [--seeds 2 3 4 5 6]
                                      [--output artifacts/coldstart]
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

PHASES = [
    ('early', 0.0, 10.0),
    ('mid',  10.0, 30.0),
    ('late', 30.0, 60.0),
]

POLICIES = [
    'ps-anneal-alerts',
    'ps-anneal-adaptive',
    'ps-anneal-dfl2-specialist',
]

FAMILIES = ['f1-detectors', 'f3-evasive']

BASE = ROOT / 'configs' / 'ids-place-smokescreen-x3.json'


def evaluate_subset(sc, tasks_subset, decisions, finish):
    """Compute objective metrics for a subset of tasks."""
    from fogids.smokescreen import FLOOD, TOL
    if not tasks_subset:
        return None

    # Build class counts only over this subset
    counts = {}
    for t in tasks_subset:
        for c, _ in t.attacks:
            counts[c] = counts.get(c, 0) + 1

    det = {c: 0.0 for c in counts}
    fa = 0.0
    benign = 0
    dropped = 0
    late = 0

    for t in tasks_subset:
        opt = decisions.get(t.id)
        from fogids.smokescreen import DROP
        if opt is None:
            opt = DROP
        on_time = (opt.node is not None and finish.get(t.id) is not None
                   and finish[t.id] <= t.release_s + t.deadline_s + TOL)
        if opt.node is None:
            dropped += 1
        elif not on_time:
            late += 1
        for c, p in sc.detection(t, opt, on_time).items():
            if c in det:
                det[c] += p
        fa += sc.false_alerts(t, opt, on_time)
        benign += t.benign_flows

    if not counts:
        return None

    detected = {c: det[c] / counts[c] for c in counts}
    quiet = [c for c in counts if c not in FLOOD]
    flood = [c for c in counts if c in FLOOD]
    missed = 1.0 - sum(detected.values()) / max(1, len(detected))
    return {
        'objective': missed + sc.p['fa_weight'] * fa / max(1, benign),
        'missed_quiet': 1.0 - sum(detected[c] for c in quiet) / max(1, len(quiet)),
        'missed_flood': 1.0 - sum(detected[c] for c in flood) / max(1, len(flood)),
        'false_alert_rate': fa / max(1, benign),
        'dropped': dropped / max(1, len(tasks_subset)),
        'late': late / max(1, len(tasks_subset)),
        'n_tasks': len(tasks_subset),
    }


def _load_variants():
    from fogids.families import FAMILIES
    return dict(FAMILIES)


def _policy_for(policy, variant, kw):
    if policy == 'ps-anneal-dfl2-specialist':
        from fogids.dfl import DFLPSAnneal
        return DFLPSAnneal(ROOT / f'artifacts/models/value-gbm-v2-only-{variant}.pkl',
                           ROOT / f'artifacts/models/dfl2-only-{variant}.json', **kw)
    # For other policies use robustness.policy_for via sys.path trick
    import importlib.util
    spec = importlib.util.spec_from_file_location('_robustness', ROOT / 'scripts' / 'robustness.py')
    rob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rob)
    return rob.policy_for(policy, variant, kw)


def run_one(variant, policy_name, seed):
    """Run a single scenario and return per-phase metrics."""
    from fogids.smokescreen import Engine, generate

    variants = _load_variants()

    c = json.loads(BASE.read_text(encoding='utf-8'))
    c = copy.deepcopy(c)
    c['seed'] = seed
    variants[variant](c)
    sc = generate(c)

    policy = _policy_for(policy_name, variant, {})
    engine = Engine(sc)
    engine.run(policy)

    decisions = engine.decisions
    finish = engine.finish

    results = {}
    for phase, t0, t1 in PHASES:
        subset = [t for t in sc.tasks if t0 <= t.release_s < t1]
        metrics = evaluate_subset(sc, subset, decisions, finish)
        if metrics is not None:
            results[phase] = metrics

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='+', default=FAMILIES)
    ap.add_argument('--seeds', nargs='+', type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument('--output', default='artifacts/coldstart')
    args = ap.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'runs.jsonl'

    done = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                done.add((r['variant'], r['policy'], r['seed']))

    total = len(args.families) * len(POLICIES) * len(args.seeds)
    n = 0
    with open(out_file, 'a') as fh:
        for variant in args.families:
            for policy in POLICIES:
                for seed in args.seeds:
                    n += 1
                    key = (variant, policy, seed)
                    if key in done:
                        print(f'[skip] {variant} {policy} seed {seed}')
                        continue
                    print(f'[{n}/{total}] {variant:<20} {policy:<30} seed {seed}', end='', flush=True)
                    try:
                        phases = run_one(variant, policy, seed)
                        row = {'variant': variant, 'policy': policy, 'seed': seed, 'phases': phases}
                        fh.write(json.dumps(row) + '\n')
                        fh.flush()
                        early_obj = phases.get('early', {}).get('objective', float('nan'))
                        late_obj = phases.get('late', {}).get('objective', float('nan'))
                        print(f'  early {early_obj:.4f}  late {late_obj:.4f}')
                    except Exception as e:
                        print(f'  ERROR: {e}')

    # Print summary table
    print('\n=== Cold-start summary (mean objective by phase) ===')
    runs = []
    with open(out_file) as f:
        for line in f:
            runs.append(json.loads(line))

    from collections import defaultdict
    agg = defaultdict(list)
    for r in runs:
        for phase, metrics in r['phases'].items():
            agg[(r['variant'], r['policy'], phase)].append(metrics['objective'])

    short = {
        'ps-anneal-alerts': 'formula',
        'ps-anneal-adaptive': 'adaptive',
        'ps-anneal-dfl2-specialist': 'specialist',
    }

    for variant in args.families:
        print(f'\n{variant}:')
        print(f"  {'policy':<12}  {'early(0-10s)':>12}  {'mid(10-30s)':>12}  {'late(30-60s)':>12}")
        for policy in POLICIES:
            row = [f"  {short[policy]:<12}"]
            for phase, _, _ in PHASES:
                vals = agg.get((variant, policy, phase), [])
                if vals:
                    row.append(f"  {sum(vals)/len(vals):>12.4f}")
                else:
                    row.append(f"  {'---':>12}")
            print(''.join(row))

    out_dir_str = str(out_dir.relative_to(ROOT))
    print(f'\nResults saved to {out_dir_str}/runs.jsonl')


if __name__ == '__main__':
    main()
