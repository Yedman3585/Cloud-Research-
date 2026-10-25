"""Robustness and ablations of the smokescreen results (Python simulator).

Variants are changes of one parameter of the x3 scenario (load, extraction cost,
stream-separation error, topology); ablations are changes of one component of
ps-anneal-alerts. Every (variant, policy, seed) is one job; results are appended to
<output>/runs.jsonl as they finish, and --resume skips jobs already there.

    python scripts/robustness.py --workers 2 --seeds 2 3 4 5 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

BASE = ROOT / 'configs/ids-place-smokescreen-x3.json'
X1 = ROOT / 'configs/ids-place-smokescreen.json'
LARGE = ROOT / 'configs/ids-place-large-calibrated.json'
POLICIES = ('greedy-v2', 'value-greedy-alerts', 'ps-anneal-alerts')


def _scale(c, load):
    """Benign and flood rates at ``load`` times the x1 configuration."""
    x1 = json.loads(X1.read_text(encoding='utf-8'))
    c['traffic']['benign_flows_per_s'] = x1['traffic']['benign_flows_per_s'] * load
    rates = {a['class']: a['flows_per_s'] for a in x1['attacks']}
    for a in c['attacks']:
        if a['class'] in ('ddos', 'dos'):
            a['flows_per_s'] = round(rates[a['class']] * load, 1)
    c['name'] = f'smokescreen-x{load:g}'


def _large(c):
    """Large topology (8 fogs x 16 gateways); the attack pattern of every 32 gateways is
    repeated on every block of 32."""
    t = json.loads(LARGE.read_text(encoding='utf-8'))['topology']
    blocks = t['fog_nodes'] * t['gateways_per_fog'] // 32
    c['topology'].update(fog_nodes=t['fog_nodes'], gateways_per_fog=t['gateways_per_fog'])
    for a in c['attacks']:
        a['gateways'] = sorted(g + 32 * b for g in a['gateways'] for b in range(blocks))
    c['name'] = 'smokescreen-x3-large'


def _drift(c):
    """Non-stationary x3: a moderate DDoS (x1.5) on one set of gateways for 5-25 s, then
    a heavy DDoS and DoS (x5) on other gateways for 30-55 s; low-rate attacks run
    throughout on the same gateways as in x3. One static split cannot fit both phases."""
    x1 = json.loads(X1.read_text(encoding='utf-8'))
    rate = {a['class']: a['flows_per_s'] for a in x1['attacks']}
    flood = [
        {'class': 'ddos', 'start_s': 5.0, 'duration_s': 20.0, 'ramp_s': 1.0,
         'gateways': [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19], 'flows_per_s': round(rate['ddos'] * 1.5, 1)},
        {'class': 'ddos', 'start_s': 30.0, 'duration_s': 25.0, 'ramp_s': 1.0,
         'gateways': [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27], 'flows_per_s': round(rate['ddos'] * 5, 1)},
        {'class': 'dos', 'start_s': 30.0, 'duration_s': 25.0, 'ramp_s': 1.0,
         'gateways': [20, 21, 22, 23], 'flows_per_s': round(rate['dos'] * 5, 1)},
    ]
    quiet = [dict(a, start_s=5.0, duration_s=52.0) for a in c['attacks'] if a['class'] not in ('ddos', 'dos')]
    c['attacks'] = flood + quiet
    c['name'] = 'smokescreen-drift'


VARIANTS = {
    'base': lambda c: None,
    'load-x1': lambda c: _scale(c, 1),
    'load-x1.5': lambda c: _scale(c, 1.5),
    'load-x2': lambda c: _scale(c, 2),
    'load-x2.5': lambda c: _scale(c, 2.5),
    'load-x4': lambda c: _scale(c, 4),
    'load-x5': lambda c: _scale(c, 5),
    'extract-1us': lambda c: c['v2'].update(extract_us_per_packet=1.0),
    'extract-3us': lambda c: c['v2'].update(extract_us_per_packet=3.0),
    'separation-0': lambda c: c['v2'].update(separation_error=0.0),
    'separation-5': lambda c: c['v2'].update(separation_error=0.05),
    'large': _large,
    'drift': _drift,
}

ABLATIONS = {
    'ps-anneal (no alerts)': ('ps-anneal', {}),
    'no pruning': ('ps-anneal-alerts', {'prune': False}),
    'no repair': ('ps-anneal-alerts', {'repair': False}),
    'budget 10 ms': ('ps-anneal-alerts', {'budget_s': 0.01}),
    'budget 25 ms': ('ps-anneal-alerts', {'budget_s': 0.025}),
    'budget 100 ms': ('ps-anneal-alerts', {'budget_s': 0.1}),
    'alert window 10 s': ('ps-anneal-alerts', {'horizon_s': 10.0}),
    'alert window 20 s': ('ps-anneal-alerts', {'horizon_s': 20.0}),
    'oracle values': ('ps-anneal-oracle-value', {}),
}


def make_policy(name, kw):
    from fogids.smokescreen import POLICIES as RULES
    from fogids.smokescreen_qubo import AlertPSAnneal, PSAnneal, ValueGreedyAlerts
    if name == 'ps-anneal-alerts':
        return AlertPSAnneal(floor=3.0, **kw)
    if name == 'ps-anneal':
        return PSAnneal(**kw)
    if name == 'ps-anneal-oracle-value':
        return PSAnneal(oracle_value=True, **kw)
    if name == 'value-greedy-alerts':
        return ValueGreedyAlerts(**kw)
    if name == 'ps-anneal-true-a':
        from fogids.learned import TrueAttackPSAnneal
        return TrueAttackPSAnneal(floor=3.0, **kw)
    if name == 'ps-anneal-learned':  # trained on every condition (scripts/train_values.py)
        from fogids.learned import LearnedPSAnneal
        return LearnedPSAnneal(ROOT / 'artifacts/models/value-gbm.pkl', **kw)
    if name in ('sa-toca', 'fog-qiea', 'qiea-ps'):
        from fogids.baselines import QIEAPolicy, SplitPolicy
        if name == 'sa-toca':
            return SplitPolicy(path=ROOT / 'configs/sa-toca-ids-place-smokescreen-x3.json')
        return QIEAPolicy(model='ps' if name == 'qiea-ps' else 'additive', **kw)
    return RULES[name](**kw)


def run(job):
    group, label, variant, policy, kw, seed = job
    from fogids.smokescreen import Engine, generate
    c = json.loads(BASE.read_text(encoding='utf-8'))
    c = copy.deepcopy(c)
    VARIANTS[variant](c)
    c['seed'] = seed
    sc = generate(c)
    if policy == 'sa-toca-own':  # tuned offline on this variant (scripts/tune_satoca.py --variant)
        from fogids.baselines import SplitPolicy
        p = SplitPolicy(path=ROOT / f'configs/sa-toca-x3-{variant}.json')
    elif policy == 'ps-anneal-learned-loo':  # trained without this condition
        from fogids.learned import LearnedPSAnneal
        p = LearnedPSAnneal(ROOT / f'artifacts/models/value-gbm-without-{variant}.pkl', **kw)
    else:
        p = make_policy(policy, kw)
    started = time.perf_counter()
    r = Engine(sc).run(p)
    stats = getattr(p, 'stats', {})
    return {'group': group, 'label': label, 'variant': variant, 'policy': policy, 'seed': seed,
            'objective': r['objective'], 'missed_quiet': r['missed_quiet'], 'missed_flood': r['missed_flood'],
            'false_alert_rate': r['false_alert_rate'], 'dropped': r['dropped'], 'late': r['late'],
            'decision_ms': 1000 * stats['decision_s'] / stats['epochs'] if stats.get('epochs') else None,
            'tasks': len(sc.tasks), 'seconds': time.perf_counter() - started}


DEFAULT_VARIANTS = ('base', 'load-x2', 'load-x5', 'extract-1us', 'extract-3us', 'separation-0',
                    'separation-5', 'large')


def jobs(seeds, variants=DEFAULT_VARIANTS, policies=POLICIES, ablations=True):
    out = []
    for variant in variants:
        for policy in policies:
            out += [('variant', variant, variant, policy, {}, s) for s in seeds]
    if ablations:
        for label, (policy, kw) in ABLATIONS.items():
            out += [('ablation', label, 'base', policy, kw, s) for s in seeds]
    return out


def key(r):
    return (r['group'], r['label'], r['policy'], r['seed'])


def summary(rows):
    T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

    def cell(v):
        if len(v) < 2:
            return f'{v[0]:.3f}' if v else '—'
        h = T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / len(v) ** 0.5
        return f'{statistics.fmean(v):.3f} ± {h:.3f}'

    by = {}
    for r in rows:
        by.setdefault((r['group'], r['label'], r['policy']), []).append(r)
    lines = ['## Variants (one parameter changed from x3)', '',
             '| Variant | ' + ' | '.join(POLICIES) + ' | ps-anneal-alerts − value-greedy-alerts (paired) |',
             '|---|' + '---|' * (len(POLICIES) + 1)]
    for variant in [v for v in VARIANTS if any(k[0] == 'variant' and k[1] == v for k in by)]:
        cells = [cell([r['objective'] for r in by.get(('variant', variant, p), [])]) for p in POLICIES]
        a = {r['seed']: r['objective'] for r in by.get(('variant', variant, 'ps-anneal-alerts'), [])}
        b = {r['seed']: r['objective'] for r in by.get(('variant', variant, 'value-greedy-alerts'), [])}
        d = [a[s] - b[s] for s in a if s in b]
        lines.append(f'| {variant} | ' + ' | '.join(cells) + f' | {cell(d)} |')
    lines += ['', '## Ablations of ps-anneal-alerts (x3)', '',
              '| Change | Objective | Missed low-rate | Missed flood | Decision, ms |', '|---|---|---|---|---|']
    base = by.get(('variant', 'base', 'ps-anneal-alerts'), [])
    lines.append(f"| none | {cell([r['objective'] for r in base])} | "
                 f"{cell([r['missed_quiet'] for r in base])} | {cell([r['missed_flood'] for r in base])} | "
                 f"{cell([r['decision_ms'] for r in base if r['decision_ms']])} |")
    for label, (policy, _) in ABLATIONS.items():
        rs = by.get(('ablation', label, policy), [])
        lines.append(f"| {label} | {cell([r['objective'] for r in rs])} | {cell([r['missed_quiet'] for r in rs])} | "
                     f"{cell([r['missed_flood'] for r in rs])} | {cell([r['decision_ms'] for r in rs if r['decision_ms']])} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/robustness')
    parser.add_argument('--only', choices=('variants', 'ablations'), default=None)
    parser.add_argument('--variants', nargs='+', default=list(DEFAULT_VARIANTS), choices=sorted(VARIANTS))
    parser.add_argument('--policies', nargs='+', default=list(POLICIES))
    parser.add_argument('--no-ablations', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'runs.jsonl'
    rows = []
    if args.resume and path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    done = {key(r) for r in rows}
    todo = [j for j in jobs(args.seeds, args.variants, args.policies, not args.no_ablations)
            if (j[0], j[1], j[3], j[5]) not in done]
    if args.only:
        todo = [j for j in todo if j[0] == args.only[:-1]]
    print(f'{len(done)} runs kept, {len(todo)} to do', flush=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        f.flush()
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run, j) for j in todo]
            for k, future in enumerate(as_completed(futures), 1):
                r = future.result()
                rows.append(r)
                f.write(json.dumps(r) + '\n')
                f.flush()
                os.fsync(f.fileno())
                print(f"[{k}/{len(todo)}] {r['label']:22s} {r['policy']:22s} seed {r['seed']} "
                      f"objective {r['objective']:.4f} ({r['seconds']:.0f} s)", flush=True)
    table = summary(rows)
    (args.output / 'summary.md').write_text(table + '\n', encoding='utf-8')
    print(table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
