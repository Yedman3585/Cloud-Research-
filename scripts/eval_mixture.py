"""Evaluate the oracle-routed DFL mixture against alert feedback on the scenario families.

For each family and evaluation seed (2-6, disjoint from the training seeds 7-12) the
engine runs several policies on the same instance:
  alerts       static alert formula (ps-anneal-alerts)
  adaptive     alert feedback, online fpr + change detection (ps-anneal-adaptive)
  dfl2-all     DFL value weights trained on all five families (in-distribution)
  dfl2-loo     DFL trained on the other four families (transfer / out-of-distribution)
  expert       DFL trained only on this family (oracle routing: perfect classifier)
  oracle-value exact ground-truth values in the same annealer (upper bound)
All DFL policies share the same v2 attack-flow estimator (value-gbm-v2.pkl); only the
value weights differ, so the comparison isolates the weight specialisation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODELS = ROOT / 'artifacts/models'
BASE = ROOT / 'configs/ids-place-smokescreen-x3.json'
FAMILIES_ORDER = ['f1-detectors', 'f2-botnet', 'f3-evasive', 'f4-surges', 'f5-infra']


def build(policy, family):
    gbm = MODELS / 'value-gbm-v2.pkl'
    if policy == 'alerts':
        from fogids.smokescreen_qubo import AlertPSAnneal
        return AlertPSAnneal(floor=3.0)
    if policy == 'adaptive':
        from fogids.adaptive import AdaptiveAlertPSAnneal
        return AdaptiveAlertPSAnneal(floor=3.0)
    if policy == 'oracle-value':
        from fogids.smokescreen_qubo import PSAnneal
        return PSAnneal(oracle_value=True)
    if policy == 'true-a':
        from fogids.learned import TrueAttackPSAnneal
        return TrueAttackPSAnneal(floor=3.0)
    from fogids.dfl import DFLPSAnneal
    if policy == 'dfl2-all':
        return DFLPSAnneal(gbm, MODELS / 'dfl2-all.json')
    if policy == 'dfl2-loo':
        return DFLPSAnneal(gbm, MODELS / f'dfl2-without-{family}.json')
    if policy == 'expert':
        return DFLPSAnneal(gbm, MODELS / f'dfl2-{family}.json')
    raise ValueError(policy)


def run(job):
    family, policy, seed = job
    from fogids.smokescreen import Engine, generate
    from fogids.families import FAMILIES
    c = copy.deepcopy(json.loads(BASE.read_text(encoding='utf-8')))
    c['seed'] = seed
    FAMILIES[family](c)
    sc = generate(c)
    p = build(policy, family)
    r = Engine(sc).run(p)
    return {'family': family, 'policy': policy, 'seed': seed, 'objective': r['objective'],
            'missed_quiet': r['missed_quiet'], 'missed_flood': r['missed_flood'],
            'false_alert_rate': r['false_alert_rate']}


def cell(v):
    T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
    if not v:
        return '—'
    if len(v) < 2:
        return f'{v[0]:.3f}'
    h = T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / len(v) ** 0.5
    return f'{statistics.fmean(v):.3f} ± {h:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='+', default=FAMILIES_ORDER)
    ap.add_argument('--policies', nargs='+',
                    default=['alerts', 'adaptive', 'dfl2-all', 'dfl2-loo', 'expert', 'oracle-value'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/mixture')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(f, p, s) for f in args.families for p in args.policies for s in args.seeds]
    rows = []
    path = args.output / 'runs.jsonl'
    with path.open('w', encoding='utf-8') as fh, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print(f"[{k}/{len(jobs)}] {r['family']:13s} {r['policy']:12s} seed {r['seed']} "
                  f"obj {r['objective']:.4f}", flush=True)
    by = {}
    for r in rows:
        by.setdefault((r['family'], r['policy']), []).append(r['objective'])
    pol = args.policies
    lines = ['# Objective (lower is better), mean +/- 95% CI over seeds ' + str(args.seeds), '',
             '| Family | ' + ' | '.join(pol) + ' |', '|---|' + '---|' * len(pol)]
    for f in args.families:
        lines.append(f'| {f} | ' + ' | '.join(cell(by.get((f, p), [])) for p in pol) + ' |')
    table = '\n'.join(lines)
    (args.output / 'summary.md').write_text(table + '\n', encoding='utf-8')
    print('\n' + table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
