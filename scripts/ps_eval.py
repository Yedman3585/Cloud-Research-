"""Evaluate the processor-sharing model (src/fogids/ps_model.py) and ps-greedy.

For every (config, seed): the fluid lower bound, risk-split and queue-greedy (with
their decisions replayed through PSModel, so both latency estimators are scored on
identical decisions), ps-greedy without and with deadline slack, and the epoch QUBO (qubo-ps). Writes one JSON line
per run and a Markdown table of means with 95 % Student-t half-widths.

    python scripts/ps_eval.py --seeds 2 3 4 5 6 \
        --config configs/ids-place-medium-calibrated.json \
        --config configs/ids-place-medium-calibrated-x3.json \
        --config configs/ids-place-medium-calibrated-x10.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
POLICIES = ('risk-split', 'queue-greedy', 'ps-greedy', 'ps-greedy-s0.1')


def run(job):
    config_path, seed, slack, gamma = job
    from fogids.bounds import fluid_bound
    from fogids.bridge import simulate
    from fogids.generator import generate, load_config
    from fogids.metrics import summarize
    from fogids.oracle import fidelity
    from fogids.policies import by_name
    from fogids.ps_policies import PSGreedy, Shadow
    from fogids.qubo_place import QuboPlace
    config = load_config(config_path)
    config['seed'] = seed
    instance = generate(config)
    base = {'config': config['name'], 'seed': seed}
    bound = fluid_bound(instance)
    rows = [{**base, 'policy': 'fluid-bound', 'miss': bound['deadline_miss_rate'], 'fa': bound['false_alert_rate']}]
    for name, policy in (('risk-split', Shadow(by_name('risk-split'))), ('queue-greedy', Shadow(by_name('queue-greedy'))),
                         ('ps-greedy', PSGreedy(0.0)), (f'ps-greedy-s{slack:g}', PSGreedy(slack)),
                         ('qubo-ps', QuboPlace(gamma=gamma, slack=slack))):
        end = simulate(instance, policy, timeout_s=3600)
        s = summarize(instance, end)
        row = {**base, 'policy': name, 'miss': s['all']['deadline_miss_rate'], 'fa': s['false_alert_rate'],
               'macro': s['missed_attack_macro'], 'decision_ms': 1000 * s['decision_mean_s']}
        if hasattr(policy, 'stats') and policy.stats.get('tasks'):
            row['repaired'] = policy.stats['repaired'] / policy.stats['tasks']
            row['variables_per_epoch'] = policy.stats['variables'] / policy.stats['epochs']
        if isinstance(policy, Shadow):  # the policy's own estimate on its own decisions
            own = fidelity(instance, policy.inner.predicted, end)
            row.update(own_spearman=own.get('spearman'), own_mae_s=own.get('mae_s'),
                       own_broken=own.get('broken_promises'))
        ps = fidelity(instance, policy.predicted, end)
        row.update(ps_spearman=ps.get('spearman'), ps_mae_s=ps.get('mae_s'), ps_broken=ps.get('broken_promises'))
        rows.append(row)
    return rows


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['config'], r['policy']), []).append(r)

    def cell(values, digits=4):
        values = [v for v in values if v is not None]
        if not values:
            return '—'
        mean = statistics.fmean(values)
        half = T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        return f'{mean:.{digits}f} ± {half:.{digits}f}'

    lines = ['| Config | Policy | Miss rate | False alerts | Estimate Spearman (own / PS) | MAE, ms (own / PS) | Decision, ms |',
             '|---|---|---|---|---|---|---|']
    for (config, policy), rs in groups.items():
        own = cell([r.get('own_spearman') for r in rs], 2)
        ps = cell([r.get('ps_spearman') for r in rs], 2)
        mae_own = cell([1000 * r['own_mae_s'] for r in rs if r.get('own_mae_s') is not None], 0)
        mae_ps = cell([1000 * r['ps_mae_s'] for r in rs if r.get('ps_mae_s') is not None], 0)
        lines.append(f"| {config} | {policy} | {cell([r['miss'] for r in rs])} | {cell([r['fa'] for r in rs])} | "
                     f"{own} / {ps} | {mae_own} / {mae_ps} | {cell([r.get('decision_ms') for r in rs], 1)} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, action='append', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--slack', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=10.0, help='Latency weight of the epoch QUBO')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ps-eval')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(str(c), s, args.slack, args.gamma) for c in args.config for s in args.seeds]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool, (args.output / 'runs.jsonl').open('w', encoding='utf-8') as f:
        for result in pool.map(run, jobs):
            for row in result:
                rows.append(row)
                f.write(json.dumps(row) + '\n')
            print(f"{result[0]['config']} seed {result[0]['seed']} done", flush=True)
    table = summary(rows)
    (args.output / 'summary.md').write_text(table + '\n', encoding='utf-8')
    print(table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
