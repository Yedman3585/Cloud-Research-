"""Command-line driver: run one instance under several policies and save the results."""
import csv
import json

from .bridge import simulate
from .metrics import summarize, task_rows
from .policies import by_name

DEFAULT_POLICIES = ['edge-light', 'fog-full', 'cloud-full', 'greedy-finish']


def _decision_time(value):
    return value if value in ('none', 'measured') else float(value)


def run_cli(args, root):
    from .generator import generate, load_config
    from .problem import Instance
    if args.instance:
        instance = Instance.load(args.instance)
    else:
        config = load_config(args.config)
        if args.seed is not None:
            config['seed'] = args.seed
        instance = generate(config)
    output = args.output or root / 'artifacts/runs' / f'{instance.name}-s{instance.seed}'
    output.mkdir(parents=True, exist_ok=True)
    instance.save(output / 'instance.json')
    summaries = []
    for name in args.policy or DEFAULT_POLICIES:
        end = simulate(instance, by_name(name), decision_time=_decision_time(args.decision_time),
                       log_path=output / f'{name}.log')
        rows = task_rows(instance, end)
        with (output / f'{name}-tasks.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries.append(summarize(instance, end))
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n', encoding='utf-8')
    header = f"{'policy':<16}{'miss all':>10}{'miss short':>12}{'p95 s':>9}{'unfinished':>12}{'missed attack':>15}{'cloud cost':>12}"
    print(header)
    for s in summaries:
        p95 = s['all']['latency_p95_s']
        print(f"{s['policy']:<16}{s['all']['deadline_miss_rate']:>10.3f}{s['short_deadline']['deadline_miss_rate']:>12.3f}"
              f"{(p95 if p95 is not None else float('nan')):>9.3f}{s['all']['unfinished']:>12}"
              f"{s['missed_attack_flow_fraction']:>15.4f}{s['cloud_cost']:>12.3f}")
    print(f'Results: {output}')
    return 0
