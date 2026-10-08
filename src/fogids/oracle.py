"""Milestone M5: how far are the baselines from the best possible, and how well do
scheduler-side latency estimates predict the simulator?

For each instance this runs:

* the fluid lower bound (`bounds.fluid_bound`), a clairvoyant LP relaxation;
* the LP plan replayed in the simulator (`policies.Planned`), an achievable but
  clairvoyant reference showing how loose the fluid bound is;
* the reference policies, recording the finish time each one predicted for every task,
  and compares predictions with simulated finish times (fidelity).
"""
import csv
import json
import statistics


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation with average ranks for ties; None if undefined."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None


def fidelity(instance, predicted, end):
    """Compare predicted and simulated latencies of completed tasks."""
    pairs = []
    for r in end['results']:
        if r['finish_s'] is None or r['id'] not in predicted:
            continue
        t = instance.tasks[r['id']]
        pairs.append((predicted[r['id']] - t.release_s, r['finish_s'] - t.release_s, t.deadline_s))
    if not pairs:
        return {}
    pred, real, deadline = zip(*pairs)
    pred_meet = [p <= d + 1e-5 for p, d in zip(pred, deadline)]
    real_meet = [r <= d + 1e-5 for r, d in zip(real, deadline)]
    promised = sum(pred_meet)
    return {
        'spearman': spearman(list(pred), list(real)),
        'mae_s': statistics.fmean(abs(p - r) for p, r in zip(pred, real)),
        'mean_error_s': statistics.fmean(r - p for p, r in zip(pred, real)),  # > 0: optimistic
        'deadline_agreement': sum(a == b for a, b in zip(pred_meet, real_meet)) / len(pairs),
        'broken_promises': (sum(p and not r for p, r in zip(pred_meet, real_meet)) / promised) if promised else 0.0,
    }


def analyse(config_path, seed, policies=('risk-split', 'queue-greedy', 'edge-adaptive', 'greedy-finish')):
    from .bounds import fluid_bound
    from .bridge import simulate
    from .generator import generate, load_config
    from .metrics import summarize
    from .policies import Planned, by_name
    config = load_config(config_path)
    config['seed'] = seed
    instance = generate(config)
    bound = fluid_bound(instance)
    plan = summarize(instance, simulate(instance, Planned(bound.pop('plan'))))
    rows = [{'config': config['name'], 'seed': seed, 'policy': 'fluid-bound', 'miss_all': bound['deadline_miss_rate'],
             'false_alert_rate': bound['false_alert_rate'], 'missed_attack': bound['missed_attack']},
            {'config': config['name'], 'seed': seed, 'policy': 'lp-plan', 'miss_all': plan['all']['deadline_miss_rate'],
             'false_alert_rate': plan['false_alert_rate'], 'missed_attack': plan['missed_attack_flow_fraction']}]
    for name in policies:
        policy = by_name(name)
        end = simulate(instance, policy)
        s = summarize(instance, end)
        row = {'config': config['name'], 'seed': seed, 'policy': name, 'miss_all': s['all']['deadline_miss_rate'],
               'false_alert_rate': s['false_alert_rate'], 'missed_attack': s['missed_attack_flow_fraction']}
        row.update(fidelity(instance, getattr(policy, 'predicted', {}), end))
        rows.append(row)
    return rows


def run_cli(args, root):
    out = args.output or root / 'artifacts/oracle' / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in args.config:
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            for row in analyse(config, seed):
                rows.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (out / 'oracle.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results: {out / "oracle.csv"}')
    return 0
