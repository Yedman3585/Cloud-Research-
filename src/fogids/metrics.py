"""Evaluation metrics for one simulated run of an IDS-PLACE instance.

Ground-truth labels are used here and only here. Latency is measured from task
release to inference completion. Unfinished tasks count as deadline misses and
are excluded from the latency percentiles, which therefore always come with the
count of unfinished tasks.

Detection quality has two sides. Missed detection: attack flows not flagged,
expected value ``attack_flows * (1 - recall[model][class])``, all of them if the task
never finishes. False alerts: benign flows flagged as attacks, expected value
``benign_flows * false_positive_rate[model]``, none if the task never finishes.
Missed detection is reported per class and as a macro average over the attack
classes present, so that rare, hard classes are not drowned by flood traffic.
"""
import math

TOLERANCE_S = 1e-5


def percentile(values, q):
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def task_rows(instance, end):
    rows = []
    for r in end['results']:
        t = instance.tasks[r['id']]
        finish = r['finish_s']
        latency = None if finish is None else finish - t.release_s
        model = instance.model(r['model'])
        recall = model.recall.get(t.label, 0.0) if t.attack_flows else 0.0
        benign = t.n_flows - t.attack_flows
        rows.append({
            'id': t.id, 'gateway': t.gateway, 'node': r['node'], 'model': r['model'],
            'release_s': t.release_s, 'dispatch_s': r['dispatch_s'], 'finish_s': finish,
            'latency_s': latency, 'deadline_s': t.deadline_s,
            'missed': latency is None or latency > t.deadline_s + TOLERANCE_S,
            'short_deadline': None, 'weight': t.weight, 'label': t.label,
            'attack_flows': t.attack_flows, 'cold_start': r['cold_start'],
            # Expected undetected attack flows; an unfinished task detects nothing in time.
            'missed_attack_flows': t.attack_flows * (1 - (recall if latency is not None else 0.0)),
            'benign_flows': benign,
            'false_alert_flows': benign * model.false_positive_rate if latency is not None else 0.0,
        })
    shortest = min((t.deadline_s for t in instance.tasks), default=0)
    for row in rows:
        row['short_deadline'] = row['deadline_s'] <= shortest
    return rows


def _group(rows):
    latencies = [r['latency_s'] for r in rows if r['latency_s'] is not None]
    return {
        'tasks': len(rows),
        'unfinished': sum(r['latency_s'] is None for r in rows),
        'deadline_miss_rate': sum(r['missed'] for r in rows) / len(rows) if rows else None,
        'latency_p50_s': percentile(latencies, 50),
        'latency_p95_s': percentile(latencies, 95),
        'latency_p99_s': percentile(latencies, 99),
        'latency_mean_s': sum(latencies) / len(latencies) if latencies else None,
    }


def energy_j(instance, rows, duration_s, exclude=()):
    """Linear power model: idle power over the run plus (busy - idle) power while busy.

    The simulated CPU is work-conserving at full speed whenever a node has work, so a
    node's busy time is its executed work (including cold starts) divided by its MIPS.
    """
    busy = {n.name: 0.0 for n in instance.nodes}
    for r in rows:
        node, model = instance.node(r['node']), instance.model(r['model'])
        work = model.work_mi(instance.tasks[r['id']].n_flows) + (model.load_s * node.mips if r['cold_start'] else 0)
        busy[node.name] += work / node.mips
    return sum(n.idle_w * duration_s + (n.busy_w - n.idle_w) * min(busy[n.name], duration_s)
               for n in instance.nodes if n.name not in exclude)


def missed_by_class(rows):
    """Fraction of each attack class's flows that went undetected."""
    totals, missed = {}, {}
    for r in rows:
        if r['attack_flows']:
            totals[r['label']] = totals.get(r['label'], 0) + r['attack_flows']
            missed[r['label']] = missed.get(r['label'], 0.0) + r['missed_attack_flows']
    return {c: missed[c] / totals[c] for c in sorted(totals)}


def summarize(instance, end):
    rows = task_rows(instance, end)
    attack = sum(r['attack_flows'] for r in rows)
    missed_flows = sum(r['missed_attack_flows'] for r in rows)
    benign = sum(r['benign_flows'] for r in rows)
    false_alerts = sum(r['false_alert_flows'] for r in rows)
    by_class = missed_by_class(rows)
    weights = {t.id: t.criticality for t in instance.tasks}
    cloud = {n.name for n in instance.tier('cloud')}
    cloud_mi = sum(instance.model(r['model']).work_mi(instance.tasks[r['id']].n_flows)
                   for r in rows if r['node'] in cloud)
    cost = sum(n.cost_per_gi for n in instance.tier('cloud')) * cloud_mi / 1000
    placement = {}
    for r in rows:
        key = f"{instance.node(r['node']).tier}/{r['model']}"
        placement[key] = placement.get(key, 0) + 1
    decisions = end.get('decision_s', [])
    return {
        'policy': end.get('policy'), 'instance': instance.name, 'seed': instance.seed,
        'decision_time': end.get('decision_time'), 'simulated_end_s': end['t'],
        'all': _group(rows),
        'short_deadline': _group([r for r in rows if r['short_deadline']]),
        'long_deadline': _group([r for r in rows if not r['short_deadline']]),
        'missed_attack_flow_fraction': missed_flows / attack if attack else 0.0,
        'missed_attack_macro': sum(by_class.values()) / len(by_class) if by_class else 0.0,
        'missed_attack_by_class': by_class,
        'false_alert_flows': false_alerts,
        'false_alert_rate': false_alerts / benign if benign else 0.0,
        'weighted_missed_attack_flows': sum(r['missed_attack_flows'] * weights[r['id']] for r in rows),
        'cold_starts': sum(r['cold_start'] for r in rows),
        'placement': dict(sorted(placement.items())),
        'energy_edge_fog_j': energy_j(instance, rows, end['t'], exclude=cloud),
        'cloud_cost': cost,
        'decision_epochs': len(decisions),
        'decision_mean_s': sum(decisions) / len(decisions) if decisions else 0.0,
        'decision_max_s': max(decisions, default=0.0),
    }
