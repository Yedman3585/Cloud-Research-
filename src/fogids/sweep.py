"""Multi-seed experiments: every (config, seed, policy) combination, aggregated with
confidence intervals across seeds.

Each run generates the instance for a seed, simulates one policy in iFogSim and
keeps the summary metrics. Runs execute in parallel worker processes (each starts
its own JVM). Outputs, in the sweep directory:

* ``runs.csv``      one row per run;
* ``summary.csv``   mean and 95 % confidence half-width per (config, policy);
* ``summary.md``    the same as Markdown tables, one per config;
* ``figures/*.png`` deadline-miss rate against load and a timeliness / false-alert
  trade-off plot (only when matplotlib is installed).
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import statistics
import time

METRICS = [
    ('miss_all', 'Deadline miss rate (all)'),
    ('miss_short', 'Deadline miss rate (short deadline)'),
    ('miss_long', 'Deadline miss rate (long deadline)'),
    ('p95_s', 'p95 latency of completed tasks, s'),
    ('unfinished', 'Unfinished tasks'),
    ('missed_attack', 'Missed attack flows (fraction)'),
    ('missed_macro', 'Missed attack flows (macro over classes)'),
    ('false_alert_rate', 'False-alert rate on benign flows'),
    ('energy_kj', 'Edge and fog energy, kJ'),
    ('cloud_cost', 'Cloud cost'),
    ('decision_ms', 'Mean decision time per epoch, ms'),
]

# Two-sided 95 % Student t quantiles by degrees of freedom; 1.96 beyond the table.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
       18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t95(df):
    if df in T95:
        return T95[df]
    smaller = [d for d in T95 if d < df]
    return T95[max(smaller)] if df <= 30 else 1.96


def run_one(job):
    """Worker: generate, simulate and summarize one (config, seed, policy) combination."""
    from .bridge import simulate
    from .generator import generate, load_config
    from .metrics import summarize
    from .policies import by_name
    config_path, seed, policy, decision_time = job
    config = load_config(config_path)
    config['seed'] = seed
    instance = generate(config)
    started = time.perf_counter()
    end = simulate(instance, by_name(policy), decision_time=decision_time)
    s = summarize(instance, end)
    p95 = s['all']['latency_p95_s']
    return {
        'config': config['name'], 'load_scale': config.get('calibration', {}).get('load_scale', 1),
        'seed': seed, 'policy': policy, 'tasks': s['all']['tasks'],
        'miss_all': s['all']['deadline_miss_rate'],
        'miss_short': s['short_deadline']['deadline_miss_rate'],
        'miss_long': s['long_deadline']['deadline_miss_rate'],
        'p95_s': p95 if p95 is not None else float('nan'),
        'unfinished': s['all']['unfinished'],
        'missed_attack': s['missed_attack_flow_fraction'], 'missed_macro': s['missed_attack_macro'],
        'false_alert_rate': s['false_alert_rate'], 'energy_kj': s['energy_edge_fog_j'] / 1000,
        'cloud_cost': s['cloud_cost'], 'decision_ms': 1000 * s['decision_mean_s'],
        'wall_s': time.perf_counter() - started,
    }


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['config'], r['load_scale'], r['policy']), []).append(r)
    out = []
    for (config, scale, policy), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        entry = {'config': config, 'load_scale': scale, 'policy': policy, 'seeds': len(runs)}
        for key, _ in METRICS:
            values = [r[key] for r in runs if not math.isnan(r[key])]
            mean = statistics.fmean(values) if values else float('nan')
            half = (t95(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
                    if len(values) > 1 else 0.0)
            entry[key], entry[key + '_ci'] = mean, half
        out.append(entry)
    return out


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(summary, policies):
    order = {p: i for i, p in enumerate(policies)}
    columns = [('miss_all', 3), ('miss_short', 3), ('p95_s', 2), ('unfinished', 0),
               ('missed_macro', 4), ('false_alert_rate', 4), ('cloud_cost', 3)]
    lines = []
    for config in sorted({e['config'] for e in summary}):
        entries = sorted((e for e in summary if e['config'] == config), key=lambda e: order.get(e['policy'], 99))
        lines += [f"### {config} ({entries[0]['seeds']} seeds, mean ± 95 % CI)", '',
                  '| Policy | Miss (all) | Miss (short) | p95, s | Unfinished | Missed attack (macro) '
                  '| False alerts | Cloud cost |', '|---|---|---|---|---|---|---|---|']
        for e in entries:
            cells = [f"{e[k]:.{d}f} ± {e[k + '_ci']:.{d}f}" for k, d in columns]
            lines.append(f"| {e['policy']} | " + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


# Fixed policy -> colour and marker, so a policy keeps its identity across figures.
STYLE = {
    'edge-light': ('#2a78d6', 'o'), 'edge-full': ('#eb6834', 's'), 'fog-full': ('#1baf7a', '^'),
    'cloud-full': ('#eda100', 'v'), 'greedy-finish': ('#e87ba4', 'D'), 'edge-adaptive': ('#008300', 'P'),
    'queue-greedy': ('#4a3aa7', 'X'), 'risk-split': ('#e34948', '*'),
}


def figures(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                         'xtick.color': muted, 'ytick.color': muted, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    written = []
    families = {}
    for e in summary:
        families.setdefault(e['config'].split('-calibrated')[0], []).append(e)
    for family, entries in families.items():
        scales = sorted({e['load_scale'] for e in entries})
        if len(scales) > 1:  # deadline misses against load
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for policy, (color, marker) in STYLE.items():
                pts = sorted((e['load_scale'], e['miss_all'], e['miss_all_ci']) for e in entries
                             if e['policy'] == policy)
                if pts:
                    xs, ys, cis = zip(*pts)
                    ax.errorbar(xs, ys, yerr=cis, color=color, marker=marker, markersize=6, linewidth=2,
                                capsize=3, label=policy)
            ax.set_xscale('log')
            ax.set_xticks(scales, [f'x{s:g}' for s in scales])
            ax.set_xlabel('Traffic load scale')
            ax.set_ylabel('Deadline miss rate')
            ax.grid(axis='y', color=grid, linewidth=0.8)
            ax.set_title(f'{family}: deadline misses grow with load', loc='left', fontsize=10, color=ink)
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
            path = out_dir / f'{family}-miss-vs-load.png'
            fig.savefig(path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            written.append(path)
        for scale in scales:  # timeliness against alert quality
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            at_scale = {e['policy']: e for e in entries if e['load_scale'] == scale}
            for policy, (color, marker) in STYLE.items():
                if policy not in at_scale:
                    continue
                e = at_scale[policy]
                ax.errorbar(e['false_alert_rate'], e['miss_all'], xerr=e['false_alert_rate_ci'],
                            yerr=e['miss_all_ci'], color=color, marker=marker, markersize=8, capsize=3,
                            linestyle='none', label=e['policy'])
            ax.set_xlabel('False-alert rate on benign flows (lower is better)')
            ax.set_ylabel('Deadline miss rate (lower is better)')
            ax.grid(color=grid, linewidth=0.8)
            ax.set_title(f'{family} x{scale:g}: timeliness vs. alert quality', loc='left', fontsize=10,
                         color=ink)
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
            path = out_dir / f'{family}-x{scale:g}-tradeoff.png'
            fig.savefig(path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            written.append(path)
    return written


def run_cli(args, root):
    configs = [Path(c) for c in args.config]
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    policies = args.policy
    decision_time = args.decision_time if args.decision_time in ('none', 'measured') else float(args.decision_time)
    out = args.output or root / 'artifacts/sweeps' / args.name
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(c), s, p, decision_time) for c in configs for s in seeds for p in policies]
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            r = rows[-1]
            print(f"[{i}/{len(jobs)}] {r['config']} seed {r['seed']} {r['policy']}: "
                  f"miss {r['miss_all']:.3f}, false alerts {r['false_alert_rate']:.3f} ({r['wall_s']:.1f} s)",
                  flush=True)
    rows.sort(key=lambda r: (r['config'], r['seed'], r['policy']))
    _write_csv(out / 'runs.csv', rows)
    summary = aggregate(rows)
    _write_csv(out / 'summary.csv', summary)
    (out / 'summary.md').write_text(markdown(summary, policies), encoding='utf-8')
    (out / 'sweep.json').write_text(json.dumps({
        'configs': [str(c) for c in configs], 'seeds': seeds, 'policies': policies,
        'decision_time': decision_time, 'wall_s': time.perf_counter() - started}, indent=2) + '\n',
        encoding='utf-8')
    written = figures(summary, out / 'figures')
    print(markdown(summary, policies))
    print(f'Sweep: {out}' + (f' ({len(written)} figures)' if written else ' (install matplotlib for figures)'))
    return 0
