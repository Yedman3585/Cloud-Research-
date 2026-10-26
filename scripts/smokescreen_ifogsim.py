"""Final comparison of the smokescreen scenario in iFogSim.

    python scripts/smokescreen_ifogsim.py --config configs/ids-place-smokescreen-x3.json \
        --seeds 2 3 4 5 6 --policies greedy-v2 value-greedy-alerts ps-anneal-alerts \
        [--variant extract-1us] [--decision-time measured|0.016] [--workers 2] [--resume]

``--variant`` applies one of the parameter changes of ``scripts/robustness.py``;
``--decision-time`` charges the policy's decision time to the simulated clock;
``--resume`` skips (variant, decision time, seed, policy) runs already in runs.jsonl.
Policies outside the fixed list are built by ``scripts/robustness.py``: the ``-own`` and
``-loo`` policies of the condition, and settings as ``name@key=value`` (for example
``ps-anneal-alerts@fa_weight=0.01``).

Each run executes in iFogSim (org.fogids.IdsStaged) with decisions from Python; one
JSON line per run is appended to <output>/runs.jsonl, with the metrics from iFogSim's
finish times and, under "shadow", those of the scheduler's own model of the run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fogids.generator import load_config  # noqa: E402
from fogids.smokescreen import POLICIES, generate  # noqa: E402
from fogids.smokescreen_ifogsim import simulate_ifogsim  # noqa: E402
from fogids.smokescreen_qubo import AlertPSAnneal, PSAnneal, ValueGreedyAlerts  # noqa: E402
from fogids.baselines import QIEAPolicy, SplitPolicy  # noqa: E402

SATOCA = ROOT / 'configs/sa-toca-ids-place-smokescreen-x3.json'

FACTORIES = {
    **{name: cls for name, cls in POLICIES.items()},
    'ps-anneal': lambda: PSAnneal(),
    'ps-anneal-alerts': lambda: AlertPSAnneal(floor=3.0),
    'ps-anneal-alerts-10ms': lambda: AlertPSAnneal(floor=3.0, budget_s=0.01),
    'ps-anneal-oracle-value': lambda: PSAnneal(oracle_value=True),
    'value-greedy-alerts': lambda: ValueGreedyAlerts(),
    'sa-toca': lambda: SplitPolicy(path=SATOCA),
    'fog-qiea': lambda: QIEAPolicy(),
    'qiea-ps': lambda: QIEAPolicy(model='ps'),
}


def _decision_time(text):
    return text if text in ('none', 'measured') else float(text)


def run(job):
    config_path, variant, decision_time, seed, name = job
    sys.path.insert(0, str(ROOT / 'scripts'))
    import robustness
    config = load_config(config_path)
    robustness.VARIANTS[variant](config)
    config['seed'] = seed
    sc = generate(config)
    started = time.perf_counter()
    if name in FACTORIES:
        policy = FACTORIES[name]()
    else:  # e.g. 'sa-toca-own', 'ps-anneal-dfl-loo', 'ps-anneal-alerts@fa_weight=0.01'
        base, kw = robustness.parse_policy(name)
        policy = robustness.policy_for(base, variant, kw)
    r = simulate_ifogsim(sc, policy, decision_time=decision_time)
    return {'config': config['name'], 'variant': variant, 'decision_time': str(decision_time), 'seed': seed,
            'policy': name, 'seconds': time.perf_counter() - started,
            **{k: v for k, v in r.items() if not isinstance(v, dict)},
            'shadow_objective': r['shadow']['objective']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--policies', nargs='+', default=['greedy-v2', 'value-greedy-alerts', 'ps-anneal-alerts'])
    parser.add_argument('--variant', default='base', help='a variant name of scripts/robustness.py')
    parser.add_argument('--decision-time', type=_decision_time, nargs='+', default=['none'],
                        help="'none', 'measured' or seconds; several values run each")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/smokescreen-ifogsim')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'runs.jsonl'
    done = set()
    if args.resume and path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r.get('variant', 'base'), r.get('decision_time', 'none'), r['seed'], r['policy']))
    jobs = [(str(args.config), args.variant, dt, seed, name)
            for dt in args.decision_time for seed in args.seeds for name in args.policies
            if (args.variant, str(dt), seed, name) not in done]
    print(f'{len(done)} runs kept, {len(jobs)} to do', flush=True)
    with path.open('a', encoding='utf-8') as f, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for k, future in enumerate(as_completed(futures), 1):
            row = future.result()
            f.write(json.dumps(row) + '\n')
            f.flush()
            os.fsync(f.fileno())
            print(f"[{k}/{len(jobs)}] {row['variant']} dt={row['decision_time']} seed {row['seed']} "
                  f"{row['policy']:24s} objective {row['objective']:.4f} (shadow {row['shadow_objective']:.4f}) "
                  f"missed low-rate {row['missed_quiet']:.3f} flood {row['missed_flood']:.3f} "
                  f"decision {1000 * row['decision_mean_s']:.0f} ms ({row['seconds']:.0f} s)", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
