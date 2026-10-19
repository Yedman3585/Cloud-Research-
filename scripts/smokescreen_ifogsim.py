"""Final comparison of the smokescreen scenario in iFogSim.

    python scripts/smokescreen_ifogsim.py --config configs/ids-place-smokescreen-x3.json \
        --seeds 2 3 4 5 6 --policies greedy-v2 value-greedy-alerts ps-anneal-alerts

Each run executes in iFogSim (org.fogids.IdsStaged) with decisions from Python; one
JSON line per run is appended to <output>/runs.jsonl, with the metrics from iFogSim's
finish times and, under "shadow", those of the scheduler's own model of the run.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fogids.generator import load_config  # noqa: E402
from fogids.smokescreen import POLICIES, generate  # noqa: E402
from fogids.smokescreen_ifogsim import simulate_ifogsim  # noqa: E402
from fogids.smokescreen_qubo import AlertPSAnneal, PSAnneal, ValueGreedyAlerts  # noqa: E402

FACTORIES = {
    **{name: cls for name, cls in POLICIES.items()},
    'ps-anneal': lambda: PSAnneal(),
    'ps-anneal-alerts': lambda: AlertPSAnneal(floor=3.0),
    'ps-anneal-oracle-value': lambda: PSAnneal(oracle_value=True),
    'value-greedy-alerts': lambda: ValueGreedyAlerts(),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    parser.add_argument('--policies', nargs='+', default=['greedy-v2', 'value-greedy-alerts', 'ps-anneal-alerts'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/smokescreen-ifogsim')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runs.jsonl').open('a', encoding='utf-8') as f:
        for seed in args.seeds:
            config = load_config(args.config)
            config['seed'] = seed
            sc = generate(config)
            for name in args.policies:
                started = time.perf_counter()
                r = simulate_ifogsim(sc, FACTORIES[name]())
                row = {'config': config['name'], 'seed': seed, 'policy': name,
                       'seconds': time.perf_counter() - started,
                       **{k: v for k, v in r.items() if not isinstance(v, dict)},
                       'shadow_objective': r['shadow']['objective']}
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(f"{config['name']} seed {seed} {name:24s} objective {r['objective']:.4f} "
                      f"(shadow {r['shadow']['objective']:.4f}) missed low-rate {r['missed_quiet']:.3f} "
                      f"flood {r['missed_flood']:.3f} finish error {1000 * r['finish_error_mean_s']:.1f} ms",
                      flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
