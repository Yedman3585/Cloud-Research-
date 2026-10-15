"""Screen the smokescreen scenario (src/fogids/smokescreen.py): reference policies in
the exact processor-sharing simulator and the fluid lower bound.

    python scripts/screen_smokescreen.py --config configs/ids-place-smokescreen-x3.json \
        --seeds 1 --bound gateway
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fogids.generator import load_config  # noqa: E402
from fogids.smokescreen import POLICIES, Engine, fluid_bound, generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--policies', nargs='+', default=sorted(POLICIES))
    parser.add_argument('--bound', choices=('fog', 'gateway', 'none'), default='fog',
                        help="LP scope: 'fog' is tighter, 'gateway' needs far less memory")
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/smokescreen')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runs.jsonl').open('a', encoding='utf-8') as f:
        for seed in args.seeds:
            config = load_config(args.config)
            config['seed'] = seed
            sc = generate(config)
            runs = [(name, lambda n=name: Engine(sc).run(POLICIES[n]())) for name in args.policies]
            if args.bound != 'none':
                runs.append((f'bound-{args.bound}', lambda: fluid_bound(sc, scope=args.bound)))
            for name, fn in runs:
                started = time.perf_counter()
                r = fn()
                row = {'config': config['name'], 'seed': seed, 'policy': name, 'seconds': time.perf_counter() - started,
                       **{k: (float(v) if not isinstance(v, dict) else {c: float(x) for c, x in v.items()})
                          for k, v in r.items()}}
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(f"{config['name']} seed {seed} {name:22s} objective {r['objective']:.4f} "
                      f"missed quiet {r['missed_quiet']:.3f} flood {r['missed_flood']:.3f} "
                      f"false alerts {r['false_alert_rate']:.4f}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
