"""Tune the SA-TOCA-style split baseline offline (src/fogids/baselines.py).

    python scripts/tune_satoca.py --seeds 7 --iterations 600 [--variant load-x2]

Simulated annealing over per-class route shares, model and sampling fraction, scored
with the scenario objective on the training seeds (disjoint from the evaluation seeds
2-21). Writes configs/sa-toca-<config name>.json, or configs/sa-toca-x3-<variant>.json
with ``--variant`` (a parameter change of scripts/robustness.py applied to the config).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fogids.baselines import tune_splits  # noqa: E402
from fogids.generator import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ids-place-smokescreen-x3.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 8])
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0, help='seed of the annealer')
    parser.add_argument('--variant', default=None, help='a variant of scripts/robustness.py')
    args = parser.parse_args()
    configs = []
    for s in args.seeds:
        c = load_config(args.config)
        c['seed'] = s  # before the variant: the scenario families draw from it
        if args.variant:
            sys.path.insert(0, str(ROOT / 'scripts'))
            import robustness
            robustness.VARIANTS[args.variant](c)
        configs.append(c)
    params, objective, history = tune_splits(configs, args.iterations, args.seed,
                                             log=lambda m: print(m, flush=True))
    out = ROOT / 'configs' / (f'sa-toca-x3-{args.variant}.json' if args.variant
                              else f"sa-toca-{configs[0]['name']}.json")
    out.write_text(json.dumps({'config': str(args.config.relative_to(ROOT)), 'variant': args.variant, 'training_seeds': args.seeds,
                               'iterations': args.iterations, 'training_objective': objective,
                               'params': params, 'history': history}, indent=1) + '\n', encoding='utf-8')
    print(out, objective)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
