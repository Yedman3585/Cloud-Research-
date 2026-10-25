"""Train the attack-flow regressor of ``ps-anneal-learned`` (two-stage: fit to ground truth).

Reads the task files of ``scripts/collect_values.py`` (artifacts/value-data/ is x3,
artifacts/value-data/<variant>/ the other conditions). Target: the attack flows of the
task (low-rate flows only in a residual task, see ``target``). Model: gradient-boosted trees with Poisson loss. The last two training seeds of
every condition are held out for the report; the saved model is refitted on all of them.

    python scripts/train_values.py               # one model on every condition
    python scripts/train_values.py --loo         # also one model without each condition

Output: artifacts/models/value-gbm.pkl and value-gbm-without-<variant>.pkl.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load(data):
    from fogids.learned import feature_row
    rows = []
    for path in sorted(data.rglob('seed-*.tasks.jsonl.gz')):
        variant = 'base' if path.parent == data else path.parent.name
        seed = int(path.name.split('-')[1].split('.')[0])
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                if 'prefilter_score' not in r:  # never offered to the scheduler
                    continue
                rows.append((variant, seed, feature_row(r), target(r), r))
    return rows


def target(r):
    """Attack flows the value formula stands for: low-rate flows in a residual task (flood
    flows leaked there by the header pass are left out), attack flows in a flood task."""
    from fogids.smokescreen import FLOOD
    if r['stream'] == 'flood':
        return float(sum(r['attacks'].values()))
    return float(sum(a for c, a in r['attacks'].items() if c not in FLOOD))


def formula(r):
    """Attack-flow estimate of ps-anneal-alerts (floor 3, flood minimum 5)."""
    if r['stream'] == 'flood':
        return max(5.0, r['n_flows'] * r['prefilter_score'])
    return max(3.0, r['share_40'] * r['n_flows'])


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss='poisson', max_iter=400, learning_rate=0.08,
                                         max_leaf_nodes=31, min_samples_leaf=40, random_state=0).fit(X, y)


def report(rows, model):
    from scipy.stats import spearmanr
    import numpy as np
    out = {}
    groups = [(v, s) for v in sorted({r[0] for r in rows}) for s in ('flood', 'residual')]
    for variant, stream in groups:
        sub = [r for r in rows if r[0] == variant and r[4]['stream'] == stream]
        variant = f'{variant}/{stream}'
        if len(sub) < 2:
            continue
        y = np.array([r[3] for r in sub])
        pred = model.predict([r[2] for r in sub])
        base = np.array([formula(r[4]) for r in sub])
        out[variant] = {'tasks': len(sub),
                        'mae_model': float(np.mean(np.abs(pred - y))), 'mae_formula': float(np.mean(np.abs(base - y))),
                        'spearman_model': float(spearmanr(pred, y).statistic),
                        'spearman_formula': float(spearmanr(base, y).statistic)}
    return out


def train(rows, exclude, out_path):
    from fogids.learned import FEATURES
    use = [r for r in rows if r[0] != exclude]
    seeds = {}
    for r in use:
        seeds.setdefault(r[0], set()).add(r[1])
    held = {v: set(sorted(s)[-2:] if len(s) > 2 else sorted(s)[1:]) for v, s in seeds.items()}
    train_rows = [r for r in use if r[1] not in held[r[0]]]
    test_rows = [r for r in use if r[1] in held[r[0]]] + [r for r in rows if r[0] == exclude]
    model = fit([r[2] for r in train_rows], [r[3] for r in train_rows])
    rep = report(test_rows, model)
    final = fit([r[2] for r in use], [r[3] for r in use])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump({'model': final, 'features': FEATURES, 'trained_on': sorted(seeds)}, fh)
    print(f'\n{out_path.name}: trained on {", ".join(sorted(seeds))} ({len(use)} tasks)')
    print(f'{"held-out":20s} {"tasks":>7s} {"MAE model":>10s} {"MAE formula":>12s} {"rho model":>10s} {"rho formula":>12s}')
    for v, s in rep.items():
        mark = '  (unseen condition)' if v.split('/')[0] == exclude else ''
        print(f'{v:20s} {s["tasks"]:7d} {s["mae_model"]:10.2f} {s["mae_formula"]:12.2f} '
              f'{s["spearman_model"]:10.3f} {s["spearman_formula"]:12.3f}{mark}')
    return rep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts/value-data')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/models')
    parser.add_argument('--loo', action='store_true', help='also train one model without each condition')
    args = parser.parse_args()
    rows = load(args.data)
    variants = sorted({r[0] for r in rows})
    print(f'{len(rows)} tasks from {", ".join(variants)}', flush=True)
    reports = {'all': train(rows, None, args.output / 'value-gbm.pkl')}
    if args.loo:
        for v in variants:
            reports[f'without-{v}'] = train(rows, v, args.output / f'value-gbm-without-{v}.pkl')
    (args.output / 'report.json').write_text(json.dumps(reports, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
