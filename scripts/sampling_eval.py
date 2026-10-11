"""Detection quality of the calibrated models under flow sampling.

When feature extraction overloads the gateway, an IDS analyses only a fraction f of
the flows (records) of a window. Unsampled flows are not classified: they can be
neither detected nor falsely flagged. A sampled flow is classified as usual, except
that the context of the ``full`` model then consists of the previous *sampled*
records, i.e. every (1/f)-th record of the capture.

This script measures, on the held-out blocks of the calibration sample, the per-class
recall and the benign false-positive rate of both models with the context taken at
stride s = 1/f (s = 1, 2, 4, 10), on one fixed set of target rows (those with
15 * max(s) records before them in their block), with and without context
contamination (rho, see ``calibrate.py``). Per-flow outcomes of a task analysed at
fraction f are then recall * f and false-positive rate * f.

    python scripts/sampling_eval.py
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import calibrate  # noqa: E402

STRIDES = (1, 2, 4, 10)


def windows(X, idx, stride, context, rho, pool, rng):
    """Inputs of the targets ``idx`` with the context at ``stride``."""
    import numpy as np
    offsets = stride * np.arange(1 - context, 1)
    S = X[idx[:, None] + offsets[None, :]].copy()
    if rho > 0:
        mask = rng.random(S.shape[:2]) < rho
        mask[:, -1] = False
        S[mask] = X[rng.choice(pool, int(mask.sum()))]
    return S


def scores(y, p):
    import numpy as np
    out = {'recall': {}}
    for k, c in enumerate(calibrate.CLASSES):
        mask = y == k
        if mask.any() and c != calibrate.BENIGN:
            out['recall'][c] = float((p[mask] != 0).mean())
    out['benign_false_positive_rate'] = float((p[y == 0] != 0).mean())
    out['targets'] = int(len(y))
    return out


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/ciciot2023-sample.npz')
    parser.add_argument('--models', type=Path, default=ROOT / 'artifacts/calibration')
    parser.add_argument('--rho', type=float, nargs='+', default=[0.0, 0.5])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/calibration/sampling.json')
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = np.load(args.data, allow_pickle=False)
    X, y, block = d['X'].astype(np.float32), d['y'], d['block']
    seed = json.loads((args.models / 'full-metrics.json').read_text())['seed']
    in_tr, _, in_te = calibrate._split_blocks(y, block, seed)
    full = calibrate._load_model(args.models / 'full.pt')
    light = calibrate._load_model(args.models / 'light.pt')
    context = full.context
    starts = np.r_[0, np.flatnonzero(np.diff(block)) + 1]
    pos = np.arange(len(block)) - np.repeat(starts, np.diff(np.r_[starts, len(block)]))
    targets = np.flatnonzero(in_te & (pos >= (context - 1) * max(STRIDES)))
    pool = np.flatnonzero(in_tr)
    result = {'seed': seed, 'context': context, 'strides': list(STRIDES), 'targets': int(len(targets)),
              'light': None, 'full': {}}
    with torch.inference_mode():
        p = np.concatenate([light(torch.as_tensor(X[targets[i:i + 8192]])).argmax(1).numpy()
                            for i in range(0, len(targets), 8192)])
        result['light'] = scores(y[targets], p)
        for rho in args.rho:
            for s in STRIDES:
                rng = np.random.default_rng(seed + 1)
                preds = []
                for i in range(0, len(targets), 4096):
                    x = windows(X, targets[i:i + 4096], s, context, rho, pool, rng)
                    preds.append(full(torch.as_tensor(x)).argmax(1).numpy())
                r = scores(y[targets], np.concatenate(preds))
                result['full'][f'rho{rho:g}-stride{s}'] = {'rho': rho, 'stride': s, 'fraction': 1 / s, **r}
                print(f"full rho={rho:g} stride={s}: fpr={r['benign_false_positive_rate']:.4f} "
                      + ' '.join(f'{c}={v:.3f}' for c, v in r['recall'].items()), flush=True)
    print('light: fpr={:.4f} '.format(result['light']['benign_false_positive_rate'])
          + ' '.join(f'{c}={v:.3f}' for c, v in result['light']['recall'].items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
