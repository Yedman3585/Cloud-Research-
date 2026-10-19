"""Which group (benign, flood, low-rate) do the calibrated models report for each
class? Held-out blocks, targets with a full context, context contamination 0.5 for
the context model. Writes artifacts/calibration/alert-groups.json, used to justify how
the smokescreen simulator reports alerts (src/fogids/smokescreen.py, Engine._alert).

    python scripts/alert_groups.py
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import calibrate  # noqa: E402
from sampling_eval import windows  # noqa: E402

GROUP = {c: 'benign' if c == calibrate.BENIGN else 'flood' if c in ('ddos', 'dos', 'mirai') else 'low-rate'
         for c in calibrate.CLASSES}


def main():
    import numpy as np
    import torch
    d = np.load(ROOT / 'data/ciciot2023-sample.npz', allow_pickle=False)
    X, y, block = d['X'].astype(np.float32), d['y'], d['block']
    models = ROOT / 'artifacts/calibration'
    seed = json.loads((models / 'full-metrics.json').read_text())['seed']
    in_tr, _, in_te = calibrate._split_blocks(y, block, seed)
    starts = np.r_[0, np.flatnonzero(np.diff(block)) + 1]
    pos = np.arange(len(block)) - np.repeat(starts, np.diff(np.r_[starts, len(block)]))
    targets = np.flatnonzero(in_te & (pos >= calibrate.CONTEXT - 1))
    pool = np.flatnonzero(in_tr)
    out = {}
    for kind in ('light', 'full'):
        model = calibrate._load_model(models / f'{kind}.pt')
        rng = np.random.default_rng(seed + 1)
        preds = []
        with torch.inference_mode():
            for i in range(0, len(targets), 4096):
                idx = targets[i:i + 4096]
                x = X[idx] if kind == 'light' else windows(X, idx, 1, model.context, 0.5, pool, rng)
                preds.append(model(torch.as_tensor(x)).argmax(1).numpy())
        p = np.concatenate(preds)
        out[kind] = {}
        for k, c in enumerate(calibrate.CLASSES):
            mask = y[targets] == k
            groups = [GROUP[calibrate.CLASSES[j]] for j in p[mask]]
            out[kind][c] = {g: groups.count(g) / len(groups) for g in ('benign', 'flood', 'low-rate')}
    (models / 'alert-groups.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    for kind, rows in out.items():
        print(kind, {c: {g: round(v, 3) for g, v in r.items()} for c, r in rows.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
