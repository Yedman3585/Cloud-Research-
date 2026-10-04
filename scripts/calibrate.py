"""Calibrate IDS-PLACE model parameters on CICIoT2023 (milestone M4).

Pipeline (each step can be run on its own; ``all`` runs them in order):

  prepare  stream the CICIoT2023 CSV files and keep a capped, seeded sample per
           original label, grouped into the seven IDS-PLACE attack classes + benign;
  train    train the ``light`` (MLP) and ``full`` (CNN-LSTM) IDS models, record
           per-class detection recall on a held-out test split;
  time     measure single-thread CPU inference time of one window as a function
           of the number of flows, plus cold-start time and memory, in a fresh
           process per model; fit time = fixed + per_flow * flows;
  configs  convert the measurements into a hardware profile
           (configs/calibration-<profile>.json) and write calibrated generator
           configs (configs/ids-place-*-calibrated.json); refresh the results
           section of docs/calibration.md.

Conversion to simulator units: work [MI] = seconds on the reference core x
``ref_mips``, where ``ref_mips`` is the MIPS rating assigned to one core of the
measuring machine (default: the fog-node MIPS of the base configs, i.e. a fog node
is modelled as one such core). Recall is *detection* recall: the fraction of
flows of an attack class that the model flags as any attack, which is what the
simulator's missed-detection metric uses.

Requires the optional dependencies: ``pip install -e '.[calibration]'``.
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
WORK = ROOT / 'artifacts' / 'calibration'
BASE_CONFIGS = ('ids-place-small', 'ids-place-medium', 'ids-place-large')
REPORT = ROOT / 'docs' / 'calibration.md'
REPORT_BEGIN, REPORT_END = '<!-- results:begin -->', '<!-- results:end -->'

BENIGN = 'benign'
CLASSES = (BENIGN, 'ddos', 'dos', 'mirai', 'recon', 'spoofing', 'web', 'bruteforce')
ATTACKS = CLASSES[1:]
_EXACT = {
    'BenignTraffic': BENIGN,
    'VulnerabilityScan': 'recon',
    'DNS_Spoofing': 'spoofing',
    'MITM-ArpSpoofing': 'spoofing',
    'DictionaryBruteForce': 'bruteforce',
    'SqlInjection': 'web', 'XSS': 'web', 'CommandInjection': 'web',
    'BrowserHijacking': 'web', 'Backdoor_Malware': 'web', 'Uploading_Attack': 'web',
}
_PREFIX = (('DDoS', 'ddos'), ('DoS', 'dos'), ('Mirai', 'mirai'), ('Recon', 'recon'))
TIMING_FLOWS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


# Pure helpers (no third-party imports) -------------------------------------

def label_group(label):
    """Map an original CICIoT2023 label to an IDS-PLACE class (the 8-class grouping)."""
    label = label.strip()
    if label.startswith('Benign'):
        return BENIGN
    if label in _EXACT:
        return _EXACT[label]
    for prefix, group in _PREFIX:
        if label.startswith(prefix + '-'):
            return group
    raise ValueError(f'Unknown CICIoT2023 label {label!r}')


def fit_linear(xs, ys):
    """Least-squares fit y = a + b x; returns (a, b, r2). Intercept and slope clipped at 0."""
    n = len(xs)
    if n < 2:
        raise ValueError('Need at least two points')
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if a < 0:  # refit through the origin
        a, b = 0.0, sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    b = max(b, 0.0)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def calibrated_config(base, profile, n_features):
    """Copy a generator config and replace model costs and recalls by a profile."""
    out = json.loads(json.dumps(base))
    out['name'] = base['name'] + '-calibrated'
    out['calibrated'] = True
    out['_comment'] = (f"Model costs and recalls calibrated on CICIoT2023, hardware profile "
                       f"'{profile['profile']}' (configs/calibration-{profile['profile']}.json).")
    out['calibration'] = {'profile': profile['profile'], 'ref_mips': profile['ref_mips']}
    by_name = {m['name']: m for m in profile['models']}
    for m in out['models']:
        p = by_name[m['name']]
        m.update(fixed_mi=p['fixed_mi'], mi_per_flow=p['mi_per_flow'], memory_mb=p['memory_mb'],
                 load_s=p['load_s'], recall={c: p['recall'][c] for c in ATTACKS})
    out['traffic']['bytes_per_flow'] = 4 * n_features  # one float32 feature vector per flow
    return out


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# Step 1: sample --------------------------------------------------------------

def file_label(path):
    """Label of a per-attack CSV file of the 2024 release, taken from its folder name."""
    name = Path(path).parent.name
    return 'BenignTraffic' if name.lower().startswith('benign') else name


def prepare(raw, out, cap, seed, drop=()):
    """Sample at most ``cap`` rows per original label.

    Two layouts are supported: the original release (shuffled ``part-*.csv`` files with a
    ``label`` column) and the 2024 release (one folder per label, ``*.pcap.csv`` files
    without a label column). In the second case each file of a label contributes a seeded
    random subset of ceil(cap / files) rows, so that every capture file is represented.
    """
    import numpy as np
    import pandas as pd
    files = sorted(Path(raw).rglob('*.csv'))
    if not files:
        raise SystemExit(f'No CSV files under {raw}')
    rng = np.random.default_rng(seed)
    header = [c.strip() for c in pd.read_csv(files[0], nrows=0).columns]
    labelled = any(c.lower() == 'label' for c in header)
    columns = [c for c in header if c.lower() != 'label' and c not in drop]
    kept, seen = {}, {}

    def take(label, frame, quota):
        seen[label] = seen.get(label, 0) + len(frame)
        room = min(quota, cap - sum(len(x) for x in kept.get(label, [])))
        if room > 0:
            rows = frame.iloc[rng.permutation(len(frame))[:room]]
            kept.setdefault(label, []).append(rows[columns].to_numpy(np.float64))

    if labelled:
        for i, f in enumerate(files):
            for chunk in pd.read_csv(f, chunksize=200_000):
                chunk.columns = [c.strip() for c in chunk.columns]
                label_col = next(c for c in chunk.columns if c.lower() == 'label')
                for label, rows in chunk.groupby(label_col, sort=True):
                    label_group(label)
                    take(label, rows, cap)
            print(f'[{i + 1}/{len(files)}] {f.name}')
    else:
        by_label = {}
        for f in files:
            by_label.setdefault(file_label(f), []).append(f)
        for label in by_label:
            label_group(label)  # fail early on an unknown folder name
        done = 0
        for label, group in sorted(by_label.items()):
            quota = math.ceil(cap / len(group))
            for f in (group[k] for k in rng.permutation(len(group))):
                frame = pd.read_csv(f)
                frame.columns = [c.strip() for c in frame.columns]
                if [c for c in frame.columns if c not in drop] != columns:
                    raise SystemExit(f'{f}: feature columns differ from {files[0]}')
                take(label, frame, quota)
                done += 1
            print(f'[{done}/{len(files)}] {label}: {sum(len(x) for x in kept.get(label, []))} rows kept '
                  f'of {seen.get(label, 0)}')
    labels = sorted(kept)
    X = np.concatenate([np.concatenate(kept[l]) for l in labels]).astype(np.float32)
    fine = np.concatenate([[l] * sum(len(x) for x in kept[l]) for l in labels])
    y = np.array([CLASSES.index(label_group(l)) for l in fine], dtype=np.int64)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, fine=fine, columns=np.array(columns))
    summary = {'files': len(files), 'layout': 'label column' if labelled else 'folder per label',
               'cap_per_label': cap, 'seed': seed, 'features': len(columns), 'columns': columns,
               'dropped': list(drop), 'rows': int(len(y)), 'seen_per_label': seen,
               'kept_per_label': {l: sum(len(x) for x in kept[l]) for l in labels},
               'kept_per_class': {c: int((y == k).sum()) for k, c in enumerate(CLASSES)}}
    _write_json(Path(out).with_suffix('.json'), summary)
    print(json.dumps(summary['kept_per_class'], indent=2))
    missing = [c for c in CLASSES if summary['kept_per_class'][c] == 0]
    if missing:
        print(f'WARNING: no rows for classes {missing}; download more CSV files.')


# Models ----------------------------------------------------------------------

def build_model(kind, n_features, mean=None, std=None):
    import torch
    from torch import nn

    class Normalize(nn.Module):
        """Signed log1p then standardization; part of the model so timing includes it."""

        def __init__(self):
            super().__init__()
            self.register_buffer('mean', torch.zeros(n_features) if mean is None else torch.as_tensor(mean))
            self.register_buffer('std', torch.ones(n_features) if std is None else torch.as_tensor(std))

        def forward(self, x):
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return (torch.sign(x) * torch.log1p(torch.abs(x)) - self.mean) / self.std

    class Light(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Normalize()
            self.net = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, len(CLASSES)))

        def forward(self, x):
            return self.net(self.norm(x))

    class Full(nn.Module):
        """1D-CNN over the feature vector followed by an LSTM over the pooled positions."""

        def __init__(self):
            super().__init__()
            self.norm = Normalize()
            self.conv = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
            self.lstm = nn.LSTM(64, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(CLASSES)))

        def forward(self, x):
            h = self.conv(self.norm(x).unsqueeze(1)).transpose(1, 2)
            _, (last, _) = self.lstm(h)
            return self.head(last[-1])

    return {'light': Light, 'full': Full}[kind]()


def _load_model(path):
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt['kind'], ckpt['n_features'])
    model.load_state_dict(ckpt['state'])
    return model.eval()


def _split(y, seed, fractions=(0.7, 0.15)):
    """Stratified train/val/test index split."""
    import numpy as np
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        a = int(round(len(idx) * fractions[0]))
        b = a + int(round(len(idx) * fractions[1]))
        for part, sl in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            part.append(sl)
    return [rng.permutation(np.concatenate(p)) for p in parts]


def _evaluate(model, X, y, device):
    import numpy as np
    import torch
    preds = []
    with torch.inference_mode():
        for i in range(0, len(X), 8192):
            preds.append(model(torch.as_tensor(X[i:i + 8192], device=device)).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    out = {'class_recall': {}, 'recall': {}}
    for k, c in enumerate(CLASSES):
        mask = y == k
        if mask.any():
            out['class_recall'][c] = float((p[mask] == k).mean())
            if c != BENIGN:
                out['recall'][c] = float((p[mask] != 0).mean())  # detected as any attack
    benign = y == 0
    out['benign_false_positive_rate'] = float((p[benign] != 0).mean()) if benign.any() else None
    out['balanced_accuracy'] = float(np.mean(list(out['class_recall'].values())))
    out['accuracy'] = float((p == y).mean())
    return out


def train(data, kind, seed, epochs, out_dir):
    import numpy as np
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = np.load(data, allow_pickle=False)
    X, y = d['X'], d['y']
    tr, va, te = _split(y, seed)
    Xt = np.nan_to_num(X[tr], nan=0.0, posinf=0.0, neginf=0.0)
    logx = np.sign(Xt) * np.log1p(np.abs(Xt))
    mean, std = logx.mean(0), logx.std(0)
    std[std < 1e-6] = 1.0
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(kind, X.shape[1], mean.astype(np.float32), std.astype(np.float32)).to(device)
    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(CLASSES), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.as_tensor(X[tr])
    ytr = torch.as_tensor(y[tr])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, history = -1.0, None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tr), generator=gen)
        total = 0.0
        for i in range(0, len(order), 1024):
            b = order[i:i + 1024]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b].to(device)), ytr[b].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        val = _evaluate(model, X[va], y[va], device)
        history.append({'epoch': epoch + 1, 'loss': total / len(tr), 'val_balanced_accuracy': val['balanced_accuracy']})
        print(f'{kind} epoch {epoch + 1}/{epochs}: loss {total / len(tr):.4f}, '
              f'val balanced accuracy {val["balanced_accuracy"]:.4f}')
        if val['balanced_accuracy'] > best:
            best = val['balanced_accuracy']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = _evaluate(model.eval(), X[te], y[te], device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'kind': kind, 'n_features': int(X.shape[1]), 'state': best_state}, out_dir / f'{kind}.pt')
    np.save(out_dir / 'timing_inputs.npy', X[te][:max(TIMING_FLOWS)])
    params = sum(p.numel() for p in model.parameters())
    result = {'model': kind, 'seed': seed, 'epochs': epochs, 'device': device, 'parameters': int(params),
              'train_s': time.perf_counter() - started, 'split_sizes': [len(tr), len(va), len(te)],
              'history': history, 'test': test}
    _write_json(out_dir / f'{kind}-metrics.json', result)
    print(json.dumps({'recall': test['recall'], 'benign_fpr': test['benign_false_positive_rate']}, indent=2))


# Step 3: timing (one fresh process per model) ----------------------------------

def _maxrss_mb():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2 ** 20 if sys.platform == 'darwin' else rss / 2 ** 10  # bytes on macOS, KB on Linux


def probe(model_path, inputs_path, min_seconds, min_repeats):
    """Runs in a child process with one CPU thread; prints a JSON line."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    X = torch.as_tensor(np.load(inputs_path))
    rss0 = _maxrss_mb()
    t0 = time.perf_counter()
    model = _load_model(model_path)
    with torch.inference_mode():
        model(X[:1])
    load_s = time.perf_counter() - t0
    points = []
    with torch.inference_mode():
        for n in TIMING_FLOWS:
            x = X[:n]
            for _ in range(10):
                model(x)
            samples, start = [], time.perf_counter()
            while len(samples) < min_repeats or time.perf_counter() - start < min_seconds:
                s = time.perf_counter()
                model(x)
                samples.append(time.perf_counter() - s)
            samples.sort()
            points.append({'flows': n, 'median_s': samples[len(samples) // 2],
                           'p90_s': samples[int(len(samples) * 0.9)], 'repeats': len(samples)})
    weights_mb = sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()]) / 2 ** 20
    print(json.dumps({'load_s': load_s, 'weights_mb': weights_mb, 'rss_increase_mb': _maxrss_mb() - rss0,
                      'memory_mb': max(weights_mb, _maxrss_mb() - rss0), 'points': points}))


def _hardware():
    info = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version()}
    try:
        import torch
        info['torch'] = torch.__version__
    except ImportError:
        pass
    if sys.platform == 'darwin':
        for key in ('machdep.cpu.brand_string', 'hw.model', 'hw.memsize'):
            try:
                info[key] = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True).stdout.strip()
            except OSError:
                pass
    else:
        info['processor'] = platform.processor()
    return info


def measure(work, min_seconds, min_repeats):
    results = {'hardware': _hardware(), 'threads': 1, 'models': {}}
    for kind in ('light', 'full'):
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        out = subprocess.run([sys.executable, __file__, '_probe', str(work / f'{kind}.pt'),
                              str(work / 'timing_inputs.npy'), str(min_seconds), str(min_repeats)],
                             capture_output=True, text=True, env=env, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        a, b, r2 = fit_linear([p['flows'] for p in r['points']], [p['median_s'] for p in r['points']])
        r.update(fixed_s=a, per_flow_s=b, r2=r2)
        results['models'][kind] = r
        print(f'{kind}: fixed {a * 1e6:.1f} us + {b * 1e6:.3f} us/flow (R2 {r2:.4f}), '
              f'load {r["load_s"] * 1e3:.1f} ms, weights {r["weights_mb"]:.3f} MB, '
              f'peak RSS increase {r["rss_increase_mb"]:.1f} MB')
    _write_json(work / 'timing.json', results)


# Step 4: profile, configs, report --------------------------------------------

def write_configs(work, sample_path, profile_name, ref_mips):
    timing = _read_json(work / 'timing.json')
    models = []
    for kind in ('light', 'full'):
        t, m = timing['models'][kind], _read_json(work / f'{kind}-metrics.json')
        models.append({'name': kind, 'fixed_mi': round(t['fixed_s'] * ref_mips, 6),
                       'mi_per_flow': round(t['per_flow_s'] * ref_mips, 6),
                       'memory_mb': max(1, math.ceil(t['memory_mb'])), 'load_s': round(t['load_s'], 4),
                       'recall': {c: round(m['test']['recall'].get(c, 0.0), 4) for c in ATTACKS},
                       'fit': {'fixed_s': t['fixed_s'], 'per_flow_s': t['per_flow_s'], 'r2': t['r2']},
                       'benign_false_positive_rate': m['test']['benign_false_positive_rate'],
                       'balanced_accuracy': m['test']['balanced_accuracy'], 'parameters': m['parameters']})
    sample = _read_json(Path(sample_path).with_suffix('.json'))
    profile = {'profile': profile_name, 'ref_mips': ref_mips, 'hardware': timing['hardware'],
               'threads': timing['threads'], 'dataset': {k: sample[k] for k in
                                                          ('files', 'cap_per_label', 'seed', 'features', 'rows',
                                                           'kept_per_class')},
               'models': models}
    _write_json(ROOT / 'configs' / f'calibration-{profile_name}.json', profile)
    for name in BASE_CONFIGS:
        base = _read_json(ROOT / 'configs' / f'{name}.json')
        _write_json(ROOT / 'configs' / f'{name}-calibrated.json', calibrated_config(base, profile, sample['features']))
    _update_report(profile, timing)
    print(f'Wrote configs/calibration-{profile_name}.json and calibrated configs.')


def _update_report(profile, timing):
    hw = profile['hardware']
    cpu = hw.get('machdep.cpu.brand_string') or hw.get('processor') or hw['machine']
    lines = [f"Profile `{profile['profile']}`: {cpu}, {hw['platform']}, Python {hw['python']}, "
             f"PyTorch {hw.get('torch', '?')}, one CPU thread, reference {profile['ref_mips']} MIPS.",
             f"Sample: {profile['dataset']['rows']} rows from {profile['dataset']['files']} CSV files, "
             f"at most {profile['dataset']['cap_per_label']} per original label.", '',
             '| Model | Params | fixed, µs | per flow, µs | R² | fixed_mi | mi_per_flow | load_s | memory_mb | '
             'balanced acc. | benign FPR |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in profile['models']:
        lines.append(f"| {m['name']} | {m['parameters']} | {m['fit']['fixed_s'] * 1e6:.1f} | "
                     f"{m['fit']['per_flow_s'] * 1e6:.3f} | {m['fit']['r2']:.4f} | {m['fixed_mi']} | "
                     f"{m['mi_per_flow']} | {m['load_s']} | {m['memory_mb']} | {m['balanced_accuracy']:.4f} | "
                     f"{m['benign_false_positive_rate']:.4f} |")
    lines += ['', 'Detection recall per class (test split):', '',
              '| Model | ' + ' | '.join(ATTACKS) + ' |', '|---|' + '---|' * len(ATTACKS)]
    for m in profile['models']:
        lines.append(f"| {m['name']} | " + ' | '.join(f"{m['recall'][c]:.4f}" for c in ATTACKS) + ' |')
    lines += ['', 'Rows per class in the sample: ' +
              ', '.join(f'{c} {n}' for c, n in profile['dataset']['kept_per_class'].items()) + '.']
    text = REPORT.read_text(encoding='utf-8')
    head, rest = text.split(REPORT_BEGIN, 1)
    _, tail = rest.split(REPORT_END, 1)
    REPORT.write_text(head + REPORT_BEGIN + '\n' + '\n'.join(lines) + '\n' + REPORT_END + tail, encoding='utf-8')


# CLI ------------------------------------------------------------------------

def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == '_probe':
        _, _, model, inputs, min_s, min_r = sys.argv
        return probe(model, inputs, float(min_s), int(min_r))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('prepare', 'train', 'time', 'configs', 'all'))
    p.add_argument('--raw', type=Path, default=DATA / 'CICIoT2023', help='Directory with CICIoT2023 CSV files')
    p.add_argument('--sample', type=Path, default=DATA / 'ciciot2023' / 'sample.npz')
    p.add_argument('--cap', type=int, default=20000, help='Maximum rows kept per original label')
    p.add_argument('--drop', nargs='*', default=[], help='Feature columns to exclude, e.g. Time_To_Live IAT')
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--model', choices=('light', 'full', 'both'), default='both')
    p.add_argument('--epochs-light', type=int, default=15)
    p.add_argument('--epochs-full', type=int, default=10)
    p.add_argument('--min-seconds', type=float, default=0.3, help='Minimum timing duration per window size')
    p.add_argument('--min-repeats', type=int, default=50)
    p.add_argument('--profile', default='mac', help='Hardware profile name')
    p.add_argument('--ref-mips', type=float, default=None,
                   help='MIPS of one measuring core (default: fog MIPS of ids-place-small)')
    a = p.parse_args(argv)
    ref_mips = a.ref_mips or _read_json(ROOT / 'configs' / 'ids-place-small.json')['topology']['fog']['mips']
    steps = ('prepare', 'train', 'time', 'configs') if a.step == 'all' else (a.step,)
    for step in steps:
        if step == 'prepare':
            prepare(a.raw, a.sample, a.cap, a.seed, tuple(a.drop))
        elif step == 'train':
            for kind in (('light', 'full') if a.model == 'both' else (a.model,)):
                train(a.sample, kind, a.seed, a.epochs_light if kind == 'light' else a.epochs_full, WORK)
        elif step == 'time':
            measure(WORK, a.min_seconds, a.min_repeats)
        else:
            write_configs(WORK, a.sample, a.profile, ref_mips)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
