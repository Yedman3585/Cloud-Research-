"""Calibrate IDS-PLACE model parameters on CICIoT2023 (milestone M4).

Pipeline (each step can be run on its own; ``all`` runs them in order):

  prepare  read the CICIoT2023 CSV files and keep a capped, seeded sample of
           contiguous blocks of records per original label, grouped into the seven
           IDS-PLACE attack classes + benign;
  train    train the ``light`` model (MLP on one flow record) and the ``full`` model
           (GRU over the flow and the K-1 records before it at the same capture),
           record per-class detection recall on held-out blocks;
  time     measure single-thread CPU inference time of one window as a function
           of the number of flows, plus cold-start time and memory, in a fresh
           process per model; fit time = fixed + per_flow * flows;
  traffic  measure how intense each traffic class is in the captures (records per
           second, from the packet rate and packets per record of every row);
  configs  convert the measurements into a hardware profile
           (configs/calibration-<profile>.json) and write calibrated generator
           configs (configs/ids-place-*-calibrated.json); refresh the results
           section of docs/calibration.md.

Conversion to simulator units: work [MI] = seconds on the reference core x
``ref_mips``, where ``ref_mips`` is the MIPS rating assigned to one core of the
measuring machine (default: the fog-node MIPS of the base configs, i.e. a fog node
is modelled as one such core). Recall is *detection* recall: the fraction of
flows of an attack class that the model flags as any attack, which is what the
simulator's missed-detection metric uses. For ``full`` the reported recall is
measured with part of the context replaced by unrelated records (``--eval-rho``),
which stands for traffic of other hosts mixed in at the same gateway.

Node speeds: work is measured on one core of the measuring machine; edge and fog
nodes are rated relative to that core with published single/multi-core benchmark
ratios (``HARDWARE``), selectable with ``--edge`` and ``--fog``. Attack bursts in the
calibrated configs get the class intensities measured by ``traffic``, relative to the
benign rate; the absolute per-gateway volume stays a free parameter that is swept
with ``--load-scales`` (configs ``*-calibrated-x<k>.json``).

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
# Speed of a device relative to one performance core of Apple M2, from Geekbench 5
# (cpu-monkey.com): M2 single-core 1874, M2 multi-core 8853, Raspberry Pi 4 B multi-core
# 601, Raspberry Pi 5 multi-core 1635. A proxy for inference speed, not a measurement.
HARDWARE = {
    'm2-core': ('Apple M2, one performance core', 1.0),
    'm2-chip': ('Apple M2, all 8 cores', 8853 / 1874),
    'rpi4': ('Raspberry Pi 4 Model B, 4 x Cortex-A72 1.5 GHz', 601 / 1874),
    'rpi5': ('Raspberry Pi 5, 4 x Cortex-A76 2.4 GHz', 1635 / 1874),
}
BLOCK = 200  # rows per contiguous block; blocks are the unit of sampling and splitting
CONTEXT = 16  # records seen by the full model: the flow itself and the 15 before it


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


def calibrated_config(base, profile, n_features, scale=1.0):
    """Copy a generator config and apply a profile: model costs, recalls and benign
    false-positive rates, node speeds, class-relative burst intensities; ``scale``
    multiplies every traffic rate."""
    out = json.loads(json.dumps(base))
    suffix = '' if scale == 1 else f'-x{scale:g}'
    out['name'] = base['name'] + '-calibrated' + suffix
    out['calibrated'] = True
    out['_comment'] = (f"Model costs and recalls calibrated on CICIoT2023, hardware profile "
                       f"'{profile['profile']}' (configs/calibration-{profile['profile']}.json); "
                       f"traffic volume x{scale:g} of the base config.")
    out['calibration'] = {'profile': profile['profile'], 'ref_mips': profile['ref_mips'], 'load_scale': scale}
    by_name = {m['name']: m for m in profile['models']}
    for m in out['models']:
        p = by_name[m['name']]
        m.update(fixed_mi=p['fixed_mi'], mi_per_flow=p['mi_per_flow'], memory_mb=p['memory_mb'],
                 load_s=p['load_s'], recall={c: p['recall'][c] for c in ATTACKS},
                 false_positive_rate=round(p.get('benign_false_positive_rate',
                                                 m.get('false_positive_rate', 0.0)), 4))
    out['traffic']['bytes_per_flow'] = 4 * n_features  # one float32 feature vector per flow
    for tier, spec in profile.get('tiers', {}).items():
        out['topology'][tier]['mips'] = spec['mips']
    rates = out['traffic']['benign_flows_per_s']
    out['traffic']['benign_flows_per_s'] = ([r * scale for r in rates] if isinstance(rates, list)
                                            else rates * scale)
    ratio = profile.get('traffic', {}).get('class_ratio')
    benign = rates[0] if isinstance(rates, list) else rates
    for burst in out['attacks']:
        burst['flows_per_s'] = round((benign * ratio[burst['class']] if ratio else burst['flows_per_s']) * scale, 1)
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
    """Sample at most about ``cap`` rows per original label as contiguous blocks.

    Two layouts are supported. The 2024 release has one folder per label with
    ``*.pcap.csv`` files in capture order and no label column; each file of a label
    contributes randomly chosen, non-overlapping blocks of ``BLOCK`` consecutive rows
    (about ceil(cap / files) rows), so the ``full`` model can see real context. The
    original release (shuffled ``part-*.csv`` files with a ``label`` column) has no
    usable order; there every row is its own block and only ``light`` can be trained.
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
    parts, labels_out, blocks_out, seen = [], [], [], {}
    next_block = 0

    def add(label, rows, block_ids):
        parts.append(rows[columns].to_numpy(np.float32))
        labels_out.extend([label] * len(rows))
        blocks_out.append(block_ids)

    if labelled:
        kept = {}
        for i, f in enumerate(files):
            for chunk in pd.read_csv(f, chunksize=200_000):
                chunk.columns = [c.strip() for c in chunk.columns]
                label_col = next(c for c in chunk.columns if c.lower() == 'label')
                for label, rows in chunk.groupby(label_col, sort=True):
                    label_group(label)
                    seen[label] = seen.get(label, 0) + len(rows)
                    room = cap - kept.get(label, 0)
                    if room > 0:
                        rows = rows.iloc[rng.permutation(len(rows))[:room]]
                        add(label, rows, np.arange(next_block, next_block + len(rows)))
                        next_block += len(rows)
                        kept[label] = kept.get(label, 0) + len(rows)
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
            kept = 0
            for f in (group[k] for k in rng.permutation(len(group))):
                frame = pd.read_csv(f)
                frame.columns = [c.strip() for c in frame.columns]
                if [c for c in frame.columns if c not in drop] != columns:
                    raise SystemExit(f'{f}: feature columns differ from {files[0]}')
                seen[label] = seen.get(label, 0) + len(frame)
                n_blocks = max(1, len(frame) // BLOCK)
                take = min(n_blocks, math.ceil(min(quota, cap - kept) / BLOCK))
                for b in sorted(rng.permutation(n_blocks)[:take]):
                    rows = frame.iloc[b * BLOCK:(b + 1) * BLOCK if n_blocks > 1 else len(frame)]
                    add(label, rows, np.full(len(rows), next_block))
                    next_block += 1
                    kept += len(rows)
                done += 1
            print(f'[{done}/{len(files)}] {label}: {kept} rows kept of {seen[label]}')
    X = np.concatenate(parts)
    fine = np.array(labels_out)
    block = np.concatenate(blocks_out).astype(np.int64)
    y = np.array([CLASSES.index(label_group(l)) for l in fine], dtype=np.int64)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, fine=fine, block=block, columns=np.array(columns),
                        ordered=np.array(not labelled))
    labels = sorted(set(labels_out))
    summary = {'files': len(files), 'layout': 'label column' if labelled else 'folder per label',
               'ordered': not labelled, 'block_rows': BLOCK, 'blocks': int(next_block),
               'cap_per_label': cap, 'seed': seed, 'features': len(columns), 'columns': columns,
               'dropped': list(drop), 'rows': int(len(y)), 'seen_per_label': seen,
               'kept_per_label': {l: int((fine == l).sum()) for l in labels},
               'kept_per_class': {c: int((y == k).sum()) for k, c in enumerate(CLASSES)}}
    _write_json(Path(out).with_suffix('.json'), summary)
    print(json.dumps(summary['kept_per_class'], indent=2))
    missing = [c for c in CLASSES if summary['kept_per_class'][c] == 0]
    if missing:
        print(f'WARNING: no rows for classes {missing}; download more CSV files.')


def traffic(raw, out):
    """Records per second of every label: median over rows of packet rate / packets per record.

    The absolute rate depends on the testbed; what the configs use is the ratio of each
    class to benign traffic (median over the labels of a class).
    """
    import numpy as np
    import pandas as pd
    files = sorted(Path(raw).rglob('*.csv'))
    if not files or any(c.strip().lower() == 'label' for c in pd.read_csv(files[0], nrows=0).columns):
        raise SystemExit('traffic needs the 2024 folder-per-label layout')
    per_label = {}
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=['Rate', 'Number'])
        per_label.setdefault(file_label(f), []).append(d['Rate'] / d['Number'].clip(lower=1))
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f'[{i + 1}/{len(files)}] read')
    labels = {l: float(pd.concat(v).median()) for l, v in sorted(per_label.items())}
    by_class = {}
    for l, rate in labels.items():
        by_class.setdefault(label_group(l), []).append(rate)
    class_rate = {c: float(np.median(v)) for c, v in by_class.items()}
    ratio = {c: class_rate[c] / class_rate[BENIGN] for c in class_rate}
    result = {'labels': labels,
              'class_records_per_s': class_rate, 'class_ratio': ratio}
    _write_json(out, result)
    print(json.dumps({c: round(r, 2) for c, r in ratio.items()}, indent=2))


# Models ----------------------------------------------------------------------

def build_model(kind, n_features, mean=None, std=None, context=CONTEXT):
    """``light``: MLP on one record, input (B, F). ``full``: GRU over (B, K, F), last = the flow."""
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
        context = 1

        def __init__(self):
            super().__init__()
            self.norm = Normalize()
            self.net = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, len(CLASSES)))

        def forward(self, x):
            return self.net(self.norm(x))

    class Full(nn.Module):
        """Record embedding, a GRU over the K records, and a head on [GRU state, current record]."""

        def __init__(self):
            super().__init__()
            self.context = context
            self.norm = Normalize()
            self.embed = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU())
            self.gru = nn.GRU(64, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(CLASSES)))

        def forward(self, x):
            h = self.embed(self.norm(x))
            _, last = self.gru(h)
            return self.head(torch.cat([last[-1], h[:, -1]], dim=1))

    return {'light': Light, 'full': Full}[kind]()


def _load_model(path):
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt['kind'], ckpt['n_features'], context=ckpt.get('context', CONTEXT))
    model.load_state_dict(ckpt['state'])
    return model.eval()


def _split_blocks(y, block, seed, fractions=(0.7, 0.15)):
    """Assign whole blocks to train/val/test, stratified by class; returns row masks."""
    import numpy as np
    rng = np.random.default_rng(seed)
    first = np.unique(block, return_index=True)[1]
    part = np.zeros(int(block.max()) + 1, dtype=np.int64)
    for k in np.unique(y):
        ids = rng.permutation(block[first][y[first] == k])
        a = int(round(len(ids) * fractions[0]))
        b = a + int(round(len(ids) * fractions[1]))
        part[ids[a:b]], part[ids[b:]] = 1, 2
    return [part[block] == p for p in range(3)]


class Windows:
    """Builds model inputs for target rows: the row itself, or the row and the
    ``context - 1`` rows before it in the same block. Context records can be
    replaced by random records from ``pool`` with probability ``rho``."""

    def __init__(self, X, block, context, pool):
        import numpy as np
        self.X, self.context, self.pool = X, context, pool
        pos = np.zeros(len(block), dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(block)) + 1]
        lengths = np.diff(np.r_[starts, len(block)])
        pos = np.arange(len(block)) - np.repeat(starts, lengths)
        self.eligible = pos >= context - 1

    def __call__(self, idx, rho=0.0, rng=None):
        import numpy as np
        if self.context == 1:
            return self.X[idx]
        S = self.X[idx[:, None] + np.arange(1 - self.context, 1)[None, :]]
        if rho > 0:
            mask = rng.random(S.shape[:2]) < rho
            mask[:, -1] = False
            S[mask] = self.X[rng.choice(self.pool, int(mask.sum()))]
        return S


def _evaluate(model, windows, idx, y, device, rho=0.0, seed=0):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    preds = []
    with torch.inference_mode():
        for i in range(0, len(idx), 4096):
            x = torch.as_tensor(windows(idx[i:i + 4096], rho, rng), device=device)
            preds.append(model(x).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    y = y[idx]
    out = {'rho': rho, 'class_recall': {}, 'recall': {}}
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


def train(data, kind, seed, epochs, out_dir, train_rho=0.5, eval_rho=0.5):
    import numpy as np
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(data, allow_pickle=False)
    X, y, block = d['X'], d['y'], d['block']
    context = CONTEXT if kind == 'full' else 1
    if context > 1 and not bool(d['ordered']):
        raise SystemExit('The full model needs records in capture order (2024 folder layout).')
    in_tr, in_va, in_te = _split_blocks(y, block, seed)
    # Both models are trained and scored on the same targets: rows with a full context.
    eligible = Windows(X, block, CONTEXT, None).eligible if bool(d['ordered']) else np.ones(len(y), bool)
    tr, va, te = (np.flatnonzero(m & eligible) for m in (in_tr, in_va, in_te))
    windows = Windows(X, block, context, np.flatnonzero(in_tr))
    Xt = np.nan_to_num(X[in_tr], nan=0.0, posinf=0.0, neginf=0.0)
    logx = np.sign(Xt) * np.log1p(np.abs(Xt))
    mean, std = logx.mean(0), logx.std(0)
    std[std < 1e-6] = 1.0
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(kind, X.shape[1], mean.astype(np.float32), std.astype(np.float32), context).to(device)
    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(CLASSES), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ytr = torch.as_tensor(y)
    best, best_state, history = -1.0, None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(tr)
        total = 0.0
        for i in range(0, len(order), 1024):
            b = order[i:i + 1024]
            x = windows(b, rng.uniform(0, train_rho), rng)  # context contamination as augmentation
            opt.zero_grad()
            loss = loss_fn(model(torch.as_tensor(x, device=device)), ytr[b].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        val = _evaluate(model, windows, va, y, device, eval_rho if context > 1 else 0.0, seed)
        history.append({'epoch': epoch + 1, 'loss': total / len(tr), 'val_balanced_accuracy': val['balanced_accuracy']})
        print(f'{kind} epoch {epoch + 1}/{epochs}: loss {total / len(tr):.4f}, '
              f'val balanced accuracy {val["balanced_accuracy"]:.4f}')
        if val['balanced_accuracy'] > best:
            best = val['balanced_accuracy']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    test_clean = _evaluate(model, windows, te, y, device, 0.0, seed + 1)
    test = _evaluate(model, windows, te, y, device, eval_rho, seed + 1) if context > 1 else test_clean
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'kind': kind, 'n_features': int(X.shape[1]), 'context': context, 'state': best_state},
               out_dir / f'{kind}.pt')
    np.save(out_dir / 'timing_inputs.npy', Windows(X, block, CONTEXT, None)(te[:max(TIMING_FLOWS)]))
    params = sum(p.numel() for p in model.parameters())
    result = {'model': kind, 'context': context, 'seed': seed, 'epochs': epochs, 'device': device,
              'parameters': int(params), 'train_rho_max': train_rho if context > 1 else 0.0,
              'train_s': time.perf_counter() - started, 'split_sizes': [len(tr), len(va), len(te)],
              'history': history, 'test': test, 'test_clean_context': test_clean}
    _write_json(out_dir / f'{kind}-metrics.json', result)
    print(json.dumps({'recall': test['recall'], 'benign_fpr': test['benign_false_positive_rate'],
                      'balanced_accuracy': test['balanced_accuracy']}, indent=2))


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
    if model.context == 1:
        X = X[:, -1].contiguous()  # one record per flow
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

def write_configs(work, sample_path, profile_name, ref_mips, edge='rpi4', fog='m2-chip', scales=(1,)):
    timing = _read_json(work / 'timing.json')
    models = []
    for kind in ('light', 'full'):
        t, m = timing['models'][kind], _read_json(work / f'{kind}-metrics.json')
        models.append({'name': kind, 'fixed_mi': round(t['fixed_s'] * ref_mips, 6),
                       'mi_per_flow': round(t['per_flow_s'] * ref_mips, 6),
                       'memory_mb': max(1, math.ceil(t['memory_mb'])), 'load_s': round(t['load_s'], 4),
                       'recall': {c: round(m['test']['recall'].get(c, 0.0), 4) for c in ATTACKS},
                       'context': m.get('context', 1), 'eval_rho': m['test'].get('rho', 0.0),
                       'recall_clean_context': {c: round(m['test_clean_context']['recall'].get(c, 0.0), 4)
                                                for c in ATTACKS} if 'test_clean_context' in m else None,
                       'balanced_accuracy_clean_context': m.get('test_clean_context', m['test'])['balanced_accuracy'],
                       'benign_false_positive_rate_clean_context':
                           m.get('test_clean_context', m['test'])['benign_false_positive_rate'],
                       'fit': {'fixed_s': t['fixed_s'], 'per_flow_s': t['per_flow_s'], 'r2': t['r2']},
                       'benign_false_positive_rate': m['test']['benign_false_positive_rate'],
                       'balanced_accuracy': m['test']['balanced_accuracy'], 'parameters': m['parameters']})
    sample = _read_json(Path(sample_path).with_suffix('.json'))
    profile = {'profile': profile_name, 'ref_mips': ref_mips, 'hardware': timing['hardware'],
               'threads': timing['threads'], 'dataset': {k: sample[k] for k in
                                                          ('files', 'cap_per_label', 'seed', 'features', 'rows',
                                                           'kept_per_class')},
               'models': models,
               'tiers': {tier: {'device': HARDWARE[key][0], 'key': key, 'ratio_to_reference_core':
                                round(HARDWARE[key][1], 4), 'mips': round(ref_mips * HARDWARE[key][1])}
                         for tier, key in (('edge', edge), ('fog', fog))}}
    if (work / 'traffic.json').exists():
        profile['traffic'] = _read_json(work / 'traffic.json')
    else:
        print('No traffic.json: burst intensities stay as in the base configs (run the traffic step).')
    _write_json(ROOT / 'configs' / f'calibration-{profile_name}.json', profile)
    loads = {}
    for name in BASE_CONFIGS:
        base = _read_json(ROOT / 'configs' / f'{name}.json')
        for scale in scales:
            config = calibrated_config(base, profile, sample['features'], scale)
            _write_json(ROOT / 'configs' / f"{config['name']}.json", config)
            if name == 'ids-place-medium':
                loads[config['name']] = offered_load(config)
    _update_report(profile, timing, loads)
    print(f'Wrote configs/calibration-{profile_name}.json and calibrated configs.')


def offered_load(config):
    """Peak and mean one-second utilization if every task ran the full model on its own
    gateway, and peak fog utilization if every task were sent to its fog node."""
    sys.path.insert(0, str(ROOT / 'src'))
    from fogids.generator import generate
    inst = generate(config)
    full = max(inst.models, key=lambda m: m.mi_per_flow)
    edge, fog = {}, {}
    for t in inst.tasks:
        work, second = full.work_mi(t.n_flows), int(t.release_s)
        edge[(t.gateway, second)] = edge.get((t.gateway, second), 0.0) + work
        parent = inst.node(t.gateway).parent
        fog[(parent, second)] = fog.get((parent, second), 0.0) + work
    edge_mips, fog_mips = inst.tier('edge')[0].mips, inst.tier('fog')[0].mips
    return {'tasks': len(inst.tasks), 'edge_peak': max(edge.values()) / edge_mips,
            'edge_mean': sum(edge.values()) / (len(inst.tier('edge')) * inst.horizon_s * edge_mips),
            'fog_peak_all_offloaded': max(fog.values()) / fog_mips}


def _update_report(profile, timing, loads=None):
    hw = profile['hardware']
    cpu = hw.get('machdep.cpu.brand_string') or hw.get('processor') or hw['machine']
    lines = [f"Profile `{profile['profile']}`: {cpu}, {hw['platform']}, Python {hw['python']}, "
             f"PyTorch {hw.get('torch', '?')}, one CPU thread, reference {profile['ref_mips']} MIPS.",
             f"Sample: {profile['dataset']['rows']} rows from {profile['dataset']['files']} CSV files, "
             f"at most {profile['dataset']['cap_per_label']} per original label.", '',
             '| Model | Records | Params | fixed, µs | per flow, µs | R² | fixed_mi | mi_per_flow | load_s | '
             'memory_mb |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for m in profile['models']:
        lines.append(f"| {m['name']} | {m.get('context', 1)} | {m['parameters']} | {m['fit']['fixed_s'] * 1e6:.1f} | "
                     f"{m['fit']['per_flow_s'] * 1e6:.3f} | {m['fit']['r2']:.4f} | {m['fixed_mi']} | "
                     f"{m['mi_per_flow']} | {m['load_s']} | {m['memory_mb']} |")
    rows = []
    for m in profile['models']:
        if m.get('recall_clean_context') and m.get('context', 1) > 1:
            rows.append((f"{m['name']}, clean context", m['recall_clean_context'],
                         m['balanced_accuracy_clean_context'], m['benign_false_positive_rate_clean_context']))
            rows.append((f"{m['name']}, {m['eval_rho']:.0%} of context replaced (used)", m['recall'],
                         m['balanced_accuracy'], m['benign_false_positive_rate']))
        else:
            rows.append((m['name'], m['recall'], m['balanced_accuracy'], m['benign_false_positive_rate']))
    lines += ['', 'Test blocks: detection recall per class, balanced 8-class accuracy, benign false-positive rate:', '',
              '| Model | ' + ' | '.join(ATTACKS) + ' | balanced acc. | benign FPR |',
              '|---|' + '---|' * (len(ATTACKS) + 2)]
    for name, rec, bal, fpr in rows:
        lines.append(f"| {name} | " + ' | '.join(f"{rec[c]:.4f}" for c in ATTACKS) + f" | {bal:.4f} | {fpr:.4f} |")
    if profile.get('tiers'):
        lines += ['', 'Node speeds: ' + '; '.join(f"{t} = {v['device']}, {v['ratio_to_reference_core']} x reference "
                                                 f"core = {v['mips']} MIPS" for t, v in profile['tiers'].items()) + '.']
    if profile.get('traffic'):
        r = profile['traffic']['class_ratio']
        lines += ['', 'Traffic intensity relative to benign (median records/s per class): ' +
                  ', '.join(f"{c} {r[c]:.2f}" for c in CLASSES if c in r) + '.']
    if loads:
        lines += ['', 'Offered load with the full model (one-second bins): peak and mean utilization of a '
                      'gateway that runs all its tasks itself, and peak fog utilization if all tasks go to fog:', '',
                  '| Config | Tasks | Edge peak | Edge mean | Fog peak (all offloaded) |', '|---|---|---|---|---|']
        for name, l in loads.items():
            lines.append(f"| {name} | {l['tasks']} | {l['edge_peak']:.2f} | {l['edge_mean']:.3f} | "
                         f"{l['fog_peak_all_offloaded']:.2f} |")
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
    sys.stdout.reconfigure(line_buffering=True)  # progress stays visible when piped to tee
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('prepare', 'train', 'time', 'traffic', 'configs', 'all'))
    p.add_argument('--raw', type=Path, default=DATA / 'CICIoT2023', help='Directory with CICIoT2023 CSV files')
    # Kept outside the raw-data folder; on case-insensitive file systems data/ciciot2023
    # and data/CICIoT2023 would be the same directory.
    p.add_argument('--sample', type=Path, default=DATA / 'ciciot2023-sample.npz')
    p.add_argument('--cap', type=int, default=20000, help='Maximum rows kept per original label')
    p.add_argument('--drop', nargs='*', default=[], help='Feature columns to exclude, e.g. Time_To_Live IAT')
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--model', choices=('light', 'full', 'both'), default='both')
    p.add_argument('--epochs-light', type=int, default=15)
    p.add_argument('--epochs-full', type=int, default=10)
    p.add_argument('--train-rho', type=float, default=0.5,
                   help='Maximum share of context records replaced by random records during training')
    p.add_argument('--eval-rho', type=float, default=0.5,
                   help='Share of context records replaced when scoring the full model')
    p.add_argument('--min-seconds', type=float, default=0.3, help='Minimum timing duration per window size')
    p.add_argument('--min-repeats', type=int, default=50)
    p.add_argument('--profile', default='mac', help='Hardware profile name')
    p.add_argument('--edge', choices=sorted(HARDWARE), default='rpi4', help='Device modelled as an edge gateway')
    p.add_argument('--fog', choices=sorted(HARDWARE), default='m2-chip', help='Device modelled as a fog node')
    p.add_argument('--load-scales', type=float, nargs='+', default=[1, 3, 10],
                   help='Traffic volume multipliers; one set of calibrated configs per value')
    p.add_argument('--ref-mips', type=float, default=None,
                   help='MIPS of one measuring core (default: fog MIPS of ids-place-small)')
    a = p.parse_args(argv)
    ref_mips = a.ref_mips or _read_json(ROOT / 'configs' / 'ids-place-small.json')['topology']['fog']['mips']
    steps = ('prepare', 'train', 'time', 'traffic', 'configs') if a.step == 'all' else (a.step,)
    for step in steps:
        if step == 'prepare':
            prepare(a.raw, a.sample, a.cap, a.seed, tuple(a.drop))
        elif step == 'train':
            for kind in (('light', 'full') if a.model == 'both' else (a.model,)):
                train(a.sample, kind, a.seed, a.epochs_light if kind == 'light' else a.epochs_full, WORK,
                      a.train_rho, a.eval_rho)
        elif step == 'time':
            measure(WORK, a.min_seconds, a.min_repeats)
        elif step == 'traffic':
            traffic(a.raw, WORK / 'traffic.json')
        else:
            write_configs(WORK, a.sample, a.profile, ref_mips, a.edge, a.fog,
                          tuple(int(x) if x == int(x) else x for x in a.load_scales))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
